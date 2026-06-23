"""Servicio de detección de drift para pronósticos.

Compara las features del request contra la distribución de entrenamiento
extraída del Pipeline serializado en MLflow. El baseline NO requiere
reentrenar: se reconstruye en runtime a partir de los pasos del Pipeline:

  - LagFeatureTransformer.history_                -> KG/HA, FORMATO, FUNDO, FECHA
  - CustomKNNImputer.scaler_.center_/scale_       -> mediana/IQR de numéricas
  - OutlierCapper.lower_/upper_                   -> bounds extremos de numéricas

Si alguno de esos pasos falta (modelo entrenado con otra arquitectura),
las features afectadas se reportan como "sin baseline" y la respuesta
sigue siendo válida en lugar de fallar.

El baseline se cachea por (variety, run_id): se invalida cuando MLflow
publica una nueva versión del modelo (cuando MLflowService.reload_models
detecta `updated`).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp

from app.services.drift_baseline import (
    CATEGORICAL_FEATURES,
    MIN_HISTORY_SAMPLES,
    NUMERIC_FEATURES,
    DriftBaselineExtractor,
    NumericBaseline,
    VarietyBaseline,
)

if TYPE_CHECKING:
    from app.services.mlflow_service import MLflowService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Umbrales (algoritmo, no deployment)
# ---------------------------------------------------------------------------

# z-score sobre numéricas (usa mediana e IQR, robusto a colas largas).
Z_OK_THRESHOLD: float = 1.0
Z_WARNING_THRESHOLD: float = 3.0

# Frecuencia mínima de una categoría para considerarla "habitual".
CATEGORY_RARE_THRESHOLD: float = 0.01  # <1% del baseline = rara

# Population Stability Index: regla de oro de la industria (riesgo crediticio).
PSI_OK_THRESHOLD: float = 0.10
PSI_WARNING_THRESHOLD: float = 0.25

# Tamaño mínimo de muestra para que K-S / Chi² / PSI sean confiables.
MIN_BATCH_FOR_DISTRIBUTION_TESTS: int = 30

# Pequeña constante para evitar log(0) en PSI.
_PSI_EPS: float = 1e-4

# p-value bajo = drift significativo (convención estándar α=0.05).
_PVALUE_ALPHA: float = 0.05

# NUMERIC_FEATURES, CATEGORICAL_FEATURES, MIN_HISTORY_SAMPLES y las
# estructuras de baseline (NumericBaseline, VarietyBaseline) viven ahora en
# `drift_baseline.py` junto al extractor que las produce; aquí se importan.


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DriftService:
    """Calcula el drift entre features de inferencia y el baseline.

    Cache: variety -> (run_id, VarietyBaseline). Se invalida solo cuando
    MLflow publica una nueva versión.
    """

    def __init__(self, mlflow_service: MLflowService) -> None:
        self._mlflow_service = mlflow_service
        self._cache: dict[str, tuple[str, VarietyBaseline]] = {}
        # Extractor stateless: reconstruye el baseline desde el Pipeline.
        self._extractor = DriftBaselineExtractor()
        # `compute`/`compute_batch` corren en threadpool (ver ForecastService):
        # el lock evita que dos requests concurrentes del mismo cold path
        # descarguen y desempaqueten el modelo dos veces.
        self._baseline_lock = threading.Lock()

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def compute(self, variety: str, features_df: pd.DataFrame) -> list[dict[str, Any] | None]:
        """Devuelve un reporte de drift por fila (o None si no hay baseline).

        Diseñado para no romper el flujo de predicción: ante CUALQUIER
        error (modelo sin sklearn flavor, pipeline sin transformers
        esperados, error de cálculo) devuelve `None` y deja que el
        request siga adelante. El error se loguea como warning.
        """
        if features_df.empty:
            return []
        try:
            baseline = self._get_baseline(variety)
        except Exception as exc:
            logger.warning(
                "DriftService: no se pudo construir baseline para '%s': %s",
                variety,
                exc,
                exc_info=True,
            )
            return [None] * len(features_df)

        if baseline is None:
            return [None] * len(features_df)

        try:
            return [self._row_report(baseline, row) for _, row in features_df.iterrows()]
        except Exception as exc:
            logger.warning(
                "DriftService.compute fallo para '%s': %s",
                variety,
                exc,
                exc_info=True,
            )
            return [None] * len(features_df)

    def compute_batch(
        self,
        variety: str,
        features_df: pd.DataFrame,
        *,
        per_row_reports: list[dict[str, Any] | None] | None = None,
    ) -> dict[str, Any] | None:
        """Drift agregado del lote: PSI + K-S (numéricas) + Chi² (categóricas).

        Solo se calcula cuando el lote es suficientemente grande (≥30
        filas) para que los tests sean confiables. Devuelve `None` para:
          - Lotes muy chicos (PSI/K-S/Chi² no son representativos).
          - Variedades sin baseline.
          - Cualquier error inesperado (degrade graceful).
        """
        if len(features_df) < MIN_BATCH_FOR_DISTRIBUTION_TESTS:
            return None
        try:
            baseline = self._get_baseline(variety)
        except Exception as exc:
            logger.warning(
                "DriftService.compute_batch: baseline error '%s': %s",
                variety,
                exc,
                exc_info=True,
            )
            return None
        if baseline is None:
            return None

        try:
            return self._batch_report(baseline, features_df, per_row_reports)
        except Exception as exc:
            logger.warning(
                "DriftService.compute_batch fallo para '%s': %s",
                variety,
                exc,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Construcción de baseline (lazy, una vez por (variety, run_id))
    # ------------------------------------------------------------------

    def _get_baseline(self, variety: str) -> VarietyBaseline | None:
        version_info = self._mlflow_service.get_latest_version_info(variety)
        if not version_info:
            return None

        run_id = version_info["run_id"]
        cached = self._cache.get(variety)
        if cached and cached[0] == run_id:
            return cached[1]

        with self._baseline_lock:
            # Double-check: otro thread pudo construirlo mientras esperábamos.
            cached = self._cache.get(variety)
            if cached and cached[0] == run_id:
                return cached[1]

            model_name = f"{self._mlflow_service.experiment_prefix}{variety}"
            uri = f"models:/{model_name}/{version_info['version']}"
            try:
                sklearn_pipeline = mlflow.sklearn.load_model(uri)
            except Exception as exc:
                logger.warning(
                    "DriftService: no se pudo cargar sklearn flavor de '%s' (%s)."
                    " El reporte de drift quedará deshabilitado para esta variedad.",
                    variety,
                    exc,
                )
                return None

            baseline = self._extractor.extract(variety, run_id, sklearn_pipeline)
            if baseline is None:
                return None
            self._cache[variety] = (run_id, baseline)
        logger.info(
            "📐 Drift baseline cargado para '%s' | n=%d | rango FECHA=%s..%s | "
            "numéricas=%d | categóricas=%d",
            variety,
            baseline.n_samples,
            baseline.date_from,
            baseline.date_to,
            len(baseline.numeric),
            len(baseline.categorical),
        )
        return baseline

    # ------------------------------------------------------------------
    # Reporte por fila
    # ------------------------------------------------------------------

    def _row_report(
        self,
        baseline: VarietyBaseline,
        row: pd.Series,
    ) -> dict[str, Any]:
        per_feature: list[dict[str, Any]] = []
        worst_status = "ok"
        score_acc = 0.0
        n_features = 0

        for col in NUMERIC_FEATURES:
            nb = baseline.numeric.get(col)
            if nb is None:
                continue
            value = row.get(col)
            # Si la feature es opcional (%INDUS, P/BAYA) y no vino en el
            # request, la mostramos igual con "no enviado" para que el
            # panel exponga el rango histórico — el usuario ve qué valor
            # esperaría el modelo si decidiera incluirla.
            if value is None or pd.isna(value):
                per_feature.append(
                    {
                        "feature": col,
                        "value": None,
                        "value_str": "no enviado",
                        "baseline_median": nb.center,
                        "baseline_iqr": nb.scale,
                        "baseline_p05": nb.p05,
                        "baseline_p95": nb.p95,
                        "baseline_freq": None,
                        "z_score": None,
                        "is_unseen_category": False,
                        "status": "ok",
                        "source": nb.source,
                    }
                )
                continue
            value_f = float(value)
            z = (value_f - nb.center) / nb.scale
            abs_z = abs(z)
            if abs_z >= Z_WARNING_THRESHOLD:
                status = "alert"
            elif abs_z >= Z_OK_THRESHOLD:
                status = "warning"
            else:
                status = "ok"
            worst_status = self._merge_status(worst_status, status)
            per_feature.append(
                {
                    "feature": col,
                    "value": value_f,
                    "value_str": None,
                    "baseline_median": nb.center,
                    "baseline_iqr": nb.scale,
                    "baseline_p05": nb.p05,
                    "baseline_p95": nb.p95,
                    "baseline_freq": None,
                    "z_score": float(z),
                    "is_unseen_category": False,
                    "status": status,
                    "source": nb.source,
                }
            )
            score_acc += min(abs_z / Z_WARNING_THRESHOLD, 1.0)
            n_features += 1

        for col in CATEGORICAL_FEATURES:
            cb = baseline.categorical.get(col)
            if cb is None:
                continue
            value = row.get(col)
            if value is None or pd.isna(value):
                continue
            value_str = str(value)
            freq = cb.get(value_str, 0.0)
            if freq <= 0.0:
                status = "alert"
                contribution = 1.0
            elif freq < CATEGORY_RARE_THRESHOLD:
                status = "warning"
                contribution = 0.5
            else:
                status = "ok"
                contribution = 0.0
            worst_status = self._merge_status(worst_status, status)
            per_feature.append(
                {
                    "feature": col,
                    "value": None,
                    "value_str": value_str,
                    "baseline_median": None,
                    "baseline_iqr": None,
                    "baseline_p05": None,
                    "baseline_p95": None,
                    "baseline_freq": freq,
                    "z_score": None,
                    "is_unseen_category": (freq <= 0.0),
                    "status": status,
                    "source": "history",
                }
            )
            score_acc += contribution
            n_features += 1

        score = score_acc / max(n_features, 1)

        return {
            "score": float(score),
            "status": worst_status,
            "verdict": self._verdict(worst_status, baseline),
            "training_window": {
                "start": baseline.date_from,
                "end": baseline.date_to,
                "n_samples": baseline.n_samples,
            },
            "per_feature": per_feature,
        }

    @staticmethod
    def _merge_status(a: str, b: str) -> str:
        rank = {"ok": 0, "warning": 1, "alert": 2}
        return a if rank[a] >= rank[b] else b

    @staticmethod
    def _verdict(status: str, baseline: VarietyBaseline) -> str:
        n = f"{baseline.n_samples:,}" if baseline.n_samples else "—"
        window = ""
        if baseline.date_from and baseline.date_to:
            window = f" ({baseline.date_from} a {baseline.date_to})"
        if status == "ok":
            return (
                f"Predicción confiable: las condiciones del pronóstico están "
                f"dentro del rango histórico de {n} cosechas{window}."
            )
        if status == "warning":
            return (
                "Confianza moderada: alguna variable está en el límite del "
                "rango habitual. Validar con criterio agronómico."
            )
        return (
            "Predicción extrapolando: una o más variables están fuera del "
            "rango histórico o corresponden a una categoría no vista en "
            "entrenamiento. Revisar antes de tomar decisiones."
        )

    # ------------------------------------------------------------------
    # Reporte agregado del lote (PSI + K-S + Chi²)
    # ------------------------------------------------------------------

    def _batch_report(
        self,
        baseline: VarietyBaseline,
        features_df: pd.DataFrame,
        per_row_reports: list[dict[str, Any] | None] | None,
    ) -> dict[str, Any]:
        per_feature: list[dict[str, Any]] = []
        psi_sum = 0.0
        psi_count = 0
        worst_status = "ok"

        # Numéricas: PSI (siempre que haya baseline) + K-S (solo si hay
        # samples crudos en baseline, hoy solo KG/HA).
        for col in NUMERIC_FEATURES:
            nb = baseline.numeric.get(col)
            if nb is None or col not in features_df.columns:
                continue
            values = features_df[col].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue

            psi, psi_status = self._compute_psi_numeric(values, nb)
            ks_stat, ks_pval = self._compute_ks(values, nb)

            method_parts = ["psi"]
            if ks_stat is not None:
                method_parts.append("ks")

            # Combinación: PSI domina; K-S puede subir el status si detecta
            # diferencia significativa que PSI no capturó (por ejemplo,
            # corrimiento de la mediana sin cambio de bins).
            feature_status = psi_status
            if ks_pval is not None and ks_pval < _PVALUE_ALPHA and feature_status == "ok":
                feature_status = "warning"
            worst_status = self._merge_status(worst_status, feature_status)

            per_feature.append(
                {
                    "feature": col,
                    "kind": "numeric",
                    "psi": float(psi),
                    "psi_status": psi_status,
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_pval,
                    "chi2_statistic": None,
                    "chi2_pvalue": None,
                    "method": "+".join(method_parts),
                    "status": feature_status,
                    "n_baseline": (int(len(nb.samples)) if nb.samples is not None else None),
                    "n_observed": int(len(values)),
                    "source": nb.source,
                }
            )
            psi_sum += min(psi, 1.0)  # capped para que un PSI gigante no domine
            psi_count += 1

        # Categóricas: PSI + Chi² (goodness-of-fit) cuando hay >=2 categorías
        # en común con el baseline.
        for col in CATEGORICAL_FEATURES:
            cb = baseline.categorical.get(col)
            if cb is None or col not in features_df.columns:
                continue
            observed_counts = features_df[col].dropna().astype(str).value_counts()
            if observed_counts.empty:
                continue

            psi, psi_status, n_unseen = self._compute_psi_categorical(
                observed_counts,
                cb,
            )
            chi2_stat, chi2_pval = self._compute_chi2(observed_counts, cb)

            method_parts = ["psi"]
            if chi2_stat is not None:
                method_parts.append("chi2")

            feature_status = psi_status
            if chi2_pval is not None and chi2_pval < _PVALUE_ALPHA and feature_status == "ok":
                feature_status = "warning"
            if n_unseen > 0:
                # Categoría no vista siempre eleva el estado a alert: es
                # señal categórica de drift estructural.
                feature_status = "alert"
            worst_status = self._merge_status(worst_status, feature_status)

            per_feature.append(
                {
                    "feature": col,
                    "kind": "categorical",
                    "psi": float(psi),
                    "psi_status": psi_status,
                    "ks_statistic": None,
                    "ks_pvalue": None,
                    "chi2_statistic": chi2_stat,
                    "chi2_pvalue": chi2_pval,
                    "method": "+".join(method_parts),
                    "status": feature_status,
                    "n_baseline": baseline.n_samples or None,
                    "n_observed": int(observed_counts.sum()),
                    "source": "history",
                    "unseen_categories": n_unseen,
                }
            )
            psi_sum += min(psi, 1.0)
            psi_count += 1

        score = psi_sum / max(psi_count, 1)

        # Conteo por estado de las filas individuales (si vienen).
        row_counts = {"ok": 0, "warning": 0, "alert": 0}
        if per_row_reports:
            for r in per_row_reports:
                if r is None:
                    continue
                s = r.get("status", "ok")
                if s in row_counts:
                    row_counts[s] += 1

        return {
            "n_rows": int(len(features_df)),
            "score": float(score),
            "status": worst_status,
            "verdict": self._batch_verdict(
                worst_status,
                baseline,
                len(features_df),
            ),
            "training_window": {
                "start": baseline.date_from,
                "end": baseline.date_to,
                "n_samples": baseline.n_samples,
            },
            "per_feature": per_feature,
            "row_status_counts": row_counts,
        }

    @staticmethod
    def _compute_psi_numeric(
        observed_values: np.ndarray,
        nb: NumericBaseline,
    ) -> tuple[float, str]:
        """PSI numérico con bins por percentiles del baseline.

        Edges = (-inf, p05, p25, p50, p75, p95, +inf) -> 6 bins.
        Si hay samples reales del baseline, calcula frecuencias reales
        por bin; si no, asume el reparto teórico [.05, .20, .25, .25, .20, .05].
        """
        edges = np.array(
            [-np.inf, nb.p05, nb.p25, nb.center, nb.p75, nb.p95, np.inf],
            dtype=float,
        )
        # Defensivo: edges deben ser monotónicamente crecientes. Cuando una
        # feature tiene varianza muy chica (todo el batch igual), p25=p75
        # y np.histogram falla. Forzamos monotonicidad acumulada.
        edges = np.maximum.accumulate(edges)
        # Y eliminamos duplicados internos (bins de tamaño 0 colapsan).
        unique_edges, _ = np.unique(edges, return_index=True)
        if len(unique_edges) < 3:
            # Pipeline degenerado: no hay bins distinguibles.
            return 0.0, "ok"
        edges = unique_edges

        if nb.samples is not None and len(nb.samples) >= MIN_HISTORY_SAMPLES:
            baseline_counts, _ = np.histogram(nb.samples, bins=edges)
            total = baseline_counts.sum()
            if total <= 0:
                return 0.0, "ok"
            baseline_freqs = baseline_counts.astype(float) / total
        else:
            # Reparto teórico para 6 bins. Si los edges colapsaron, repartimos
            # uniformemente entre los bins resultantes.
            n_bins = len(edges) - 1
            theoretical = np.array(
                [0.05, 0.20, 0.25, 0.25, 0.20, 0.05],
                dtype=float,
            )
            baseline_freqs = theoretical if n_bins == 6 else np.full(n_bins, 1.0 / n_bins)

        observed_counts, _ = np.histogram(observed_values, bins=edges)
        n_obs = observed_counts.sum()
        if n_obs <= 0:
            return 0.0, "ok"
        observed_freqs = observed_counts.astype(float) / n_obs

        b = np.maximum(baseline_freqs, _PSI_EPS)
        o = np.maximum(observed_freqs, _PSI_EPS)
        psi = float(np.sum((o - b) * np.log(o / b)))
        return psi, _psi_status(psi)

    @staticmethod
    def _compute_psi_categorical(
        observed_counts: pd.Series,
        baseline_freqs: dict[str, float],
    ) -> tuple[float, str, int]:
        """PSI categórico + cuenta de categorías no vistas en baseline."""
        n_obs = int(observed_counts.sum())
        if n_obs <= 0:
            return 0.0, "ok", 0
        all_cats = set(observed_counts.index) | set(baseline_freqs)
        psi = 0.0
        n_unseen = 0
        for cat in all_cats:
            b = max(baseline_freqs.get(cat, 0.0), _PSI_EPS)
            o = max(float(observed_counts.get(cat, 0)) / n_obs, _PSI_EPS)
            if baseline_freqs.get(cat, 0.0) <= 0.0 and observed_counts.get(cat, 0) > 0:
                n_unseen += 1
            psi += (o - b) * np.log(o / b)
        return float(psi), _psi_status(psi), n_unseen

    @staticmethod
    def _compute_ks(
        observed_values: np.ndarray,
        nb: NumericBaseline,
    ) -> tuple[float | None, float | None]:
        """K-S de dos muestras. Solo aplicable si baseline guarda samples."""
        if (
            nb.samples is None
            or len(nb.samples) < MIN_HISTORY_SAMPLES
            or len(observed_values) < MIN_HISTORY_SAMPLES
        ):
            return None, None
        try:
            result = ks_2samp(nb.samples, observed_values, method="auto")
            return float(result.statistic), float(result.pvalue)
        except Exception as exc:
            logger.debug("K-S falló: %s", exc)
            return None, None

    @staticmethod
    def _compute_chi2(
        observed_counts: pd.Series,
        baseline_freqs: dict[str, float],
    ) -> tuple[float | None, float | None]:
        """Chi² goodness-of-fit alineando categorías con baseline.

        Solo evalúa categorías presentes en el baseline; las no vistas se
        reportan vía PSI (n_unseen). Devuelve (None, None) si quedan <2
        categorías comunes — el test no es definible.
        """
        common_cats = [c for c in observed_counts.index if baseline_freqs.get(c, 0.0) > 0.0]
        if len(common_cats) < 2:
            return None, None
        obs = np.array(
            [float(observed_counts[c]) for c in common_cats],
            dtype=float,
        )
        expected_p = np.array(
            [baseline_freqs[c] for c in common_cats],
            dtype=float,
        )
        # Renormalizamos para que sum(expected) == sum(observed) (requisito
        # de scipy.stats.chisquare). Las masas omitidas (categorías no
        # vistas en baseline) ya se contaron en PSI.
        if expected_p.sum() <= 0:
            return None, None
        expected = expected_p / expected_p.sum() * obs.sum()
        if (expected <= 0).any():
            return None, None
        try:
            result = chisquare(f_obs=obs, f_exp=expected)
            return float(result.statistic), float(result.pvalue)
        except Exception as exc:
            logger.debug("Chi² falló: %s", exc)
            return None, None

    @staticmethod
    def _batch_verdict(
        status: str,
        baseline: VarietyBaseline,
        n_rows: int,
    ) -> str:
        n_base = f"{baseline.n_samples:,}" if baseline.n_samples else "—"
        if status == "ok":
            return (
                f"Lote estable: {n_rows} pronósticos comparados contra "
                f"{n_base} cosechas históricas — todas las variables dentro "
                "de la distribución de entrenamiento (PSI<0.10)."
            )
        if status == "warning":
            return (
                f"Drift moderado en el lote ({n_rows} filas): alguna "
                "variable cambió de distribución (PSI 0.10–0.25 o p-value "
                "K-S/Chi² <0.05). Validar antes de presentar a gerencia."
            )
        return (
            f"Drift severo en el lote ({n_rows} filas): una o más variables "
            "tienen distribución incompatible con el entrenamiento (PSI≥0.25 "
            "o categoría no vista). Reentrenar antes de tomar decisiones."
        )


def _psi_status(psi: float) -> str:
    """Regla de oro PSI (riesgo crediticio, McKinley 2007)."""
    if psi < PSI_OK_THRESHOLD:
        return "ok"
    if psi < PSI_WARNING_THRESHOLD:
        return "warning"
    return "alert"
