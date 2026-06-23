"""Extracción del baseline de drift desde el Pipeline serializado en MLflow.

Responsabilidad única (SRP): dado el `Pipeline` sklearn entrenado,
reconstruir el baseline de entrenamiento (estadísticas robustas numéricas
+ frecuencias categóricas) que `DriftService` compara contra las features
de inferencia. NO sabe de MLflow, de cache ni de los tests estadísticos:
recibe un pipeline ya cargado y devuelve un `VarietyBaseline`.

El baseline se reconstruye en runtime a partir de los pasos del Pipeline,
sin reentrenar:

  - LagFeatureTransformer.history_                -> KG/HA, FORMATO, FUNDO, FECHA
  - CustomKNNImputer.scaler_.center_/scale_       -> mediana/IQR de numéricas
  - OutlierCapper.lower_/upper_                   -> bounds extremos de numéricas

Si alguno de esos pasos falta (modelo entrenado con otra arquitectura),
las features afectadas se reportan como "sin baseline" y el reporte de
drift sigue siendo válido en lugar de fallar.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Features del baseline (compartidas con DriftService)
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: tuple[str, ...] = (
    "KG/HA",
    "%INDUS",
    "DPC",
    "P/BAYA",
    "HA",
    "DIA_COSECHA",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("FORMATO", "FUNDO")


# Mínimo de filas históricas para considerar baseline numérico de KG/HA
# desde history_ (por debajo cae al fallback con scaler_).
MIN_HISTORY_SAMPLES: int = 30

# Tope de muestras del baseline a guardar en memoria para K-S. Sub-muestreo
# determinístico cuando history_ tiene >50k filas — K-S converge mucho
# antes y guardar 50k floats por variedad es suficiente (~400KB).
_MAX_BASELINE_SAMPLES: int = 50_000


# ---------------------------------------------------------------------------
# Estructuras de baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericBaseline:
    """Estadísticas numéricas robustas + bounds + (opcional) muestras crudas.

    `samples` se popula solo cuando hay datos reales en `history_` (KG/HA).
    Cuando viene de `RobustScaler`, la columna no tiene muestras y K-S queda
    deshabilitado para esa feature; PSI cae a baseline asumido por
    percentiles. `p25`/`p75` se derivan de IQR cuando vienen del scaler.
    """

    center: float  # = p50 (mediana)
    scale: float  # = IQR (p75 - p25), siempre > 0
    p05: float
    p25: float
    p75: float
    p95: float
    source: str  # "history" | "scaler"
    samples: np.ndarray | None = None


@dataclass(frozen=True)
class VarietyBaseline:
    variety: str
    run_id: str
    n_samples: int
    date_from: str
    date_to: str
    numeric: dict[str, NumericBaseline] = field(default_factory=dict)
    categorical: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class DriftBaselineExtractor:
    """Reconstruye un `VarietyBaseline` a partir de un Pipeline sklearn.

    Stateless: una sola instancia puede servir a todas las variedades.
    """

    def extract(
        self,
        variety: str,
        run_id: str,
        pipeline: Any,
    ) -> VarietyBaseline | None:
        history_df: pd.DataFrame | None = None
        scaler = None
        scaler_cols: list[str] | None = None
        outlier_lower: dict[str, float] | None = None
        outlier_upper: dict[str, float] | None = None

        for step in self._walk_pipeline_steps(pipeline):
            # LagFeatureTransformer
            history_attr = getattr(step, "history_", None)
            if isinstance(history_attr, pd.DataFrame) and history_df is None:
                history_df = history_attr

            # CustomKNNImputer (tiene scaler_ con center_ y scale_)
            scaler_attr = getattr(step, "scaler_", None)
            if scaler_attr is not None and scaler is None:
                center = getattr(scaler_attr, "center_", None)
                scale = getattr(scaler_attr, "scale_", None)
                if center is not None and scale is not None:
                    scaler = scaler_attr
                    scaler_cols = (
                        list(getattr(step, "_knn_fit_cols_", []))
                        or list(getattr(step, "numeric_cols_", []))
                        or None
                    )

            # OutlierCapper
            lower_attr = getattr(step, "lower_", None)
            upper_attr = getattr(step, "upper_", None)
            if (
                isinstance(lower_attr, dict)
                and isinstance(upper_attr, dict)
                and outlier_lower is None
            ):
                outlier_lower = {str(k): float(v) for k, v in lower_attr.items()}
                outlier_upper = {str(k): float(v) for k, v in upper_attr.items()}

        numeric_baselines = self._build_numeric_baselines(
            history_df,
            scaler,
            scaler_cols,
            outlier_lower,
            outlier_upper,
        )
        categorical_baselines = self._build_categorical_baselines(history_df)

        n_samples = int(len(history_df)) if history_df is not None else 0
        date_from, date_to = self._extract_date_window(history_df)

        if not numeric_baselines and not categorical_baselines:
            logger.warning(
                "DriftService: pipeline de '%s' no expuso baseline alguno;"
                " drift quedará deshabilitado.",
                variety,
            )
            return None

        return VarietyBaseline(
            variety=variety,
            run_id=run_id,
            n_samples=n_samples,
            date_from=date_from,
            date_to=date_to,
            numeric=numeric_baselines,
            categorical=categorical_baselines,
        )

    @staticmethod
    def _walk_pipeline_steps(pipeline: Any) -> Iterable[Any]:
        """Itera (no recursivo) sobre todos los estimators del pipeline.

        Soporta:
          - `sklearn.pipeline.Pipeline` (.named_steps / .steps)
          - `ColumnTransformer` (.transformers_)
          - `FeatureUnion` (.transformer_list)
          - `OOFEnsembleRegressor` (.models_ — lista de Pipelines fiteados,
            usamos solo el primero porque los K modelos son refits sobre
            ~(K-1)/K del mismo dataset; sus baselines son ~iguales y
            tomar el primero evita duplicar trabajo).
          - `base_pipeline` (referencia al pipeline interno antes del fit).

        No falla si el objeto no es un pipeline: simplemente lo emite tal
        cual.
        """
        seen: set[int] = set()
        stack: list[Any] = [pipeline]
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            yield obj

            # OOFEnsembleRegressor.models_ — lista de Pipelines fiteados.
            # Usamos solo el primer modelo del ensemble (representativo).
            models_ = getattr(obj, "models_", None)
            if isinstance(models_, list) and models_:
                stack.append(models_[0])

            # Pipeline.named_steps (Bunch) es dict-like, .steps es lista.
            named_steps = getattr(obj, "named_steps", None)
            if isinstance(named_steps, dict):
                stack.extend(named_steps.values())
            else:
                steps = getattr(obj, "steps", None)
                if isinstance(steps, list):
                    for entry in steps:
                        if isinstance(entry, tuple) and len(entry) >= 2:
                            stack.append(entry[1])

            # ColumnTransformer
            transformers_ = getattr(obj, "transformers_", None)
            if isinstance(transformers_, list):
                for entry in transformers_:
                    if isinstance(entry, tuple) and len(entry) >= 2:
                        stack.append(entry[1])

            # FeatureUnion
            transformer_list = getattr(obj, "transformer_list", None)
            if isinstance(transformer_list, list):
                for entry in transformer_list:
                    if isinstance(entry, tuple) and len(entry) >= 2:
                        stack.append(entry[1])

    def _build_numeric_baselines(
        self,
        history_df: pd.DataFrame | None,
        scaler: Any,
        scaler_cols: list[str] | None,
        outlier_lower: dict[str, float] | None,
        outlier_upper: dict[str, float] | None,
    ) -> dict[str, NumericBaseline]:
        baselines: dict[str, NumericBaseline] = {}

        # 1) Desde history_ (datos reales): solo KG/HA está disponible.
        if (
            history_df is not None
            and "KG/HA" in history_df.columns
            and len(history_df) >= MIN_HISTORY_SAMPLES
        ):
            values = history_df["KG/HA"].dropna().to_numpy(dtype=float)
            if len(values) >= MIN_HISTORY_SAMPLES:
                p05, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
                iqr = max(float(p75 - p25), 1e-9)
                # Sub-muestreo determinístico para K-S (cap en 50k filas).
                if len(values) > _MAX_BASELINE_SAMPLES:
                    rng = np.random.default_rng(42)
                    samples = rng.choice(
                        values,
                        size=_MAX_BASELINE_SAMPLES,
                        replace=False,
                    )
                else:
                    samples = values
                baselines["KG/HA"] = NumericBaseline(
                    center=float(p50),
                    scale=iqr,
                    p05=float(p05),
                    p25=float(p25),
                    p75=float(p75),
                    p95=float(p95),
                    source="history",
                    samples=samples,
                )

        # 2) Resto de numéricas: derivadas de RobustScaler + OutlierCapper.
        #    Sin muestras crudas -> K-S no aplica; PSI usa baseline asumido.
        if scaler is not None and scaler_cols:
            center_arr = np.asarray(scaler.center_, dtype=float).ravel()
            scale_arr = np.asarray(scaler.scale_, dtype=float).ravel()
            for col in NUMERIC_FEATURES:
                if col in baselines or col not in scaler_cols:
                    continue
                idx = scaler_cols.index(col)
                if idx >= len(center_arr) or idx >= len(scale_arr):
                    continue
                center = float(center_arr[idx])
                iqr = float(scale_arr[idx])
                if iqr <= 0:
                    iqr = 1e-9
                p25 = center - iqr / 2.0
                p75 = center + iqr / 2.0
                # OutlierCapper bounds son Q1-3*IQR / Q3+3*IQR por defecto;
                # usamos como p05/p95 aproximados (extremos visibles en
                # entrenamiento sin ser outliers).
                if outlier_lower and col in outlier_lower:
                    p05 = float(outlier_lower[col])
                else:
                    p05 = center - 1.65 * iqr  # ~p05 en distribución normal
                if outlier_upper and col in outlier_upper:
                    p95 = float(outlier_upper[col])
                else:
                    p95 = center + 1.65 * iqr
                baselines[col] = NumericBaseline(
                    center=center,
                    scale=iqr,
                    p05=p05,
                    p25=p25,
                    p75=p75,
                    p95=p95,
                    source="scaler",
                    samples=None,
                )

        return baselines

    @staticmethod
    def _build_categorical_baselines(
        history_df: pd.DataFrame | None,
    ) -> dict[str, dict[str, float]]:
        if history_df is None:
            return {}
        out: dict[str, dict[str, float]] = {}
        for col in CATEGORICAL_FEATURES:
            if col not in history_df.columns:
                continue
            counts = history_df[col].value_counts(normalize=True, dropna=False)
            freqs = {str(k): float(v) for k, v in counts.items()}
            if freqs:
                out[col] = freqs
        return out

    @staticmethod
    def _extract_date_window(history_df: pd.DataFrame | None) -> tuple[str, str]:
        if history_df is None or "FECHA" not in history_df.columns:
            return "", ""
        try:
            fechas = pd.to_datetime(history_df["FECHA"], errors="coerce").dropna()
        except Exception:
            return "", ""
        if fechas.empty:
            return "", ""
        return str(fechas.min().date()), str(fechas.max().date())
