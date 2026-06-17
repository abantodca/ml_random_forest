"""Selecciona el modelo CAMPEON entre varios entrenados para la misma variedad.

Cada modelo (XGB, LGB, ...) entrena de forma INDEPENDIENTE: su propio Optuna
study, su propio search space, su propio MLflow run. Cuando todos terminan,
comparamos sus metricas con un criterio LEX-ORDER (prioridad estricta) que
refleja el contrato de MLOps:

    1. GATE DE OVERFITTING: gap_rel = |gap|/MAE_test <= CHAMPION_MAX_GAP_REL.
       El gap es una RESTRICCION (descalifica modelos rotos), NO un objetivo
       a minimizar. Minimizar gap como criterio primario premiaba al modelo
       mas subajustado, no al que mejor predice (revision 2026-06-10).
       Relativo desde 2026-06-11: comparable entre variedades de escala
       distinta (el |gap|*100 viejo eran kilos disfrazados de pp).
    2. GENERALIZACION: menor MAPE OOF de negocio (cada fila predicha por un
       modelo que NO la vio en train). Es la metrica honesta de produccion.
    3. EFICIENCIA: menor tiempo de entrenamiento ante empate practico
       (`OOF_MAPE_TIE_TOLERANCE`).

El `full_mape` (in-sample, refit + predict all) se conserva como metrica
INFORMATIVA en ranking/dashboards, pero ya no participa en la decision:
es optimista por construccion y premiaba memorizar el train.

Adicionalmente exponemos `composite_score` (legacy) que combina MAPE de
negocio + penalizacion de gap. Lo dejamos como metrica auxiliar para logs y
MLflow tags, pero el campeon NO lo usa para la decision.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.config import (
    CHAMPION_MAX_GAP,
    CHAMPION_MAX_GAP_REL,
    OOF_MAPE_TIE_TOLERANCE,
)

if TYPE_CHECKING:
    from src.step_06_track.business_validation import BusinessValidation

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Resultado de entrenar UN modelo para UNA variedad.

    Campos clave para la decision (poblados por `single_run`):
      - metrics['nested_cv_gap_mean'] : Train-Test gap (overfitting).
      - full_metrics['mape']          : MAPE en KG/JR sobre dataset COMPLETO
                                        (refit + predict all). Estabilidad.
      - elapsed_seconds               : tiempo total de entrenamiento.

    Campos enriquecidos para los renderers (poblados por `single_run` justo
    despues del fit y consumidos UNA VEZ en `variety_runner` para construir
    el dashboard / Excel del campeon):
      - business_validation : BusinessValidation con OOF/in-sample en KG/JR.
      - full_metrics_h      : metricas in-sample en KG/JR_H (unidad modelo).
      - oof_y_true/pred     : arrays OOF en KG/JR_H (alineados con X).

    Estos campos pueden ser None cuando el modelo se reconstruye desde
    `variety_summary_*.json` (sin reabrir el pipeline). Los renderers
    deben tolerar None.
    """

    model_type: str
    metrics: dict[str, float]
    best_params: dict[str, object]
    mlflow_run_id: str
    pipeline_path: str
    elapsed_seconds: float
    business_metrics_oof: dict[str, float] | None = None
    full_metrics: dict[str, float] | None = None  # KG/JR aplicado a TODO X
    business_validation: BusinessValidation | None = None
    full_metrics_h: dict[str, float] | None = None  # KG/JR_H in-sample
    oof_y_true: np.ndarray | None = None
    oof_y_pred: np.ndarray | None = None
    # URI del Logged Model devuelto por mlflow.sklearn.log_model en MLflow 3.x
    # (formato `models:/m-<id>`). Se pasa a register_model para evitar el
    # fallback "no artifacts at artifact path 'model_pipeline'". None si el
    # log_pipeline absorbio el run inactivo o si el resultado se reconstruye
    # desde variety_summary_*.json (caso post-mortem).
    model_uri: str | None = None
    composite_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.composite_score = composite_score(self.metrics, self.business_metrics_oof)

    @property
    def abs_gap(self) -> float:
        return abs(float(self.metrics.get("nested_cv_gap_mean", 0.0)))

    @property
    def gap_rel(self) -> float:
        """Gap relativo: |MAE_test - MAE_train| / MAE_test (adimensional).

        Comparable entre variedades con targets de escala distinta; es la
        unidad del gate del campeon (CHAMPION_MAX_GAP_REL). MAE_test
        invalido (<=0 o ausente) -> inf (el gate falla, conservador).
        """
        mae_test = float(self.metrics.get("nested_cv_mae_mean", 0.0))
        if not math.isfinite(mae_test) or mae_test <= 0:
            return float("inf")
        return self.abs_gap / mae_test

    @property
    def full_mape(self) -> float:
        """MAPE en KG/JR sobre el dataset completo (mas bajo es mejor).

        IN-SAMPLE: refit + predict en TODO X (incluyendo train). Mide
        estabilidad del modelo de produccion, no generalizacion. Lo usa
        `select_champion` como criterio de desempate de "estabilidad".

        Si no hay metricas full disponibles, cae al MAPE OOF de negocio.
        Si tampoco, devuelve infinito (deja al modelo en ultimo lugar).
        """
        if self.full_metrics and "mape" in self.full_metrics:
            return float(self.full_metrics["mape"])
        # NOTA: este fallback OOF->full mezcla metricas heterogeneas (in-sample vs
        # out-of-fold) dentro del lex-order de `_decision_key`. Esta decision esta
        # BAJO REVISION; se conserva el comportamiento para no romper modelos
        # antiguos sin `full_metrics`. Logueamos un WARNING para visibilidad.
        if self.business_metrics_oof and "mape" in self.business_metrics_oof:
            logger.warning(
                "champion: full_mape fallback to OOF for model=%s "
                "(full_metrics missing)",
                self.model_type,
            )
            return float(self.business_metrics_oof["mape"])
        return float("inf")

    @property
    def oof_mape(self) -> float:
        """MAPE en KG/JR sobre predicciones OUT-OF-FOLD (honesto).

        Cada fila se predice con un modelo que NO la vio en train -> mide
        generalizacion real. Es la metrica correcta para el quality gate
        (CHAMPION_MAX_MAPE), porque `full_mape` es in-sample y subestima el
        error de produccion.

        Si no hay metricas OOF disponibles devuelve infinito (gate falla).
        No cae a `full_mape` para evitar que el gate se relaje silenciosamente
        cuando el OOF no se calculo.
        """
        if self.business_metrics_oof and "mape" in self.business_metrics_oof:
            return float(self.business_metrics_oof["mape"])
        return float("inf")


def composite_score(
    metrics: dict[str, float],
    business_metrics_oof: dict[str, float] | None = None,
    gap_weight: float = 0.05,
) -> float:
    """[LEGACY] Score auxiliar 'menor es mejor'. NO usar en codigo nuevo.

    Lo conservamos porque MLflow ya lo loguea como tag y los dashboards
    historicos lo leen. La decision del campeon usa lex-order en
    `select_champion`, NO este score.

    Single point of compute: solo se invoca desde `ModelResult.__post_init__`.
    El resto del codigo lee `r.composite_score` (atributo cacheado), nunca
    re-llama esta funcion.
    """
    gap = max(0.0, float(metrics.get("nested_cv_gap_mean", 0.0)))

    if business_metrics_oof and "mape" in business_metrics_oof:
        mape = float(business_metrics_oof["mape"])
        return mape + gap_weight * gap

    mae = float(metrics.get("nested_cv_mae_mean", float("inf")))
    return mae + 0.5 * gap


def _gap_gate_failed(r: ModelResult) -> bool:
    """True si el gap RELATIVO del modelo supera el gate de overfitting.

    gap_rel = (MAE_test - MAE_train) / MAE_test — adimensional, comparable
    entre variedades (fix 2026-06-11; el gate viejo |gap|*100 eran kilos
    disfrazados de pp: laxo para targets chicos, asfixiante para grandes).
    Un modelo que falla el gate solo gana si TODOS fallan (en cuyo caso el
    quality gate posterior decide si se registra o no).
    """
    return r.gap_rel > CHAMPION_MAX_GAP_REL


def _decision_key(r: ModelResult) -> tuple:
    """Llave lex-order para `min(...)`.

    Orden de prioridad:
      1. Gate de |gap| (0 = pasa, 1 = falla). Restriccion, no objetivo.
      2. Bucket de MAPE OOF (generalizacion honesta; empata si difiere
         < OOF_MAPE_TIE_TOLERANCE).
      3. Tiempo de entrenamiento (eficiencia).
    """
    # Bucketing por floor (`int`) en vez de `round`: dos valores que difieren
    # en exactamente `tol` deben considerarse empate (intencion documentada).
    # `round` los separaba en buckets adyacentes en la frontera; `int` (floor)
    # los agrupa de forma consistente con "diferencia < tol => empate".
    # oof_mape puede ser inf (sin metricas OOF): bucket centinela enorme para
    # mandarlo al final sin OverflowError de int(inf).
    mape_bucket = (
        int(r.oof_mape / OOF_MAPE_TIE_TOLERANCE) if math.isfinite(r.oof_mape) else 10**9
    )
    return (int(_gap_gate_failed(r)), mape_bucket, r.elapsed_seconds)


def select_champion(results: list[ModelResult]) -> ModelResult:
    """Devuelve el ganador segun lex-order (gate de gap -> MAPE OOF -> tiempo).

    Levanta ValueError si la lista esta vacia.
    """
    if not results:
        raise ValueError("select_champion: lista de results vacia")
    return min(results, key=_decision_key)


def _justification(
    champion: ModelResult,
    rivals: list[ModelResult],
) -> str:
    """Texto humano explicando por que `champion` gano sobre los rivales.

    Generado dinamicamente comparando los tres ejes de decision.
    """
    if not rivals:
        return (
            f"{champion.model_type.upper()} fue el unico modelo entrenado para "
            f"esta variedad: gap={champion.abs_gap:.4f}, "
            f"MAPE_oof={champion.oof_mape:.2f}%, "
            f"tiempo={champion.elapsed_seconds:.1f}s."
        )

    lines: list[str] = []
    for rival in rivals:
        d_mape = rival.oof_mape - champion.oof_mape
        d_time = rival.elapsed_seconds - champion.elapsed_seconds

        if _gap_gate_failed(rival) and not _gap_gate_failed(champion):
            lines.append(
                f"{rival.model_type.upper()} descartado por gate de overfitting: "
                f"gap_rel={rival.gap_rel:.2f} supera el maximo "
                f"{CHAMPION_MAX_GAP_REL} (campeon: {champion.gap_rel:.2f})."
            )
        elif d_mape > OOF_MAPE_TIE_TOLERANCE:
            lines.append(
                f"{rival.model_type.upper()} descartado por menor generalizacion: "
                f"MAPE_oof={rival.oof_mape:.2f}% vs "
                f"{champion.oof_mape:.2f}% del campeon ({d_mape:+.2f} pp)."
            )
        elif d_time > 0:
            lines.append(
                f"{rival.model_type.upper()} descartado por eficiencia "
                f"(empate tecnico en MAPE_oof): tiempo={rival.elapsed_seconds:.1f}s "
                f"vs {champion.elapsed_seconds:.1f}s ({d_time:+.1f}s mas)."
            )
        else:
            lines.append(
                f"{rival.model_type.upper()} empata tecnicamente con el campeon; "
                f"se elige {champion.model_type.upper()} por orden estable."
            )
    return " ".join(lines)


def champion_summary(
    results: list[ModelResult],
    champion: ModelResult,
) -> dict[str, object]:
    """Diccionario serializable describiendo la decision (para JSON / dashboard).

    Incluye el ranking completo, las metricas relevantes por modelo y un
    bloque de justificacion textual auto-generado.
    """
    ranking = sorted(results, key=_decision_key)
    rivals = [r for r in ranking if r.model_type != champion.model_type]
    return {
        "champion_model": champion.model_type,
        "champion_run_id": champion.mlflow_run_id,
        "champion_composite_score": champion.composite_score,
        "decision_criteria": [
            "1_gap_gate (gap_rel = |gap|/MAE_test <= CHAMPION_MAX_GAP_REL, restriccion)",
            "2_min_oof_mape (generalizacion honesta)",
            "3_min_elapsed_seconds (eficiencia)",
        ],
        "tolerances": {
            "gap_gate_rel": CHAMPION_MAX_GAP_REL,
            "gap_gate_pp_legacy": CHAMPION_MAX_GAP,
            "oof_mape_pp": OOF_MAPE_TIE_TOLERANCE,
        },
        "justification": _justification(champion, rivals),
        "ranking": [
            {
                "model": r.model_type,
                "rank": i + 1,
                "is_champion": r.model_type == champion.model_type,
                "abs_gap": r.abs_gap,
                "full_mape": r.full_mape if r.full_mape != float("inf") else None,
                "elapsed_seconds": r.elapsed_seconds,
                "composite": r.composite_score,
                "mae_test": r.metrics.get("nested_cv_mae_mean"),
                "mae_train": r.metrics.get("nested_cv_mae_train_mean"),
                "gap": r.metrics.get("nested_cv_gap_mean"),
                "r2": r.metrics.get("nested_cv_r2_mean"),
                "business_mape_oof": (
                    r.business_metrics_oof.get("mape")
                    if r.business_metrics_oof else None
                ),
                "business_r2_oof": (
                    r.business_metrics_oof.get("r2")
                    if r.business_metrics_oof else None
                ),
                "full_r2": (r.full_metrics or {}).get("r2"),
                "full_mae": (r.full_metrics or {}).get("mae"),
            }
            for i, r in enumerate(ranking)
        ],
    }
