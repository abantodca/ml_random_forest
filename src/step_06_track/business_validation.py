"""Validacion del modelo en la unidad de negocio (KG/JR).

El modelo predice KG/JR_H (kg por jornal-hora) porque es la variable que
remueve la confusion con la duracion de la jornada (H-EF). Pero el negocio
mide en KG/JR (kg por jornal). La relacion es exacta:

    KG/JR = KG/JR_H * H-EF

Esta funcion hace la traduccion inversa al cierre del pipeline:

    KG/JR_estimado = pred(KG/JR_H) * H-EF
    -> compara con KG/JR real -> recalcula MAE, RMSE, R2, MAPE

Calculamos DOS escenarios para que el gerente vea ambas perspectivas:

  1. OOF (out-of-fold, honesto): usa las predicciones del Nested CV. Cada
     fila fue predicha por un modelo que NO la vio en train. Es la
     metrica REAL que veras en produccion.

  2. In-sample (sesgado/optimista): usa el pipeline final entrenado en
     todo el dataset, prediciendo el mismo dataset. Mide AJUSTE, no
     generalizacion. Util como sanity check (confirma que el modelo
     "puede" aprender la senal) pero NO sirve para decidir despliegue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.step_05_evaluate.metrics import calculate_regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class BusinessValidation:
    """Resultado de validar el modelo en KG/JR (unidad de negocio).

    Attributes
    ----------
    metrics_oof : metricas OOF en KG/JR (honestas, gerencial-grade).
    metrics_insample : metricas in-sample en KG/JR (sesgadas, sanity check).
    n_oof, n_insample : tamano de muestra usado en cada caso.
    n_dropped_business : filas descartadas por NaN en H-EF o KG/JR.
    kg_jr_real_oof, kg_jr_pred_oof : arrays alineados (para graficar).
    """

    metrics_oof: dict[str, float] = field(default_factory=dict)
    metrics_insample: dict[str, float] = field(default_factory=dict)
    n_oof: int = 0
    n_insample: int = 0
    n_dropped_business: int = 0
    kg_jr_real_oof: np.ndarray | None = None
    kg_jr_pred_oof: np.ndarray | None = None
    kg_jr_real_insample: np.ndarray | None = None
    kg_jr_pred_insample: np.ndarray | None = None
    # Mascara boolean (sobre el dataset original) de las filas que sobrevivieron
    # al filtro NaN en OOF. Permite alinear X_raw con kg_jr_*_oof aguas abajo
    # (boxplot por subgrupo, etc).
    oof_mask: np.ndarray | None = None

    def to_mlflow_metrics(self) -> dict[str, float]:
        """Aplana las metricas para MLflow (prefijo 'business_')."""
        flat: dict[str, float] = {}
        for k, v in self.metrics_oof.items():
            flat[f"business_oof_{k}"] = float(v)
        for k, v in self.metrics_insample.items():
            flat[f"business_insample_{k}"] = float(v)
        flat["business_n_oof"] = float(self.n_oof)
        flat["business_n_insample"] = float(self.n_insample)
        flat["business_n_dropped"] = float(self.n_dropped_business)
        return flat

    def is_empty(self) -> bool:
        return not self.metrics_oof and not self.metrics_insample


def _align_and_clean(
    y_pred_h: np.ndarray,
    h_ef: np.ndarray,
    kg_jr_real: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Devuelve (kg_jr_pred, kg_jr_real, mask, n_dropped) tras filtrar NaN/inf.

    Filtra filas donde cualquiera de pred, h_ef o kg_jr_real sea NaN/inf.
    Esto pasa con OOF (folds que no cubren toda la fila) y con datos sucios.
    """
    y_pred_h = np.asarray(y_pred_h, dtype=float)
    h_ef = np.asarray(h_ef, dtype=float)
    kg_jr_real = np.asarray(kg_jr_real, dtype=float)

    kg_jr_pred = y_pred_h * h_ef
    mask = np.isfinite(kg_jr_pred) & np.isfinite(kg_jr_real)
    n_dropped = int((~mask).sum())
    return kg_jr_pred[mask], kg_jr_real[mask], mask, n_dropped


def validate_against_business_unit(
    oof: dict[str, np.ndarray],
    final_pipeline,
    X_full: pd.DataFrame,
    business_cols: pd.DataFrame,
    target_col_real: str = "KG/JR",
    h_ef_col: str = "H-EF",
) -> BusinessValidation:
    """Calcula metricas OOF + in-sample en la unidad de negocio (KG/JR).

    Parametros
    ----------
    oof : dict con 'y_pred' (predicciones out-of-fold del Nested CV).
          Filas no cubiertas por OOF (raras) deben venir como NaN.
    final_pipeline : pipeline final ya entrenado en TODO el dataset.
    X_full : features completas (mismas que recibio el pipeline).
    business_cols : DataFrame alineado con X_full, con columnas
                    [KG/JR, H-EF].
    """
    result = BusinessValidation()

    if h_ef_col not in business_cols.columns or target_col_real not in business_cols.columns:
        # data sin las columnas de negocio: no hay nada que validar
        return result

    h_ef = business_cols[h_ef_col].to_numpy(dtype=float)
    kg_jr = business_cols[target_col_real].to_numpy(dtype=float)

    # ---- OOF (honesto) ----
    kg_jr_pred_oof, kg_jr_real_oof, mask_oof, n_drop_oof = _align_and_clean(
        oof["y_pred"], h_ef, kg_jr
    )
    if kg_jr_pred_oof.size > 0:
        result.metrics_oof = calculate_regression_metrics(kg_jr_real_oof, kg_jr_pred_oof)
        result.n_oof = kg_jr_pred_oof.size
        result.kg_jr_real_oof = kg_jr_real_oof
        result.kg_jr_pred_oof = kg_jr_pred_oof
        result.oof_mask = mask_oof

    # ---- In-sample (modelo final aplicado a todo el dataset) ----
    # In-sample falla rara vez (mismo X que se uso en fit). Si pasa, log
    # con traceback para diagnosticar; el caller obtiene metrics_insample
    # vacio (BusinessValidation tolera ausencia in-sample sin romper).
    try:
        y_pred_h_full = final_pipeline.predict(X_full)
    except Exception:
        logger.warning(
            "business_validation: final_pipeline.predict(X_full) fallo; "
            "metrics_insample queda vacio",
            exc_info=True,
        )
        y_pred_h_full = None

    if y_pred_h_full is not None:
        kg_jr_pred_in, kg_jr_real_in, _, _ = _align_and_clean(
            np.asarray(y_pred_h_full, dtype=float), h_ef, kg_jr
        )
        if kg_jr_pred_in.size > 0:
            result.metrics_insample = calculate_regression_metrics(kg_jr_real_in, kg_jr_pred_in)
            result.n_insample = kg_jr_pred_in.size
            result.kg_jr_real_insample = kg_jr_real_in
            result.kg_jr_pred_insample = kg_jr_pred_in

    result.n_dropped_business = n_drop_oof  # OOF es la referencia
    return result
