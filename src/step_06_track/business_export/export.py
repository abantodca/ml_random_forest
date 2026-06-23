"""Orquestador del Excel ejecutivo: ensambla DataFrames + escribe + formatea."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.step_06_track.business_export.builders import (
    build_acciones_df,
    build_glosario_df,
    build_inicio_df,
    build_predictions_df,
    build_resumen_df,
    build_subgroup_summary,
)
from src.step_06_track.business_export.formatting import apply_formatting

logger = logging.getLogger(__name__)


def export_business_excel(
    *,
    variety: str,
    model_type: str,
    X_raw: pd.DataFrame,
    business_cols: pd.DataFrame,
    oof: dict[str, np.ndarray],
    final_pipeline,
    business_validation,
    nested_metrics: dict[str, float],
    output_dir: Path | str | None = None,
    filename: str | None = None,
) -> Path | None:
    """Genera el Excel ejecutivo multi-hoja del campeon.

    Devuelve la ruta del archivo o None si faltan KG/JR / H-EF.
    """
    if (
        business_cols is None
        or "H-EF" not in business_cols.columns
        or "KG/JR" not in business_cols.columns
    ):
        return None

    out_dir = Path(output_dir) if output_dir else ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = filename or f"business_export_{variety}_{model_type}_{ts}.xlsx"
    out_path = out_dir / fname

    # ---- DataFrames ----
    df_oof = build_predictions_df(
        X_raw=X_raw,
        business_cols=business_cols,
        y_h_true=oof["y_true"],
        y_h_pred=oof["y_pred"],
        fold_id=oof.get("fold_id"),
    )
    # Si el final_pipeline es OOFEnsembleRegressor (caso normal con K>=1),
    # usamos predict_with_std para obtener bandas. Si el pipeline es legacy
    # (sklearn Pipeline simple sin K-ensemble), cae a predict() y bandas=None.
    y_h_pred_full = np.full(len(X_raw), np.nan)
    y_h_std_full: np.ndarray | None = None
    try:
        if hasattr(final_pipeline, "predict_with_std"):
            y_h_pred_full, y_h_std_full = final_pipeline.predict_with_std(X_raw)
            # std==0 todo (n_models=1) -> no aporta banda, descartamos.
            if y_h_std_full is not None and not np.any(y_h_std_full > 0):
                y_h_std_full = None
        else:
            y_h_pred_full = final_pipeline.predict(X_raw)
    except Exception:
        logger.warning(
            "business_export: final_pipeline.predict(X_raw) fallo; "
            "Predicciones_Total saldra con NaN",
            exc_info=True,
        )

    # Residuals OOF para calibrar Conformal Prediction (preferido sobre la
    # heuristica `mean +/- 1.96*std`). Se filtran NaN del OOF (folds no
    # cubiertos). Conformal devuelve bandas con cobertura garantizada al
    # 95% (split conformal, ver statistical_tests.conformal_intervals).
    oof_y_true = np.asarray(oof["y_true"], dtype=float)
    oof_y_pred = np.asarray(oof["y_pred"], dtype=float)
    oof_residuals = oof_y_true - oof_y_pred
    oof_residuals = oof_residuals[np.isfinite(oof_residuals)]

    df_total = build_predictions_df(
        X_raw=X_raw,
        business_cols=business_cols,
        y_h_true=oof_y_true,
        y_h_pred=y_h_pred_full,
        y_h_std=y_h_std_full,
        conformal_residuals=oof_residuals if len(oof_residuals) >= 20 else None,
        conformal_alpha=0.05,
    )

    by_formato = build_subgroup_summary(df_oof, "FORMATO")
    by_fundo = build_subgroup_summary(df_oof, "FUNDO")

    df_resumen = build_resumen_df(
        variety=variety,
        model_type=model_type,
        nested_metrics=nested_metrics,
        business_validation=business_validation,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        n_oof=len(df_oof),
        n_insample=len(df_total),
    )

    # Veredicto + acciones usan OOF (honesto). abs_gap viene de nested CV.
    abs_gap = abs(float(nested_metrics.get("nested_cv_gap_mean", 0.0)))
    oof_metrics = (business_validation.metrics_oof or {}) if business_validation else {}
    oof_mape = float(oof_metrics.get("mape", float("nan")))

    # X alineado con OOF (para subgroups en Acciones)
    X_aligned: pd.DataFrame | None = None
    bv_mask = getattr(business_validation, "oof_mask", None)
    if bv_mask is not None:
        try:
            X_aligned = X_raw.iloc[np.asarray(bv_mask, dtype=bool)].reset_index(drop=True)
        except Exception:
            X_aligned = None

    df_inicio = build_inicio_df(
        variety=variety,
        model_type=model_type,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        business_validation=business_validation,
        abs_gap=abs_gap,
        n_rows=len(df_oof),
    )
    df_acciones = build_acciones_df(
        business_validation=business_validation,
        abs_gap=abs_gap,
        full_mape=oof_mape,
        X_aligned=X_aligned,
    )
    df_glosario = build_glosario_df()

    # ---- Escritura ordenada (de simple a detallado) ----
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_inicio.to_excel(writer, sheet_name="Inicio", index=False)
        df_acciones.to_excel(writer, sheet_name="Acciones", index=False)
        df_resumen.to_excel(writer, sheet_name="Resumen", index=False)
        if by_formato is not None:
            by_formato.to_excel(writer, sheet_name="Por_FORMATO")
        if by_fundo is not None:
            by_fundo.to_excel(writer, sheet_name="Por_FUNDO")
        df_oof.to_excel(writer, sheet_name="Predicciones_OOF", index=False)
        df_total.to_excel(writer, sheet_name="Predicciones_Total", index=False)
        df_glosario.to_excel(writer, sheet_name="Glosario", index=False)
        apply_formatting(writer.book)

    return out_path
