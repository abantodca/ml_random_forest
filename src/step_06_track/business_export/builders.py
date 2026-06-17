"""Builders de DataFrames para cada hoja del Excel ejecutivo.

Cada `_build_*_df` arma UNA hoja. Son funciones puras (no tocan disco,
no llaman MLflow, no aplican estilos): reciben datos / WinnerKit, devuelven
un DataFrame listo para `to_excel`.

El renombrado tecnico -> lenguaje natural (`_PRETTY_COLUMNS`) vive aqui
porque es decision de presentacion (no de calculo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    REPORT_BUSINESS_UNIT,
    REPORT_MAE_AMBER_RATIO,
    REPORT_MAE_TARGET,
    REPORT_PROJECT_NAME,
    REPORT_R2_AMBER_THRESHOLD,
    REPORT_R2_TARGET,
)
from src.step_05_evaluate.explainability import (
    compute_verdict,
    glossary_terms,
    kpi_explanatory_power,
    kpi_precision,
    kpi_vs_baseline,
    recommended_actions,
)

# ---------------------------------------------------------------------------
# Renombrado de columnas (modelo/error -> lenguaje claro)
# ---------------------------------------------------------------------------

# Mapeo aplicado a las hojas Predicciones_*. Solo renombramos columnas
# tecnicas del modelo; las del dominio (FORMATO, FUNDO, FECHA, KG/HA, ...)
# se quedan como estan porque el negocio ya las conoce con esos nombres.
_PRETTY_COLUMNS: dict[str, str] = {
    "row_id": "ID fila",
    "KG/JR_H_real": "Productividad real (kg/jornal-hora)",
    "KG/JR_H_pred": "Productividad estimada (kg/jornal-hora)",
    "H-EF": "Horas efectivas (jornada)",
    "KG/JR_real": "Cosecha real (kg/jornal)",
    "KG/JR_estimado": "Cosecha estimada (kg/jornal)",
    "error_abs (kg/jornal)": "Error absoluto (kg/jornal)",
    "error_pct (%)": "Error porcentual (%)",
    "fold_id": "Partición CV",
}


def build_predictions_df(
    X_raw: pd.DataFrame,
    business_cols: pd.DataFrame,
    y_h_true: np.ndarray,
    y_h_pred: np.ndarray,
    fold_id: np.ndarray | None = None,
    y_h_std: np.ndarray | None = None,
    conformal_residuals: np.ndarray | None = None,
    conformal_alpha: float = 0.05,
) -> pd.DataFrame:
    """Une features + target + predicciones + error en KG/JR.

    Filas con KG/JR_H_pred = NaN (folds no cubiertos en OOF) o con
    H-EF / KG/JR NaN se descartan: no se pueden traducir a unidad de negocio.

    Bandas de confianza al 95% (en orden de preferencia):
      - `conformal_residuals` no None -> Conformal Prediction (split conformal,
        garantia estadistica de cobertura). Se prefiere a las heuristicas.
      - `y_h_std` no None -> heuristica `mean +/- 1.96*std` (asume
        homoscedasticidad + gaussianidad, no garantizada).
      - Ninguna -> sin bandas.
    """
    h_ef = business_cols["H-EF"].to_numpy(dtype=float)
    kg_jr_real = business_cols["KG/JR"].to_numpy(dtype=float)
    y_h_pred = np.asarray(y_h_pred, dtype=float)
    y_h_true = np.asarray(y_h_true, dtype=float)

    kg_jr_pred = y_h_pred * h_ef
    abs_err = np.abs(kg_jr_pred - kg_jr_real)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_err = np.where(kg_jr_real != 0, abs_err / np.abs(kg_jr_real) * 100.0, np.nan)

    df = X_raw.reset_index(drop=True).copy()
    df.insert(0, "row_id", np.arange(len(df)))
    df["KG/JR_H_real"] = y_h_true
    df["KG/JR_H_pred"] = y_h_pred
    df["H-EF"] = h_ef
    df["KG/JR_real"] = kg_jr_real
    df["KG/JR_estimado"] = kg_jr_pred

    finite_mask = (
        np.isfinite(y_h_pred)
        & np.isfinite(h_ef)
        & np.isfinite(kg_jr_real)
    )

    # Bandas: prefiere Conformal (con sustento estadistico) sobre heuristica.
    # Se calculan en KG/JR_H (espacio del modelo) y luego se multiplican por
    # H-EF para llevar a KG/JR (transformacion lineal -> linealidad).
    if conformal_residuals is not None:
        from src.step_05_evaluate.statistical_tests import conformal_intervals
        y_h_lo, y_h_hi = conformal_intervals(
            oof_residuals=np.asarray(conformal_residuals, dtype=float),
            predictions=y_h_pred,
            alpha=conformal_alpha,
        )
        df["Estimación baja (kg/jornal)"] = y_h_lo * h_ef
        df["Estimación alta (kg/jornal)"] = y_h_hi * h_ef
    elif y_h_std is not None:
        y_h_std_arr = np.asarray(y_h_std, dtype=float)
        kg_jr_lo = (y_h_pred - 1.96 * y_h_std_arr) * h_ef
        kg_jr_hi = (y_h_pred + 1.96 * y_h_std_arr) * h_ef
        df["Estimación baja (kg/jornal)"] = kg_jr_lo
        df["Estimación alta (kg/jornal)"] = kg_jr_hi

    df["error_abs (kg/jornal)"] = abs_err
    df["error_pct (%)"] = pct_err
    if fold_id is not None:
        df["fold_id"] = np.asarray(fold_id, dtype=int)

    df = df.loc[finite_mask].reset_index(drop=True)

    # Renombrado a lenguaje claro
    return df.rename(columns=_PRETTY_COLUMNS)


def build_subgroup_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame | None:
    """Agrega metricas por subgrupo y rankea de peor a mejor por error %."""
    if group_col not in df.columns:
        return None
    err_abs_col = "Error absoluto (kg/jornal)"
    err_pct_col = "Error porcentual (%)"
    real_col = "Cosecha real (kg/jornal)"
    pred_col = "Cosecha estimada (kg/jornal)"
    g = df.groupby(group_col, dropna=False)
    out = pd.DataFrame({
        "n cosechas": g.size(),
        "Real promedio (kg/jornal)": g[real_col].mean(),
        "Estimado promedio (kg/jornal)": g[pred_col].mean(),
        "MAE (kg/jornal)": g[err_abs_col].mean(),
        "Error % promedio": g[err_pct_col].mean(),
        "Error máximo (kg/jornal)": g[err_abs_col].max(),
    })
    out = out.sort_values("Error % promedio", ascending=False).round(3)
    out.insert(0, "Ranking (peor → mejor)", np.arange(1, len(out) + 1))
    return out


def build_resumen_df(
    *,
    variety: str,
    model_type: str,
    nested_metrics: dict[str, float],
    business_validation,
    timestamp: str,
    n_oof: int,
    n_insample: int,
) -> pd.DataFrame:
    """Resumen tecnico para DS / auditoria. Lenguaje tecnico OK aqui."""
    moof = business_validation.metrics_oof if business_validation else {}
    mins = business_validation.metrics_insample if business_validation else {}
    r2_kgjr = float(moof.get("r2", float("nan")))
    mae_kgh = float(nested_metrics.get("nested_cv_mae_mean", float("nan")))

    def _state_r2(v):
        if v != v:
            return "—"
        return (
            "VERDE" if v >= REPORT_R2_TARGET
            else ("AMARILLO" if v >= REPORT_R2_AMBER_THRESHOLD else "ROJO")
        )

    def _state_mae(v):
        if v != v:
            return "—"
        return (
            "VERDE" if v <= REPORT_MAE_TARGET
            else (
                "AMARILLO" if v <= REPORT_MAE_AMBER_RATIO * REPORT_MAE_TARGET
                else "ROJO"
            )
        )

    rows = [
        ("## METADATA ##", "", ""),
        ("Variedad", variety, ""),
        ("Modelo", model_type.upper(), ""),
        ("Generado", timestamp, ""),
        ("Filas evaluadas (OOF)", n_oof, ""),
        ("Filas evaluadas (Total)", n_insample, ""),
        ("", "", ""),
        ("## TARGETS GERENCIALES ##", "", ""),
        ("Target R² (KG/JR, OOF) >=", f"{REPORT_R2_TARGET:.2f}", ""),
        ("Target MAE (KG/JR_H, Test CV) <=", f"{REPORT_MAE_TARGET:.2f}", ""),
        ("", "", ""),
        ("## METRICAS DEL MODELO (KG/JR_H) ##", "", ""),
        ("MAE Test CV", f"{nested_metrics.get('nested_cv_mae_mean', float('nan')):.4f}", _state_mae(mae_kgh)),
        ("MAE std Test CV", f"{nested_metrics.get('nested_cv_mae_std', float('nan')):.4f}", ""),
        ("MAE Train CV", f"{nested_metrics.get('nested_cv_mae_train_mean', float('nan')):.4f}", ""),
        ("Brecha (Test - Train)", f"{nested_metrics.get('nested_cv_gap_mean', float('nan')):+.4f}", ""),
        ("R² Test CV", f"{nested_metrics.get('nested_cv_r2_mean', float('nan')):.4f}", ""),
        ("", "", ""),
        ("## METRICAS EN UNIDAD DE NEGOCIO (KG/JR) ##", "", ""),
        ("R² OOF (gerencial)", f"{moof.get('r2', float('nan')):.4f}", _state_r2(r2_kgjr)),
        ("MAE OOF (kg/jornal)", f"{moof.get('mae', float('nan')):.4f}", ""),
        ("RMSE OOF", f"{moof.get('rmse', float('nan')):.4f}", ""),
        ("MAPE OOF (%)", f"{moof.get('mape', float('nan')):.2f}", ""),
        ("", "", ""),
        ("R² Aplicación Total (sanity)", f"{mins.get('r2', float('nan')):.4f}", ""),
        ("MAE Aplicación Total", f"{mins.get('mae', float('nan')):.4f}", ""),
        ("MAPE Aplicación Total (%)", f"{mins.get('mape', float('nan')):.2f}", ""),
    ]
    return pd.DataFrame(rows, columns=["Métrica", "Valor", "Estado"])


def build_inicio_df(
    *,
    variety: str,
    model_type: str,
    timestamp: str,
    business_validation,
    abs_gap: float,
    n_rows: int,
) -> pd.DataFrame:
    """Hoja portada en lenguaje natural. Lo unico que un gerente necesita leer."""
    moof = business_validation.metrics_oof if business_validation else {}
    real_oof = getattr(business_validation, "kg_jr_real_oof", None)
    pred_oof = getattr(business_validation, "kg_jr_pred_oof", None)
    abs_err = (
        np.abs(np.asarray(pred_oof) - np.asarray(real_oof))
        if real_oof is not None and pred_oof is not None and len(real_oof) > 0
        else np.array([])
    )
    # Veredicto + KPIs en lenguaje natural usan SIEMPRE metricas OOF
    # (honestas, sobre datos no vistos). In-sample seria optimista.
    oof_mape = float(moof.get("mape", float("nan")))
    oof_r2 = moof.get("r2")

    verdict = compute_verdict(full_mape_pct=oof_mape, abs_gap=abs_gap)
    k1 = kpi_precision(abs_err, oof_mape)
    k2 = kpi_explanatory_power(oof_r2)
    k3 = kpi_vs_baseline(real_oof if real_oof is not None else np.array([]),
                         pred_oof if pred_oof is not None else np.array([]))

    rows = [
        ("◆◆◆ DASHBOARD EJECUTIVO ◆◆◆", "", ""),
        (REPORT_PROJECT_NAME, "", ""),
        ("", "", ""),
        ("Variedad", variety, ""),
        ("Modelo elegido", model_type.upper(), ""),
        ("Unidad de negocio", REPORT_BUSINESS_UNIT, ""),
        ("Cosechas analizadas", f"{n_rows:,}", ""),
        ("Generado", timestamp, ""),
        ("", "", ""),
        ("◆◆◆ VEREDICTO ◆◆◆", "", ""),
        (f"{verdict.icon}  {verdict.title}", "", verdict.color_key.upper()),
        (verdict.headline, "", ""),
        (verdict.body, "", ""),
        ("", "", ""),
        ("◆◆◆ LAS 3 PREGUNTAS QUE IMPORTAN ◆◆◆", "", ""),
        ("", "", ""),
        ("1. ¿Qué tan preciso es?", "", k1.score_label),
        (k1.headline, "", ""),
        (k1.detail, "", ""),
        ("", "", ""),
        ("2. ¿Cuánto explica?", "", k2.score_label),
        (k2.headline, "", ""),
        (k2.detail, "", ""),
        ("", "", ""),
        ("3. ¿Vale la pena vs no usar modelo?", "", k3.score_label),
        (k3.headline, "", ""),
        (k3.detail, "", ""),
        ("", "", ""),
        ("◆◆◆ ÍNDICE DE HOJAS ◆◆◆", "", ""),
        ("Inicio", "Esta hoja: veredicto + KPIs en lenguaje simple", ""),
        ("Acciones", "Qué hacer hoy con esta información", ""),
        ("Resumen", "Métricas técnicas globales (auditoría / DS)", ""),
        ("Por_FORMATO", "Error promedio por formato del producto", ""),
        ("Por_FUNDO", "Error promedio por fundo", ""),
        ("Predicciones_OOF", "Detalle fila a fila — predicciones honestas (no contaminadas)", ""),
        ("Predicciones_Total", "Detalle fila a fila — modelo aplicado a TODA la data (sanity)", ""),
        ("Glosario", "Términos técnicos explicados en lenguaje simple", ""),
    ]
    return pd.DataFrame(rows, columns=["Concepto", "Valor", "Nota"])


def build_acciones_df(
    *,
    business_validation,
    abs_gap: float,
    full_mape: float,
    X_aligned: pd.DataFrame | None,
) -> pd.DataFrame:
    """Tabla de acciones recomendadas auto-generada."""
    real_oof = getattr(business_validation, "kg_jr_real_oof", None)
    pred_oof = getattr(business_validation, "kg_jr_pred_oof", None)
    real = np.asarray(real_oof if real_oof is not None else [])
    pred = np.asarray(pred_oof if pred_oof is not None else [])
    abs_errs = np.abs(pred - real) if real.size > 0 else np.array([])
    global_mape = (
        float(business_validation.metrics_oof.get("mape", float("nan")))
        if business_validation else float("nan")
    )

    actions = recommended_actions(
        abs_errors=abs_errs, real=real, X_aligned=X_aligned,
        global_mape=global_mape, abs_gap=abs_gap, full_mape=full_mape,
    )
    rows = []
    for i, a in enumerate(actions, 1):
        sev_label = {
            "critical": "CRÍTICO", "warning": "ATENCIÓN", "info": "OK",
        }.get(a.severity, a.severity.upper())
        rows.append((f"#{i}", f"{a.icon} {sev_label}", a.title, a.body))
    if not rows:
        rows = [("—", "OK", "Sin acciones recomendadas", "")]
    return pd.DataFrame(rows, columns=["#", "Severidad", "Acción", "Detalle"])


def build_glosario_df() -> pd.DataFrame:
    """Hoja de glosario para lectores no-tecnicos."""
    rows = [(t, d) for t, d in glossary_terms()]
    return pd.DataFrame(rows, columns=["Término", "Significado"])
