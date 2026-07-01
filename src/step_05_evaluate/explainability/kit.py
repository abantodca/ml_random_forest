"""Kit ejecutivo unificado: bundle de datos derivados del campeon.

Centraliza el calculo de las entradas ejecutivas para que renderizadores
(winner_dashboard.py, business_export.py) NUNCA reimplementen la
construccion del kit. Si manana cambia la metrica que define
'precision', se cambia aqui y los dos canales se actualizan a la vez.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.diagnostics.statistical_tests import (
    TestResult,
    breusch_pagan_residuals,
)
from src.step_05_evaluate.explainability.actions import Action, recommended_actions
from src.step_05_evaluate.explainability.bias import GroupBias, residual_bias_by_group
from src.step_05_evaluate.explainability.context import (
    TrainingContext,
    build_context,
)
from src.step_05_evaluate.explainability.verdict import Verdict, compute_verdict
from src.step_05_evaluate.metrics import mape_safe
from src.step_05_evaluate.statistical_tests import (
    MetricCI,
    bootstrap_metric_ci,
    calibration_bins,
)

if TYPE_CHECKING:
    from src.step_05_evaluate.champion import ModelResult


@dataclass(frozen=True)
class WinnerKit:
    """Bundle de datos derivados del campeon, listos para HTML / Excel.

    Attributes
    ----------
    real, pred, abs_err : arrays OOF en KG/JR alineados.
    X_aligned           : X_raw recortado por oof_mask (para subgrupos).
    oof_mape, oof_r2    : metricas OOF en KG/JR.
    abs_gap             : |MAE_test - MAE_train| del nested CV.
    verdict             : Verdict ejecutivo (icon + headline + body).
    context             : TrainingContext (filas, fechas, fundos, formatos).
    actions             : Lista de Action auto-generadas.
    fundo_bias          : sesgos direccionales por FUNDO (lista vacia = OK).
    """

    real: np.ndarray
    pred: np.ndarray
    abs_err: np.ndarray
    X_aligned: pd.DataFrame | None
    oof_mape: float
    oof_r2: float | None
    abs_gap: float
    verdict: Verdict
    context: TrainingContext
    actions: list[Action]
    fundo_bias: list[GroupBias]
    # Diagnostico estadistico (IC bootstrap + heteroscedasticity + calibration).
    # Permiten al lector saber si las metricas puntuales son estables o ruidosas,
    # y si los intervalos del modelo son validos uniformemente. None cuando la
    # muestra es insuficiente para el calculo (e.g. <20 filas).
    mae_oof_ci: MetricCI | None = None
    mape_oof_ci: MetricCI | None = None
    r2_oof_ci: MetricCI | None = None
    heteroscedasticity: TestResult | None = None
    calibration: pd.DataFrame | None = None


def _abs_errors_aligned(
    champion: ModelResult,
    X_raw: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame | None]:
    """Devuelve (real, pred, abs_err, X_aligned) en KG/JR OOF.

    Usa `business_validation` del campeon. X_aligned es X_raw recortado
    por `oof_mask` para que coincida en filas con real/pred.
    """
    bv = champion.business_validation
    real = np.asarray(getattr(bv, "kg_jr_real_oof", []) if bv else [], dtype=float)
    pred = np.asarray(getattr(bv, "kg_jr_pred_oof", []) if bv else [], dtype=float)
    abs_err = np.abs(pred - real) if real.size > 0 else np.array([])

    X_aligned: pd.DataFrame | None = None
    mask = getattr(bv, "oof_mask", None) if bv else None
    if X_raw is not None and mask is not None:
        try:
            X_aligned = X_raw.iloc[np.asarray(mask, dtype=bool)].reset_index(drop=True)
        except Exception:
            X_aligned = None
    elif X_raw is not None and len(X_raw) == abs_err.size:
        X_aligned = X_raw.reset_index(drop=True)
    return real, pred, abs_err, X_aligned


def build_winner_kit(
    *,
    variety: str,
    champion: ModelResult,
    X_raw: pd.DataFrame | None = None,
) -> WinnerKit:
    """Construye el kit ejecutivo del campeon en una sola pasada.

    Reglas:
      - Las metricas presentables al gerente son SIEMPRE OOF (honestas).
        In-sample (refit + predict all) es solo sanity check.
      - X_aligned se restringe via `oof_mask` cuando esta disponible para
        garantizar que los subgrupos del boxplot coincidan fila a fila
        con (real, pred).
    """
    real, pred, abs_err, X_aligned = _abs_errors_aligned(champion, X_raw)
    business_oof = champion.business_metrics_oof or {}
    oof_mape = float(business_oof.get("mape", float("nan")))
    oof_r2 = business_oof.get("r2")

    abs_gap = champion.abs_gap
    verdict = compute_verdict(full_mape_pct=oof_mape, abs_gap=abs_gap)
    context = build_context(X_raw)
    actions = recommended_actions(
        abs_errors=abs_err,
        real=real,
        X_aligned=X_aligned,
        global_mape=oof_mape,
        abs_gap=abs_gap,
        full_mape=oof_mape,
    )
    fundo_bias = residual_bias_by_group(
        real=real,
        pred=pred,
        X_aligned=X_aligned,
        col="FUNDO",
    )

    # Diagnostico estadistico (defensivo: si la muestra es chica o algo falla,
    # se omite la seccion en lugar de romper el dashboard completo).
    mae_ci = mape_ci = r2_ci = None
    hetero = None
    calib = None
    if real.size >= 20:
        try:
            from sklearn.metrics import mean_absolute_error, r2_score

            mae_ci = bootstrap_metric_ci(real, pred, mean_absolute_error)
            mape_ci = bootstrap_metric_ci(real, pred, mape_safe)
            r2_ci = bootstrap_metric_ci(real, pred, r2_score)
            hetero = breusch_pagan_residuals(residuals=(pred - real), predictions=pred)
            calib = calibration_bins(real, pred, n_bins=10)
        except Exception:
            # Falla de algun calculo no debe tumbar el dashboard.
            pass

    return WinnerKit(
        real=real,
        pred=pred,
        abs_err=abs_err,
        X_aligned=X_aligned,
        oof_mape=oof_mape,
        oof_r2=oof_r2,
        abs_gap=abs_gap,
        verdict=verdict,
        context=context,
        actions=actions,
        fundo_bias=fundo_bias,
        mae_oof_ci=mae_ci,
        mape_oof_ci=mape_ci,
        r2_oof_ci=r2_ci,
        heteroscedasticity=hetero,
        calibration=calib,
    )
