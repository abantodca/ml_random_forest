"""Incertidumbre de predicción: bandas conformal + fallback legacy.

Extraído de `mlflow_service.py` (P1.3 del plan de refactor 2026-06-12):
el servicio MLflow mezclaba cliente/cache de modelos con la lógica de
incertidumbre. Aquí vive SOLO la segunda — una función pura sin estado,
testeable sin levantar el servicio (tests/test_api_conformal.py).

Prioridad (ver `predict_with_halfwidths`):
  1. `conformal_`        : metadata calibrada con residuos OOF del nested
                           CV (cuantil por fundo, cascade a global, factor
                           cold-start ~2x si el FUNDO__FORMATO no tiene
                           historia). Cobertura estadística real.
  2. `predict_with_std`  : ±1.96·std del ensemble — heurística legacy con
                           cobertura << nominal; solo modelos sin conformal_.
  3. (None)              : pickles pre-ensemble; el caller decide qué hacer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_halfwidths(conformal: dict, df: pd.DataFrame, n: int) -> np.ndarray:
    """Semiancho de banda por fila: cuantil conformal del fundo (cascade
    a global) x factor cold-start si el FUNDO__FORMATO no tiene historia
    suficiente en train (error medido ~2x)."""
    q_global = float(conformal.get("q_global", 0.0))
    q_by_fundo = conformal.get("q_by_fundo", {}) or {}
    known_ff = set(conformal.get("known_ff", []) or [])
    cold_factor = float(conformal.get("cold_factor", 2.0))

    fundos = df["FUNDO"].astype(str) if "FUNDO" in df.columns else pd.Series([""] * n)
    formatos = df["FORMATO"].astype(str) if "FORMATO" in df.columns else pd.Series([""] * n)
    out = np.empty(n, dtype=float)
    for i, (f, fmt) in enumerate(zip(fundos, formatos, strict=True)):
        q = q_by_fundo.get(f, q_global)
        if known_ff and f"{f}__{fmt}" not in known_ff:
            q *= cold_factor
        out[i] = q
    return out


def predict_with_halfwidths(target, df: pd.DataFrame):
    """(predicciones, semianchos|None) para cualquier modelo servible.

    `target` es el sklearn real (OOFEnsembleRegressor desempaquetado del
    pyfunc) o el pyfunc mismo si no se pudo desempaquetar.
    """
    conformal = getattr(target, "conformal_", None)
    if conformal:
        preds = np.asarray(target.predict(df), dtype=float)
        return preds, conformal_halfwidths(conformal, df, len(preds))
    if hasattr(target, "predict_with_std"):
        mean, std = target.predict_with_std(df)
        # ±1.96·std para aproximar el mismo contrato (semiancho de banda)
        # que el camino conformal entrega directamente.
        return mean, 1.96 * np.asarray(std, dtype=float)
    return target.predict(df), None
