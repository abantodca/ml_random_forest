"""P0.1c — build_conformal_metadata: cascade fundo->global y cold-start.

Contrato (conformal_bands.py): q por fundo SOLO con n >= MIN_GROUP (100);
known_ff SOLO con n >= MIN_FF_KNOWN (10); <20 residuos -> None.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.step_05_evaluate.conformal_bands import (
    COLD_FACTOR,
    MIN_FF_KNOWN,
    MIN_GROUP,
    build_conformal_metadata,
)


def _datos(n_a=120, n_b=30, n_b_raro=5):
    """Fundo A grande (>=MIN_GROUP), B chico; B__X es un FF raro (<10)."""
    rng = np.random.default_rng(42)
    n = n_a + n_b
    y_true = rng.uniform(3, 8, n)
    y_pred = y_true + rng.normal(0, 0.5, n)
    fundo = pd.Series(["A"] * n_a + ["B"] * n_b)
    formato = pd.Series(
        ["G"] * n_a + ["G"] * (n_b - n_b_raro) + ["X"] * n_b_raro
    )
    return y_true, y_pred, fundo, formato


def test_q_por_fundo_solo_con_historia_suficiente():
    y_true, y_pred, fundo, formato = _datos()
    meta = build_conformal_metadata(y_true, y_pred, fundo, formato)
    assert meta is not None
    assert "A" in meta["q_by_fundo"]          # n=120 >= MIN_GROUP
    assert "B" not in meta["q_by_fundo"]      # n=30 < MIN_GROUP -> cae a global
    assert meta["q_global"] > 0
    assert meta["n_calibration"] == len(y_true)
    assert meta["cold_factor"] == COLD_FACTOR
    assert MIN_GROUP == 100  # si cambia, revisar este test y la API


def test_known_ff_excluye_grupos_raros():
    y_true, y_pred, fundo, formato = _datos()
    meta = build_conformal_metadata(y_true, y_pred, fundo, formato)
    assert "A__G" in meta["known_ff"]
    assert "B__G" in meta["known_ff"]         # 25 >= MIN_FF_KNOWN
    assert "B__X" not in meta["known_ff"]     # 5 < MIN_FF_KNOWN -> cold-start
    assert MIN_FF_KNOWN == 10


def test_pocos_residuos_devuelve_none():
    y_true, y_pred, fundo, formato = _datos(n_a=10, n_b=5, n_b_raro=0)
    assert build_conformal_metadata(y_true, y_pred, fundo, formato) is None


def test_nan_en_oof_se_filtran():
    y_true, y_pred, fundo, formato = _datos()
    y_pred = y_pred.copy()
    y_pred[:30] = np.nan  # filas no cubiertas por OOF
    meta = build_conformal_metadata(y_true, y_pred, fundo, formato)
    assert meta is not None
    assert meta["n_calibration"] == len(y_true) - 30
