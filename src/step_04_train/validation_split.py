"""Particiones ligeras usadas por early stopping.

Este módulo no importa XGBoost ni LightGBM, de modo que la política temporal
puede probarse sin cargar librerías nativas.
"""

from __future__ import annotations

import numpy as np

from src.config import (
    EARLY_STOPPING_MIN_VAL,
    EARLY_STOPPING_VAL_FRACTION,
)


def temporal_tail_holdout_indices(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(train, valid)`` usando el tramo final como validación."""

    if n_rows < 2:
        raise ValueError("early stopping requiere al menos dos filas")
    n_val = min(
        max(int(n_rows * EARLY_STOPPING_VAL_FRACTION), EARLY_STOPPING_MIN_VAL),
        n_rows // 3,
    )
    n_val = max(1, n_val)
    split_at = n_rows - n_val
    return np.arange(split_at), np.arange(split_at, n_rows)
