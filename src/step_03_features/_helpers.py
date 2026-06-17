"""Helpers compartidos entre los modulos de step_03_features.

`safe_ratio(num, den)` es la version polimorfica unificada de las dos
implementaciones que existian antes (una sobre ndarray en
`feature_engineering.py` y otra sobre pd.Series en `lag_features.py`).

Ambas variantes originales aplicaban exactamente la misma logica:
NaN donde `den <= 0` (incluye 0 y negativos) o `den is NaN`; division
elementwise en otro caso. La unificacion preserva esa semantica
exacta para no alterar la salida de los features que dependen de ella.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Soporta tanto ndarray como Series (preservando el tipo de entrada).
def safe_ratio[ArrayLike: (np.ndarray, pd.Series)](
    num: ArrayLike, den: ArrayLike
) -> ArrayLike:
    """num/den retornando NaN donde den<=0 o den is NaN.

    Polimorfica: si la entrada es `pd.Series`, devuelve `pd.Series`
    (con el mismo index que `num`); si es `np.ndarray`, devuelve
    `np.ndarray`. Para cualquier otro tipo, se hace fallback al
    camino ndarray.

    Semantica (identica en ambas implementaciones originales):
      - `den > 0`           -> num / den
      - `den <= 0` (0 o <0) -> NaN
      - `den is NaN`        -> NaN  (NaN > 0 es False -> cae al else)

    Suprime warnings de divide-by-zero / invalid de NumPy.
    """
    if isinstance(den, pd.Series):
        # Camino Series: preserva index. .where(cond, other) reemplaza
        # con `other` donde cond es False -> NaN donde den<=0 o NaN.
        den_safe = den.where(den > 0, other=np.nan)
        return num / den_safe

    # Camino ndarray (default): np.where + supresion de warnings.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)
