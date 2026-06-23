"""Helpers compartidos por los transformers de step_02_clean.

Centraliza el patron `cols-> fallback config -> validar` que repetian los
tres transformers (CustomKNNImputer, OutlierCapper, MissingFlagger) con
diferencias minimas. Privado al package (`_helpers`).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def resolve_cols(
    X: pd.DataFrame,
    cols: Sequence[str] | None,
    default: Sequence[str],
    transformer_name: str,
    *,
    require_all: bool = True,
) -> list[str]:
    """Resuelve la lista de columnas a procesar.

    `cols` toma precedencia sobre `default`.

    - `require_all=True` (default, caso CustomKNNImputer / OutlierCapper):
       levanta ValueError si alguna columna no esta en X.
    - `require_all=False` (caso MissingFlagger): filtra silenciosamente las
       ausentes (algunas variedades pueden no tener todas las columnas
       opcionales para flags).
    """
    chosen = cols if cols is not None else default
    missing = [c for c in chosen if c not in X.columns]
    if missing:
        if require_all:
            raise ValueError(f"{transformer_name}: columnas inexistentes en X: {missing}")
        return [c for c in chosen if c in X.columns]
    return list(chosen)
