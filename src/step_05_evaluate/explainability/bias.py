"""Sesgo direccional por subgrupo (sobreestima o subestima sistematicamente).

Diferencia clave vs `actions.recommended_actions`:
  - `recommended_actions` mira MAGNITUD del error (MAPE > 1.5x global).
  - Este mira DIRECCION del error (mean(signed_residual) consistente).

Un subgrupo con MAPE razonable PERO con bias=+8% sostenido le cuesta
dinero al negocio sin disparar el filtro de magnitud. Las predicciones
llegan en rango aceptable pero todas hacia el mismo lado, generando
subestimacion estructural.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupBias:
    """Sesgo direccional del modelo en un subgrupo (FUNDO / FORMATO)."""

    group_value: str
    n: int
    mean_signed_bias: float          # mean(pred - real) en KG/JR. >0 = sobreestima
    bias_pct_of_real_mean: float     # bias / mean(real) * 100 (lectura ejecutiva)
    direction: str                   # 'sobreestima' | 'subestima' | 'neutro'


def residual_bias_by_group(
    *,
    real: np.ndarray,
    pred: np.ndarray,
    X_aligned: pd.DataFrame | None,
    col: str = "FUNDO",
    min_n: int = 10,
    bias_threshold_pct: float = 5.0,
) -> list[GroupBias]:
    """Devuelve solo grupos con |bias_pct| >= bias_threshold_pct, ordenados
    por |bias_pct| descendente. Lista vacia = sin sesgos significativos.
    """
    out: list[GroupBias] = []
    if X_aligned is None or col not in X_aligned.columns:
        return out
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if real.size == 0 or len(X_aligned) != real.size:
        return out

    groups = X_aligned[col].astype(str).reset_index(drop=True)
    for cat in groups.unique():
        if pd.isna(cat) or cat == "":
            continue
        mask = (groups == cat).to_numpy()
        if mask.sum() < min_n:
            continue
        cat_real = real[mask]
        cat_pred = pred[mask]
        if cat_real.size == 0:
            continue
        mean_real = float(np.mean(cat_real))
        if abs(mean_real) < 1e-9:
            continue
        signed_bias = float(np.mean(cat_pred - cat_real))
        bias_pct = signed_bias / mean_real * 100.0
        if abs(bias_pct) < bias_threshold_pct:
            continue
        if bias_pct > 0:
            direction = "sobreestima"
        elif bias_pct < 0:
            direction = "subestima"
        else:
            direction = "neutro"
        out.append(GroupBias(
            group_value=str(cat),
            n=int(mask.sum()),
            mean_signed_bias=signed_bias,
            bias_pct_of_real_mean=bias_pct,
            direction=direction,
        ))
    return sorted(out, key=lambda g: -abs(g.bias_pct_of_real_mean))
