"""Fixtures compartidos de la suite P0 (plan docs/PLAN_REFACTOR_2026-06-12.md).

La suite corre DENTRO del contenedor (mismas versiones que produccion):

    docker compose run --rm --entrypoint sh trainer -c \
      "pip install -q pytest && PYTHONPATH=/app pytest tests/ -q"

El test de la API (`test_api_conformal.py`) ademas corre en el contenedor
api (necesita fastapi); en el trainer se salta solo via importorskip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Raiz del repo y api/ importables sin instalar el paquete.
_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "api")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture()
def df_raw_minimo() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintetico con las 9 columnas raw del contrato data_loader.

    Un solo grupo FUNDO=A / FORMATO=G: 5 dias consecutivos con 1 fila c/u
    (KG/HA = 1..5) y un 6to dia con DOS filas hermanas (KG/HA = 100 y 0).
    Disenado para exponer el leakage same-day del shift(1) posicional.
    """
    fechas = [pd.Timestamp(2024, 1, d) for d in range(1, 6)] + [
        pd.Timestamp(2024, 1, 6),
        pd.Timestamp(2024, 1, 6),
    ]
    n = len(fechas)
    X = pd.DataFrame(
        {
            "FECHA": fechas,
            "FUNDO": ["A"] * n,
            "FORMATO": ["G"] * n,
            "KG/HA": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 0.0],
            "%INDUS": [0.1] * n,
            "DPC": [10.0] * n,
            "P/BAYA": [3.0] * n,
            "HA": [2.0] * n,
            "DIA_COSECHA": list(range(1, n + 1)),
        }
    )
    y = pd.Series(np.linspace(4.0, 6.0, n), name="KG/JR_H")
    return X, y
