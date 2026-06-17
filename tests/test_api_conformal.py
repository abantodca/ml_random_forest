"""P0.1d — API _conformal_halfwidths: fundo conocido/desconocido/cold-start.

Corre en el contenedor API (necesita fastapi):
    docker compose run --rm -v ./tests:/app/tests --entrypoint sh api -c \
      "pip install -q pytest && pytest /app/tests/test_api_conformal.py -q"
En el trainer se salta solo (importorskip).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi", reason="solo corre en el contenedor api")

try:
    from app.services.uncertainty import conformal_halfwidths
except Exception as exc:  # config/env del API ausentes fuera del contenedor
    pytest.skip(f"app no importable aqui: {exc}", allow_module_level=True)

_CONFORMAL = {
    "alpha": 0.10,
    "q_global": 1.0,
    "q_by_fundo": {"A": 0.5},
    "known_ff": ["A__G", "B__G"],
    "cold_factor": 2.0,
    "n_calibration": 150,
}


def _halfwidths(rows):
    df = pd.DataFrame(rows)
    return conformal_halfwidths(_CONFORMAL, df, len(df))


def test_fundo_conocido_ff_conocido_usa_q_del_fundo():
    hw = _halfwidths([{"FUNDO": "A", "FORMATO": "G"}])
    assert hw[0] == 0.5


def test_fundo_sin_q_propio_cae_a_global():
    hw = _halfwidths([{"FUNDO": "B", "FORMATO": "G"}])
    assert hw[0] == 1.0


def test_ff_desconocido_aplica_cold_factor():
    # Fundo con q propio pero FORMATO nunca visto en ese fundo -> 0.5 * 2.
    hw = _halfwidths([{"FUNDO": "A", "FORMATO": "NUEVO"}])
    assert hw[0] == 1.0
    # Fundo y formato desconocidos -> global * cold.
    hw = _halfwidths([{"FUNDO": "Z", "FORMATO": "Z"}])
    assert hw[0] == 2.0


def test_legacy_sin_conformal_no_rompe_vector():
    # Contrato de forma: una banda por fila, en orden.
    hw = _halfwidths([
        {"FUNDO": "A", "FORMATO": "G"},
        {"FUNDO": "Z", "FORMATO": "Z"},
    ])
    assert isinstance(hw, np.ndarray) and hw.shape == (2,)
    assert hw[0] == 0.5 and hw[1] == 2.0
