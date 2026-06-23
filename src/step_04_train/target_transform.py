"""Transformaciones del target para estabilizar varianza y limitar outliers.

Construye un `TransformedTargetRegressor` que envuelve el modelo y aplica
en el espacio de y:
    func        : y -> log1p(min(y, p99.5(y)))   (cap suave + log)
    inverse_func: log -> expm1                   (sin descap, predict honesto)

El cap se calcula DENTRO del fold (TTR.fit recibe el y_train del fold),
asi que es CV-safe: el y_test del scoring no pasa por func.

Nota historica iter1->iter2: empezamos con cap p99 (1% extremos
aplastados) que mejoro deciles 1-3 pero acentuo subestimacion del decil
10 (-7.34 -> -8.64). Aflojamos a p99.5 (0.5% mas alto) para recuperar
predicciones extremas sin perder estabilidad.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor

from src.config import RANDOM_STATE

CAP_PERCENTILE: float = 99.5  # p99.5 -> ~0.5% de filas aplastadas en train


# Docstring compartido por las factories model_xgb / model_lgb. Se inyecta via
# format() para evitar duplicar el bloque sobre paralelismo en cada modulo.
_PARALLELISM_DOCSTRING = """`n_jobs` lee env var MODEL_N_JOBS (default -1 = todos los cores). Antes
    estaba hardcoded a 1, lo que desperdiciaba cores cuando se entrena UNA
    sola variedad (caso comun: `task train VARIETIES=POP`). Para multi-variedad
    en paralelo (--parallel-varieties >1) setear MODEL_N_JOBS=1 para evitar
    oversubscription (parallel-varieties * n_jobs > cores)."""


def _common_kwargs() -> dict:
    """Kwargs compartidos por las factories de XGB y LGB.

    Centraliza:
      - `random_state` desde config (reproducibilidad).
      - `n_jobs` desde env var MODEL_N_JOBS (default -1 = todos los cores).

    Los kwargs especificos del backend (objective, verbose/verbosity,
    tree_method, subsample_freq, ...) se mergean en el call site.
    """
    return dict(
        random_state=RANDOM_STATE,
        n_jobs=int(os.environ.get("MODEL_N_JOBS", "-1")),
    )


def _log1p_cap(y):
    """Cap suave (CAP_PERCENTILE) y luego log1p para estabilizar varianza."""
    y = np.asarray(y, dtype=float)
    cap = float(np.nanpercentile(y, CAP_PERCENTILE))
    return np.log1p(np.minimum(y, cap))


def _expm1(y):
    return np.expm1(np.asarray(y, dtype=float))


def wrap_with_log_target(regressor: BaseEstimator) -> TransformedTargetRegressor:
    """Envuelve un regresor con log1p+cap suave.

    `check_inverse=False` porque el cap no es invertible (perdida de info
    intencionada en el lado de y_train para ganar generalizacion).
    """
    return TransformedTargetRegressor(
        regressor=regressor,
        func=_log1p_cap,
        inverse_func=_expm1,
        check_inverse=False,
    )
