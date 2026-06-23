"""Factory de LGBMRegressor con defaults sanos.

Devuelve un `TransformedTargetRegressor` que aplica log1p+cap-p99.5 a y
durante el fit (CV-safe: cap calculado en cada fold). El predict ya
invierte al espacio original (KG/JR_H), asi que las metricas quedan
en unidades del target.
"""

from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor

from src.config import N_ESTIMATORS_MAX
from src.step_04_train.early_stopping import EarlyStoppingLGBMRegressor
from src.step_04_train.target_transform import (
    _PARALLELISM_DOCSTRING,
    _common_kwargs,
    wrap_with_log_target,
)

_BACKEND_SPECIFIC = dict(
    verbose=-1,
    objective="regression_l1",
    subsample_freq=1,
    # n_estimators NO se tunea (rev. 8): techo alto + early stopping interno
    # (EarlyStoppingLGBMRegressor) decide el corte real por fold/trial.
    n_estimators=N_ESTIMATORS_MAX,
)


def get_lgb_model(**overrides) -> TransformedTargetRegressor:
    """LGBMRegressor envuelto en TransformedTargetRegressor (log1p + cap p99.5).

    `objective='regression_l1'` (= MAE nativo). Alinea la loss interna con la
    metrica de seleccion (MAE en Optuna y MAPE de negocio para el campeon).

    Usa `EarlyStoppingLGBMRegressor`: cada fit carva un holdout interno y
    corta los arboles cuando el MAE de validacion deja de mejorar
    (config EARLY_STOPPING_*). Ver step_04_train/early_stopping.py.

    {parallelism}

    `subsample_freq=1` fijo: aplicar bagging cada arbol (no cada N). Antes
    estaba en search space 1-7; valores altos anulaban el bagging porque solo
    1/N arboles veia un sample (resto entrenaba con dataset completo).
    """
    params = _common_kwargs() | _BACKEND_SPECIFIC
    params.update(overrides)
    return wrap_with_log_target(EarlyStoppingLGBMRegressor(**params))


get_lgb_model.__doc__ = get_lgb_model.__doc__.format(parallelism=_PARALLELISM_DOCSTRING)
