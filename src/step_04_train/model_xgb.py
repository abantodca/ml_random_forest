"""Factory de XGBRegressor con defaults sanos.

Devuelve un `TransformedTargetRegressor` que aplica log1p+cap-p99.5 a y
durante el fit (CV-safe: cap calculado en cada fold). El predict ya
invierte al espacio original (KG/JR_H).
"""
from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor

from src.config import N_ESTIMATORS_MAX
from src.step_04_train.early_stopping import EarlyStoppingXGBRegressor
from src.step_04_train.target_transform import (
    _PARALLELISM_DOCSTRING,
    _common_kwargs,
    wrap_with_log_target,
)

_BACKEND_SPECIFIC = dict(
    verbosity=0,
    tree_method="hist",
    objective="reg:absoluteerror",
    # n_estimators NO se tunea (rev. 7): techo alto + early stopping interno
    # (EarlyStoppingXGBRegressor) decide el corte real por fold/trial.
    n_estimators=N_ESTIMATORS_MAX,
)


def get_xgb_model(**overrides) -> TransformedTargetRegressor:
    """XGBRegressor envuelto en TransformedTargetRegressor (log1p + cap p99.5).

    `objective='reg:absoluteerror'` (XGB >=1.7) -> MAE nativo, alineado con la
    metrica de seleccion (MAE de Optuna y MAPE de negocio del campeon). Antes
    se usaba el default `reg:squarederror` (L2), lo que entrenaba penalizando
    cuadraticamente y luego se evaluaba en MAE -> objetivo desalineado.

    Usa `EarlyStoppingXGBRegressor`: cada fit carva un holdout interno y
    corta los arboles cuando el MAE de validacion deja de mejorar
    (config EARLY_STOPPING_*). Ver step_04_train/early_stopping.py.

    {parallelism}
    """
    params = _common_kwargs() | _BACKEND_SPECIFIC
    params.update(overrides)
    return wrap_with_log_target(EarlyStoppingXGBRegressor(**params))


get_xgb_model.__doc__ = get_xgb_model.__doc__.format(parallelism=_PARALLELISM_DOCSTRING)
