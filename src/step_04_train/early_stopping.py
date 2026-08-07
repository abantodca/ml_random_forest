"""Wrappers de LGBM/XGB con early stopping sobre un holdout interno.

Problema que resuelve: en un sklearn Pipeline + TransformedTargetRegressor
no hay forma limpia de pasar `eval_set` desde fuera (el X de validacion
tendria que pasar por el preprocesador del fold y el y por la transformacion
log1p del TTR). Por eso el search space capaba `n_estimators<=1000` "porque
sin early stopping los arboles extra solo memorizan". Estos wrappers
implementan el early stopping DONDE corresponde: dentro del fit del
regresor, cuando X e y ya estan en el espacio final.

Diseno:
  - Subclases SIN `__init__` propio: los hiperparametros, `get_params` y
    `clone()` son identicos al padre (sklearn no pierde params al clonar,
    que es el riesgo clasico de subclasear LGBM/XGB con params nuevos).
  - El comportamiento se controla por constantes de `src.config`
    (EARLY_STOPPING_*), NO son hiperparametros tuneables por Optuna.
  - El holdout usa el tramo final del train, cuyo orden canónico es temporal.
    Así el número de árboles se decide contra observaciones posteriores, no
    contra una mezcla aleatoria de pasado y futuro. Con menos de EARLY_STOPPING_MIN_ROWS
    filas se cae a un fit normal sin early stopping (folds smoke).
  - El wrapper vive DENTRO del TTR: el `y` recibido ya esta en espacio
    log1p+cap, asi que la metrica de corte (l1/mae) es consistente con la
    loss de entrenamiento.
  - Tras el corte, `predict` de ambos backends usa automaticamente la mejor
    iteracion (best_iteration), no los N_ESTIMATORS_MAX arboles.
"""

from __future__ import annotations

import logging

import numpy as np
from lightgbm import LGBMRegressor
from lightgbm import early_stopping as lgb_early_stopping
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.config import (
    EARLY_STOPPING_MIN_ROWS,
    EARLY_STOPPING_REFIT_FULL,
    EARLY_STOPPING_ROUNDS,
    N_ESTIMATORS_MAX,
    RANDOM_STATE,
)
from src.step_04_train.validation_split import temporal_tail_holdout_indices

logger = logging.getLogger(__name__)


def _log_overfit(est, name: str, X_tr, y_tr, X_va, y_va) -> None:
    """Observabilidad (DEBUG) del overfit por fit: corte real del boosting +
    gap MAE train->holdout. Analogo a la instrumentacion de GPBoost. El `y`
    esta en espacio log1p (dentro del TTR), asi que el gap es un indicador
    RELATIVO consistente, no el MAE de negocio. try/except: nunca tumba el fit.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        best_it = getattr(est, "best_iteration_", None)
        if best_it is None:
            best_it = getattr(est, "best_iteration", None)
        mae_tr = float(mean_absolute_error(y_tr, est.predict(X_tr)))
        mae_va = float(mean_absolute_error(y_va, est.predict(X_va)))
        logger.debug(
            "%s early-stop: best_iter=%s MAE_train=%.4f MAE_val=%.4f gap=%.4f",
            name,
            best_it,
            mae_tr,
            mae_va,
            mae_va - mae_tr,
        )
    except Exception as exc:
        logger.debug("%s early-stop: diagnostico no disponible: %s", name, exc)


def _holdout_indices(n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(train, valid)`` con el tramo final como validación.

    ``seed`` se conserva en la firma por compatibilidad con callers y pickles
    anteriores; la partición temporal es determinista.
    """
    return temporal_tail_holdout_indices(n_rows)


def _take(obj, idx: np.ndarray):
    """Slicea filas tanto de DataFrame (.iloc) como de ndarray."""
    if hasattr(obj, "iloc"):
        return obj.iloc[idx]
    return np.asarray(obj)[idx]


def _split_fit_inputs(
    X,
    y,
    sample_weight: np.ndarray | None,
    seed: int,
):
    """Particiona (X, y, sample_weight) en train/holdout para early stopping."""
    n_rows = len(np.asarray(y))
    tr, va = _holdout_indices(n_rows, seed)
    sw_tr = _take(sample_weight, tr) if sample_weight is not None else None
    sw_va = _take(sample_weight, va) if sample_weight is not None else None
    return (
        _take(X, tr),
        _take(y, tr),
        sw_tr,
        _take(X, va),
        _take(y, va),
        sw_va,
    )


def _seed_of(estimator) -> int:
    seed = getattr(estimator, "random_state", None)
    return int(seed) if seed is not None else RANDOM_STATE


def _best_n_trees(est, n_total: int, n_train: int) -> int | None:
    """Nº de arboles a usar en el refit full, o None si no se pudo determinar.

    Lee el corte real del early stopping y lo escala por `n_total / n_train`:
    el corte se descubrio entrenando sobre `n_train` filas y el refit vera
    `n_total`, que admiten algo mas de arboles antes de memorizar. Capado a
    N_ESTIMATORS_MAX (el mismo fusible que usa el fit normal).

    LGBM expone `best_iteration_` (1-based, nº de iteraciones). XGB expone
    `best_iteration` (0-based, indice) -> +1. Cualquier valor no positivo
    significa "early stopping no llego a cortar": devolvemos None y el caller
    deja el fit con holdout tal cual (conservador).
    """
    best = getattr(est, "best_iteration_", None)
    if best is None:
        best = getattr(est, "best_iteration", None)
        if best is not None:
            best = int(best) + 1
    if best is None:
        return None
    best = int(best)
    if best <= 0 or n_train <= 0:
        return None
    scaled = int(round(best * (n_total / n_train)))
    return max(1, min(scaled, N_ESTIMATORS_MAX))


def _refit_on_full(estimator, parent, name, X, y, sample_weight, n_total, n_train, kwargs):
    """Refit sobre el 100% de las filas fijando el nº de arboles ya descubierto.

    Devuelve el estimador refiteado, o `estimator` sin tocar si no se pudo
    determinar el corte. `n_estimators` se restaura despues del fit para que
    `get_params`/`clone` sigan devolviendo la config original (el booster ya
    fiteado conserva sus arboles; predict no consulta `n_estimators`).
    """
    n_trees = _best_n_trees(estimator, n_total, n_train)
    if n_trees is None:
        logger.debug("%s refit-full omitido: early stopping no reporto corte", name)
        return estimator
    original = estimator.n_estimators
    try:
        estimator.n_estimators = n_trees
        refitted = parent.fit(estimator, X, y, sample_weight=sample_weight, **kwargs)
    finally:
        estimator.n_estimators = original
    logger.debug(
        "%s refit-full: n_trees=%d sobre %d filas (corte hallado con %d)",
        name,
        n_trees,
        n_total,
        n_train,
    )
    return refitted


class EarlyStoppingLGBMRegressor(LGBMRegressor):
    """LGBMRegressor que corta arboles con early stopping interno."""

    def fit(self, X, y, sample_weight=None, **kwargs):
        n_rows = len(np.asarray(y))
        if n_rows < EARLY_STOPPING_MIN_ROWS:
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)
        X_tr, y_tr, sw_tr, X_va, y_va, sw_va = _split_fit_inputs(
            X,
            y,
            sample_weight,
            _seed_of(self),
        )
        fitted = super().fit(
            X_tr,
            y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_va)],
            eval_sample_weight=[sw_va] if sw_va is not None else None,
            eval_metric="l1",
            callbacks=[lgb_early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            **kwargs,
        )
        _log_overfit(fitted, "lgb", X_tr, y_tr, X_va, y_va)
        if EARLY_STOPPING_REFIT_FULL:
            fitted = _refit_on_full(
                self,
                LGBMRegressor,
                "lgb",
                X,
                y,
                sample_weight,
                n_rows,
                len(np.asarray(y_tr)),
                kwargs,
            )
        return fitted


class EarlyStoppingXGBRegressor(XGBRegressor):
    """XGBRegressor que corta arboles con early stopping interno.

    XGB (>=2.0) exige `early_stopping_rounds` como parametro del estimador
    (no de fit), y falla si esta seteado sin `eval_set`. Por eso el fit lo
    setea/limpia segun haya holdout o no.
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        n_rows = len(np.asarray(y))
        if n_rows < EARLY_STOPPING_MIN_ROWS:
            self.early_stopping_rounds = None
            return super().fit(X, y, sample_weight=sample_weight, **kwargs)
        self.early_stopping_rounds = EARLY_STOPPING_ROUNDS
        X_tr, y_tr, sw_tr, X_va, y_va, sw_va = _split_fit_inputs(
            X,
            y,
            sample_weight,
            _seed_of(self),
        )
        fitted = super().fit(
            X_tr,
            y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_va)],
            sample_weight_eval_set=[sw_va] if sw_va is not None else None,
            verbose=False,
            **kwargs,
        )
        _log_overfit(fitted, "xgb", X_tr, y_tr, X_va, y_va)
        if EARLY_STOPPING_REFIT_FULL:
            self.early_stopping_rounds = None
            fitted = _refit_on_full(
                self,
                XGBRegressor,
                "xgb",
                X,
                y,
                sample_weight,
                n_rows,
                len(np.asarray(y_tr)),
                kwargs,
            )
        return fitted
