"""Factory de GPBoost (GBDT + efectos aleatorios cruzados FUNDO x FORMATO).

GPBoost (Sigrist 2022) combina arboles LightGBM para las relaciones no
lineales de las features con un modelo de efectos mixtos para la
estructura GRUPAL de los datos: en lugar de aprender FUNDO/FORMATO via
dummies (splits), estima un intercepto aleatorio por grupo con shrinkage
hacia la media global — los grupos chicos no sobreajustan y los grupos
nuevos en inferencia caen al prior (media global) de forma principiada.

Diseno (hereda del piloto 2026-06-13, brazo ganador C2):
  - Efectos aleatorios CRUZADOS: FUNDO + FORMATO como dos factores
    independientes (no la interaccion FUNDO_FORMATO, que fragmenta en
    grupos demasiado chicos).
  - Las dummies de grupo (FUNDO__*, FORMATO__*, FUNDO_FORMATO__*) se
    EXCLUYEN de la matriz de arboles: esa informacion vive en los efectos
    aleatorios; dejarla duplicada hace que arboles y RE compitan.
  - El resto del pipeline es identico a XGB/LGB: mismo preprocesador,
    mismo TransformedTargetRegressor (log1p + cap p99.5), mismo nested CV,
    mismo gate de gap y misma seleccion de campeon (ADR-002).

Restricciones de la libreria (gpboost 1.6.7, verificadas en sonda):
  - `objective` debe ser 'regression' (L2). La likelihood gaussiana del
    GP define la loss; no existe regression_l1 con gp_model. En espacio
    log1p la L2 ya penaliza error RELATIVO, alineada con MAPE de negocio.
  - Bagging no esta soportado con gp_model (bagging_freq=0 obligatorio).
  - `sample_weight` no esta soportado con likelihood gaussiana: el fit lo
    acepta por contrato (TTR lo reenvia) pero lo IGNORA con un warning
    unico. La competencia sigue siendo limpia: la metrica de seleccion
    (OOF business MAPE) se computa igual para los tres backends.
  - El Booster no es picklable directo: `__getstate__`/`__setstate__`
    serializan via `save_model()` JSON (roundtrip bit-exacto verificado),
    asi joblib/MLflow lo tratan como un estimador opaco mas.

Como los demas wrappers, el early stopping vive DENTRO del fit (holdout
interno reproducible, mismas constantes EARLY_STOPPING_* de config) y la
metrica de corte es la neg-log-likelihood del conjunto de validacion con
los efectos aleatorios incluidos (`use_gp_model_for_validation=True`).
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import gpboost as gpb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor

from src.config import (
    EARLY_STOPPING_MIN_ROWS,
    EARLY_STOPPING_ROUNDS,
    N_ESTIMATORS_MAX,
)
from src.step_04_train.early_stopping import _holdout_indices, _seed_of
from src.step_04_train.target_transform import _common_kwargs, wrap_with_log_target

logger = logging.getLogger(__name__)

# Factores de efectos aleatorios cruzados (brazo C2 del piloto). El orden
# importa solo para reproducibilidad del GPModel; ambos entran simetricos.
_RE_FACTOR_PREFIXES: tuple[str, ...] = ("FUNDO__", "FORMATO__")
# Dummies que se excluyen de la matriz de arboles (su informacion vive en
# los efectos aleatorios). Incluye la interaccion explicita si esta activa.
_GROUP_DUMMY_PREFIXES: tuple[str, ...] = (
    "FUNDO__", "FORMATO__", "FUNDO_FORMATO__",
)
# Etiqueta para filas cuyo dummy-block esta todo en 0: categoria no vista
# en train (sentinel implicito del FeatureGenerator) o factor sin columnas
# (variedad mono-fundo cuya dummy constante elimino el variance_filter).
_UNSEEN_LABEL = "__UNSEEN__"

# Warning de sample_weight: una sola vez por proceso (el nested CV hace
# cientos de fits; un warning por fit seria ruido inutil).
_warned_sample_weight = False


def _warn_sample_weight_once() -> None:
    global _warned_sample_weight
    if not _warned_sample_weight:
        logger.warning(
            "GPBoost (likelihood gaussiana) no soporta sample_weight; se "
            "ignoran los pesos para este backend. XGB/LGB si los usan — la "
            "seleccion de campeon compara OOF MAPE, que es insensible a "
            "esta diferencia de capacidades."
        )
        _warned_sample_weight = True


class GPBoostMixedEffectsRegressor(BaseEstimator, RegressorMixin):
    """GPBoost sklearn-compatible: arboles + intercepto aleatorio por grupo.

    Espera el DataFrame YA preprocesado (output pandas del preprocesador):
    deriva las etiquetas de grupo desde los bloques one-hot `FUNDO__*` /
    `FORMATO__*` (argmax por fila; todo-cero -> grupo "no visto"). Eso lo
    hace autocontenido para serializacion: el API carga el mismo pipeline
    y el predict deriva los grupos de la misma transformada, sin canales
    laterales de metadata.

    Los hiperparametros de arboles usan el naming de LightGBM (GPBoost es
    un fork): el search space en `search_spaces.py` mapea 1:1.
    """

    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS_MAX,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        min_data_in_leaf: int = 20,
        min_split_gain: float = 0.0,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        feature_fraction: float = 1.0,
        likelihood: str = "gaussian",
        drop_group_dummies: bool = True,
        random_state: int | None = None,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.min_split_gain = min_split_gain
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.feature_fraction = feature_fraction
        self.likelihood = likelihood
        self.drop_group_dummies = drop_group_dummies
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Grupos desde los bloques one-hot del preprocesador
    # ------------------------------------------------------------------

    @staticmethod
    def _factor_columns(columns: list[str], prefix: str) -> list[str]:
        # "FUNDO__" NO matchea "FUNDO_FORMATO__x" (un solo '_' tras FUNDO),
        # los tres prefijos son mutuamente excluyentes por construccion.
        return [c for c in columns if c.startswith(prefix)]

    def _derive_groups(self, X: pd.DataFrame) -> np.ndarray:
        """Matriz (n, n_factores) de etiquetas string por fila.

        Usa `self.group_cols_` (fijado en fit) para que predict derive los
        grupos con EXACTAMENTE las mismas columnas/orden que el fit, aunque
        el caller pase columnas extra.
        """
        n = len(X)
        labels: list[np.ndarray] = []
        for prefix in _RE_FACTOR_PREFIXES:
            cols = self.group_cols_[prefix]
            if not cols:
                # Factor sin dummies (p.ej. variedad mono-fundo): un unico
                # grupo constante == intercepto; GPModel lo absorbe.
                labels.append(np.full(n, _UNSEEN_LABEL, dtype=object))
                continue
            block = X[cols].to_numpy(dtype=float)
            cats = np.array([c[len(prefix):] for c in cols], dtype=object)
            row_label = cats[block.argmax(axis=1)]
            row_label = np.where(block.max(axis=1) > 0, row_label, _UNSEEN_LABEL)
            labels.append(row_label.astype(object))
        return np.column_stack(labels)

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def _train_params(self) -> dict:
        params: dict = {
            "objective": "regression",  # unico compatible con gp_model
            "verbose": -1,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_gain_to_split": self.min_split_gain,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "feature_fraction": self.feature_fraction,
            "bagging_freq": 0,  # bagging no soportado con gp_model
        }
        if self.random_state is not None:
            params["seed"] = int(self.random_state)
        if self.n_jobs is not None and self.n_jobs > 0:
            params["num_threads"] = int(self.n_jobs)
        return params

    def fit(self, X, y, sample_weight=None):  # noqa: D102 — contrato sklearn
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "GPBoostMixedEffectsRegressor requiere un DataFrame con las "
                "dummies del preprocesador (set_output pandas) para derivar "
                f"los grupos; recibido {type(X).__name__}."
            )
        if sample_weight is not None:
            _warn_sample_weight_once()

        columns = list(X.columns)
        self.group_cols_ = {
            p: self._factor_columns(columns, p) for p in _RE_FACTOR_PREFIXES
        }
        self.feature_cols_ = (
            [c for c in columns if not c.startswith(_GROUP_DUMMY_PREFIXES)]
            if self.drop_group_dummies
            else columns
        )

        groups = self._derive_groups(X)
        Xm = X[self.feature_cols_].to_numpy(dtype=np.float64)
        y_arr = np.asarray(y, dtype=float)
        params = self._train_params()
        n = len(y_arr)

        if n < EARLY_STOPPING_MIN_ROWS:
            # Mismo fallback que EarlyStopping{LGBM,XGB}Regressor: con pocas
            # filas el holdout no es senal; fit directo sin early stopping.
            gp_model = gpb.GPModel(group_data=groups, likelihood=self.likelihood)
            self.booster_ = gpb.train(
                params=params,
                train_set=gpb.Dataset(Xm, y_arr),
                gp_model=gp_model,
                num_boost_round=self.n_estimators,
            )
            return self

        tr, va = _holdout_indices(n, _seed_of(self))
        try:
            # La validacion evalua con los efectos aleatorios incluidos (si
            # no, el corte ignoraria la mitad del modelo).
            self.booster_ = self._train_es(
                params, Xm, y_arr, groups, tr, va, gp_validation=True,
            )
        except gpb.basic.GPBoostError as exc:
            # Configs extremas del TPE pueden desestabilizar el solver del GP
            # ("Nan or Inf ... Conjugate Gradient"). Reintento UNA vez con
            # validacion sin gp_model: el GP sigue entrenando, solo el corte
            # de early stopping pasa a metrica L2 sobre los efectos fijos.
            # Configs asi puntuan mal de todos modos; el reintento evita que
            # un trial patologico tumbe el nested CV completo.
            if "Nan" not in str(exc) and "Inf" not in str(exc):
                raise
            logger.warning(
                "GPBoost: inestabilidad numerica con validacion GP (%s); "
                "reintento con validacion sin gp_model.", exc,
            )
            self.booster_ = self._train_es(
                params | {"metric": "l2"}, Xm, y_arr, groups, tr, va,
                gp_validation=False,
            )
        return self

    def _train_es(
        self,
        params: dict,
        Xm: np.ndarray,
        y_arr: np.ndarray,
        groups: np.ndarray,
        tr: np.ndarray,
        va: np.ndarray,
        *,
        gp_validation: bool,
    ):
        """Un intento de gpb.train con early stopping sobre el holdout.

        El GPModel se construye ADENTRO para que el reintento del caller
        parta de un estado limpio (el handle del intento fallido puede
        quedar corrupto tras el error del solver).
        """
        gp_model = gpb.GPModel(group_data=groups[tr], likelihood=self.likelihood)
        if gp_validation:
            gp_model.set_prediction_data(group_data_pred=groups[va])
        train_set = gpb.Dataset(Xm[tr], y_arr[tr])
        valid_set = gpb.Dataset(Xm[va], y_arr[va], reference=train_set)
        return gpb.train(
            params=params | {"use_gp_model_for_validation": gp_validation},
            train_set=train_set,
            gp_model=gp_model,
            num_boost_round=self.n_estimators,
            valid_sets=[valid_set],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

    def predict(self, X) -> np.ndarray:  # noqa: D102 — contrato sklearn
        groups = self._derive_groups(X)
        Xm = X[self.feature_cols_].to_numpy(dtype=np.float64)
        out = self.booster_.predict(
            data=Xm,
            group_data_pred=groups,
            predict_var=False,
            pred_latent=False,
        )
        arr = out["response_mean"] if isinstance(out, dict) else out
        return np.asarray(arr, dtype=float)

    # ------------------------------------------------------------------
    # Serializacion (joblib/MLflow): el Booster de gpboost no es picklable
    # directo; round-trip via save_model() JSON (incluye el gp_model y
    # respeta best_iteration del early stopping).
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        booster = state.pop("booster_", None)
        if booster is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "gpb_model.json"
                booster.save_model(str(path))
                state["_booster_model_json_"] = path.read_text(encoding="utf-8")
        return state

    def __setstate__(self, state: dict) -> None:
        model_json = state.pop("_booster_model_json_", None)
        self.__dict__.update(state)
        if model_json is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "gpb_model.json"
                path.write_text(model_json, encoding="utf-8")
                self.booster_ = gpb.Booster(model_file=str(path))


def get_gpb_model(**overrides) -> TransformedTargetRegressor:
    """GPBoost envuelto en TransformedTargetRegressor (log1p + cap p99.5).

    Mismo wrapper de target que XGB/LGB para que los tres backends compitan
    bit-comparables: el predict ya devuelve KG/JR_H en espacio original.

    `n_jobs` lee MODEL_N_JOBS igual que los otros backends (via
    `_common_kwargs`); GPBoost interpreta n_jobs<=0 como "todos los cores"
    (se omite num_threads y LightGBM usa su default).
    """
    params = _common_kwargs() | dict(n_estimators=N_ESTIMATORS_MAX)
    params.update(overrides)
    return wrap_with_log_target(GPBoostMixedEffectsRegressor(**params))
