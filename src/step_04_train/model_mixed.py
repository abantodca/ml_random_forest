"""Candidato experimental: componente lineal + ML sobre residuos.

No pretende ser un modelo de efectos mixtos inferencial. Es un híbrido
aditivo reproducible: Ridge captura nivel/tendencias suaves y
HistGradientBoosting aprende la estructura no lineal que queda en los
residuos. Solo entra al registry de backends con
``ENABLE_EXPERIMENTAL_MIXED_BACKEND=1`` y queda sujeto a los mismos folds,
baseline y quality gates que XGB/LGB.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.config import RANDOM_STATE
from src.step_04_train.target_transform import wrap_with_log_target


class ResidualHybridRegressor(RegressorMixin, BaseEstimator):
    """Ridge como componente estructural y boosting sobre sus residuos."""

    def __init__(
        self,
        linear_alpha: float = 10.0,
        learning_rate: float = 0.05,
        max_iter: int = 300,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.linear_alpha = linear_alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        y_arr = np.asarray(y, dtype=float)
        self.linear_ = make_pipeline(
            StandardScaler(),
            Ridge(alpha=self.linear_alpha),
        )
        linear_fit = {"ridge__sample_weight": sample_weight} if sample_weight is not None else {}
        self.linear_.fit(X, y_arr, **linear_fit)
        residual = y_arr - self.linear_.predict(X)
        self.residual_ = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=True,
            random_state=self.random_state,
        )
        self.residual_.fit(X, residual, sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, ("linear_", "residual_"))
        return np.asarray(
            self.linear_.predict(X) + self.residual_.predict(X),
            dtype=float,
        )


def get_mixed_model(**overrides):
    """Factory compatible con el target transform y el pipeline existentes."""
    return wrap_with_log_target(ResidualHybridRegressor(**overrides))
