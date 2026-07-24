"""Baselines resistentes para demostrar que el modelo agrega valor.

Un modelo avanzado no se aprueba solo porque su error sea bajo: debe superar
una regla simple que pueda operar con el mismo historial. El baseline usa una
cascada de medianas ``FUNDO+FORMATO -> FUNDO -> FORMATO -> global`` aprendida
exclusivamente en train. Es barato, interpretable y robusto a colas largas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class HierarchicalMedianRegressor(BaseEstimator, RegressorMixin):
    """Baseline de medianas jerárquicas con fallback global."""

    def __init__(
        self,
        levels: tuple[tuple[str, ...], ...] = (
            ("FUNDO", "FORMATO"),
            ("FUNDO",),
            ("FORMATO",),
        ),
        min_group_size: int = 10,
    ) -> None:
        self.levels = levels
        self.min_group_size = min_group_size

    @staticmethod
    def _keys(X: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
        if len(columns) == 1:
            return X[columns[0]].astype("string").fillna("__MISSING__").astype(str)
        return (
            X.loc[:, list(columns)]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
            .agg("__".join, axis=1)
        )

    def fit(self, X: pd.DataFrame, y) -> HierarchicalMedianRegressor:
        y_series = pd.Series(np.asarray(y, dtype=float), index=X.index)
        finite = np.isfinite(y_series.to_numpy())
        if not finite.any():
            raise ValueError("HierarchicalMedianRegressor requiere al menos un target finito")

        X_fit = X.loc[finite]
        y_fit = y_series.loc[finite]
        self.global_median_ = float(y_fit.median())
        self.level_maps_: list[tuple[tuple[str, ...], dict[str, float]]] = []

        for columns in self.levels:
            if not all(column in X_fit.columns for column in columns):
                continue
            keys = self._keys(X_fit, columns)
            stats = pd.DataFrame({"key": keys, "target": y_fit}).groupby("key")[
                "target"
            ].agg(["median", "count"])
            valid = stats[stats["count"] >= self.min_group_size]["median"]
            self.level_maps_.append(
                (columns, {str(key): float(value) for key, value in valid.items()})
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "global_median_"):
            raise RuntimeError("HierarchicalMedianRegressor no fue ajustado")

        pred = pd.Series(np.nan, index=X.index, dtype=float)
        for columns, mapping in self.level_maps_:
            unresolved = pred.isna()
            if not unresolved.any():
                break
            keys = self._keys(X.loc[unresolved], columns)
            pred.loc[unresolved] = keys.map(mapping)
        return pred.fillna(self.global_median_).to_numpy(dtype=float)


def mae_skill_score(model_mae: float, baseline_mae: float) -> float:
    """Skill relativo: 1 es perfecto, 0 empata el baseline, <0 es peor."""

    if not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return float("nan")
    return float(1.0 - model_mae / baseline_mae)
