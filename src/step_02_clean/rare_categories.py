"""Agrupación fold-safe de categorías con poco soporte.

La frecuencia de una categoría es una estadística aprendida. Calcularla sobre
todo el dataset antes de cross-validation permite que el preprocesamiento vea
la composición del fold de validación. Este transformer aprende los niveles
raros únicamente con las filas recibidas en ``fit`` y aplica exactamente el
mismo mapa en ``transform`` e inferencia.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    ADAPT_RARE_MIN_COUNT,
    RARE_GROUP_COLS,
    RARE_MIN_COUNT,
    RARE_MIN_COUNT_FLOOR,
    RARE_MIN_COUNT_FRAC,
)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Reemplaza niveles infrecuentes o desconocidos por ``other_label``.

    ``min_count=None`` conserva la política adaptativa global. Un override
    explícito viene de :class:`src.variety_config.VarietyConfig`.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        min_count: int | None = None,
        other_label: str = "OTROS",
    ) -> None:
        self.columns = columns
        self.min_count = min_count
        self.other_label = other_label

    def _effective_min_count(self, n_rows: int) -> int:
        if self.min_count is not None:
            return max(1, int(self.min_count))
        if ADAPT_RARE_MIN_COUNT:
            return min(
                RARE_MIN_COUNT,
                max(RARE_MIN_COUNT_FLOOR, round(RARE_MIN_COUNT_FRAC * n_rows)),
            )
        return RARE_MIN_COUNT

    def fit(self, X: pd.DataFrame, y=None) -> RareCategoryGrouper:
        requested = self.columns if self.columns is not None else RARE_GROUP_COLS
        self.columns_ = [column for column in requested if column in X.columns]
        self.min_count_ = self._effective_min_count(len(X))
        self.frequent_levels_: dict[str, frozenset[str]] = {}
        for column in self.columns_:
            values = X[column].astype("string").fillna(self.other_label)
            counts = values.value_counts(dropna=False)
            frequent = counts[counts >= self.min_count_].index.astype(str)
            self.frequent_levels_[column] = frozenset(frequent)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for column in getattr(self, "columns_", []):
            values = X_out[column].astype("string").fillna(self.other_label).astype(str)
            known = self.frequent_levels_.get(column, frozenset())
            X_out[column] = values.where(values.isin(known), self.other_label)
        return X_out

    def get_feature_names_out(self, input_features=None):
        return list(input_features) if input_features is not None else None
