"""Indicadores binarios de missingness (sklearn-compat).

Crea columnas `<col>__MISS` (1 si el valor original era NaN, 0 en caso
contrario) ANTES de que el imputer rellene los huecos. La ausencia es
informativa: en el dataset POP, las filas con P/BAYA NaN tenian MAPE
17.3% vs 15.6% en filas observadas, asi que esa senal vale la pena
exponerla al modelo.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import MISSING_FLAG_COLS
from src.step_02_clean._helpers import resolve_cols

SUFFIX = "__MISS"
AGG_COL = "N_MISS_RAW"


class MissingFlagger(BaseEstimator, TransformerMixin):
    """Agrega `<col>__MISS` por columna mas un agregado N_MISS_RAW.

    Va al INICIO del preprocesador (antes del imputer), de modo que el
    modelo reciba la flag y el valor imputado.

    `N_MISS_RAW` es la suma de las flags por fila: senal de "fila
    estructuralmente incompleta". Un arbol superficial (max_depth=4
    como en LGB rev 7.3) necesitaria varios splits para reconstruir
    ese conteo desde las flags individuales; exponerlo directo le
    ahorra capacidad y reduce varianza entre folds.
    """

    def __init__(self, cols: list[str] | None = None):
        self.cols = cols

    def _resolve(self, X: pd.DataFrame) -> list[str]:
        return resolve_cols(
            X,
            self.cols,
            MISSING_FLAG_COLS,
            "MissingFlagger",
            require_all=False,
        )

    def fit(self, X: pd.DataFrame, y=None) -> MissingFlagger:
        self.cols_ = self._resolve(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        flag_cols: list[str] = []
        for c in self.cols_:
            flag_name = f"{c}{SUFFIX}"
            X[flag_name] = X[c].isna().astype(int)
            flag_cols.append(flag_name)
        # Agregada: cuantas raw cols tenian NaN en esta fila. Si no hay
        # flags (cols_ vacia), N_MISS_RAW=0 por construccion.
        if flag_cols:
            X[AGG_COL] = X[flag_cols].sum(axis=1).astype(int)
        else:
            X[AGG_COL] = 0
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        flags = [f"{c}{SUFFIX}" for c in getattr(self, "cols_", [])]
        return base + flags + [AGG_COL]
