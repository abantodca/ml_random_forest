"""Imputacion de valores faltantes (sklearn-compat).

Estrategia robusta:
    - Numericas con < `median_threshold` de missing -> KNNImputer.
    - Si no quedan columnas con datos suficientes para entrenar el KNN
      (todo missing en alguna fila) cae a SimpleImputer median.
    - Permite tunear `n_neighbors` desde Optuna via set_params.

Escalado INTERNO con RobustScaler
---------------------------------
El KNN usa distancia euclidiana entre filas; sin escalar, KG/HA (escala
100-1000) domina sobre DPC (1-10) o %INDUS (0-1) y los "vecinos" se
eligen casi solo por magnitud bruta. RobustScaler (mediana/IQR) es
robusto a las colas largas que el OutlierCapper documenta para este
dataset y devuelve features comparables. El scaler se aplica solo
INTERNAMENTE: el output del transform vuelve a escala original via
inverse_transform, asi el modelo aguas abajo (XGB/LGB) recibe
EXACTAMENTE los mismos valores que antes para filas sin NaN; solo las
filas con NaN reciben imputaciones mas correctas.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.config import NUMERIC_FEATURES
from src.step_02_clean._helpers import resolve_cols


class CustomKNNImputer(BaseEstimator, TransformerMixin):
    """Wrapper sklearn-compat sobre `sklearn.impute.KNNImputer`.

    Imputa solo las columnas numericas indicadas y conserva las demas
    intactas (incluidas categoricas y la columna de fecha), de modo que
    pueda encadenarse con transformadores posteriores.

    Parametros
    ----------
    n_neighbors : int
        Vecinos del KNN. Tuneable.
    weights : {'uniform', 'distance'}
        Ponderacion de vecinos.
    add_median_fallback : bool
        Si True, primero rellena con la mediana columnas con missing
        ratio > `fallback_threshold` (KNN no aporta cuando hay > 50%
        missing) y luego corre KNN sobre el resto.
    fallback_threshold : float
        Ratio de missing por encima del cual se usa mediana antes que KNN.
    numeric_cols : list[str] | None
        Columnas a imputar; si None usa `config.NUMERIC_FEATURES`.
    """

    _DEFAULT_FALLBACK_THRESHOLD = float(os.environ.get("IMPUTER_KNN_THRESHOLD", "0.30"))

    def __init__(
        self,
        n_neighbors: int = 10,
        weights: str = "distance",
        add_median_fallback: bool = True,
        fallback_threshold: float = _DEFAULT_FALLBACK_THRESHOLD,
        numeric_cols: list[str] | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.add_median_fallback = add_median_fallback
        self.fallback_threshold = fallback_threshold
        self.numeric_cols = numeric_cols

    def _resolve_cols(self, X: pd.DataFrame) -> list[str]:
        return resolve_cols(X, self.numeric_cols, NUMERIC_FEATURES, "CustomKNNImputer")

    _GROUP_MEDIAN_MIN_OBS = 10

    def fit(self, X: pd.DataFrame, y=None) -> CustomKNNImputer:
        from src.config import IMPUTER_GROUP_MEDIAN

        cols = self._resolve_cols(X)
        self.numeric_cols_ = cols

        miss_ratio = X[cols].isna().mean()
        if self.add_median_fallback:
            self.median_cols_ = miss_ratio[miss_ratio > self.fallback_threshold].index.tolist()
        else:
            self.median_cols_ = []
        self.knn_cols_ = [c for c in cols if c not in self.median_cols_]

        if self.median_cols_:
            self.median_imputer_ = SimpleImputer(strategy="median")
            self.median_imputer_.fit(X[self.median_cols_])

        self.group_medians_: dict = {}
        self.fundo_medians_: dict = {}
        if (
            IMPUTER_GROUP_MEDIAN
            and self.median_cols_
            and "FUNDO" in X.columns
            and "FORMATO" in X.columns
        ):
            for col in self.median_cols_:
                ff_med = X.groupby([X["FUNDO"].astype(str), X["FORMATO"].astype(str)])[col].agg(
                    ["median", "count"]
                )
                ok_ff = ff_med[ff_med["count"] >= self._GROUP_MEDIAN_MIN_OBS]
                self.group_medians_[col] = {
                    f"{f}__{fmt}": m for (f, fmt), m in ok_ff["median"].items()
                }
                f_med = X.groupby(X["FUNDO"].astype(str))[col].agg(["median", "count"])
                ok_f = f_med[f_med["count"] >= self._GROUP_MEDIAN_MIN_OBS]
                self.fundo_medians_[col] = ok_f["median"].to_dict()

        if self.knn_cols_:
            self.knn_imputer_ = KNNImputer(
                n_neighbors=int(self.n_neighbors),
                weights=self.weights,
            )
            self._knn_fit_cols_ = cols
            X_knn = X[cols].copy()
            if self.median_cols_:
                X_knn[self.median_cols_] = self.median_imputer_.transform(X[self.median_cols_])
            self.scaler_ = RobustScaler()
            X_knn_scaled = self.scaler_.fit_transform(X_knn)
            self.knn_imputer_.fit(X_knn_scaled)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = self.numeric_cols_

        if getattr(self, "group_medians_", None):
            ff_keys = (
                X["FUNDO"].astype(str) + "__" + X["FORMATO"].astype(str)
                if "FUNDO" in X.columns and "FORMATO" in X.columns
                else None
            )
            for col, ff_map in self.group_medians_.items():
                if col not in X.columns:
                    continue
                na = X[col].isna()
                if ff_keys is not None and na.any():
                    X.loc[na, col] = ff_keys[na].map(ff_map)
                    na = X[col].isna()
                if "FUNDO" in X.columns and na.any():
                    X.loc[na, col] = (
                        X.loc[na, "FUNDO"].astype(str).map(self.fundo_medians_.get(col, {}))
                    )

        if self.median_cols_:
            X[self.median_cols_] = self.median_imputer_.transform(X[self.median_cols_])

        if self.knn_cols_:
            X_knn_in = X[cols].copy()
            X_knn_scaled = self.scaler_.transform(X_knn_in)
            X_knn_imputed_scaled = self.knn_imputer_.transform(X_knn_scaled)
            X_knn_out = self.scaler_.inverse_transform(X_knn_imputed_scaled)
            knn_idx = [cols.index(c) for c in self.knn_cols_]
            for i, col in zip(knn_idx, self.knn_cols_, strict=True):
                X[col] = X_knn_out[:, i]

        for c in cols:
            if X[c].isna().any():
                X[c] = X[c].fillna(X[c].median())

        return X

    def get_feature_names_out(self, input_features=None):
        return list(input_features) if input_features is not None else None
