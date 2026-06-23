"""Capping de outliers (sklearn-compat) con metodo configurable.

Soporta dos metodos:
    - 'iqr'        : Q1 - k*IQR  /  Q3 + k*IQR     (k = `factor`)
    - 'percentile' : limites en (lower_q, upper_q) percentiles

`factor` queda como hiperparametro tuneable desde Optuna.

Granularidad por GRUPO (cascade):
    `group_col` acepta str o List[str]. Lista = cascade jerarquica:

        group_col=["FUNDO", "FORMATO"]
        -> intenta bounds (FUNDO, FORMATO) [mas especifico]
        -> fallback a bounds por FUNDO solo
        -> fallback a bounds globales

    Esto resuelve el sesgo cuando UN grupo domina el dataset: en POP, el 86%
    es FORMATO=GRANEL y 72% FUNDO=A9; los bounds globales / por-FUNDO solo
    los reflejan a ellos. Cap por (FUNDO, FORMATO) cierra grupos chicos
    (CLAMSHELL 11 OZ con target μ=2.54) sin contaminar con la cola de
    GRANEL+A9 (target μ=5.35).

    Grupos con n < `min_group_size` caen al siguiente nivel del cascade
    (insuficiente data para IQR confiable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import NUMERIC_FEATURES
from src.step_02_clean._helpers import resolve_cols

_VALID_METHODS = {"iqr", "percentile"}


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Recorta valores extremos columna a columna sobre las numericas.

    Parametros
    ----------
    factor : float
        Para method='iqr': k del rango intercuartil (1.5 estandar, 3 conservador).
        Ignorado en 'percentile'.
    method : {'iqr', 'percentile'}
        Estrategia de capping.
    lower_q, upper_q : float
        Percentiles para method='percentile' (default 0.01 / 0.99).
    numeric_cols : list[str] | None
        Columnas a procesar; si None usa `config.NUMERIC_FEATURES`.
    group_col : str | List[str] | None
        - str (e.g. 'FUNDO'): bounds por valor de esa columna, fallback global.
        - List[str] (e.g. ['FUNDO', 'FORMATO']): cascade jerarquica (mas
          especifico primero, luego prefijos cada vez mas cortos, luego global).
        - None: cap global unico (legacy).
    min_group_size : int
        Tamano minimo de grupo para usar bounds de ese nivel. Si n<umbral,
        ese estrato cae al siguiente nivel del cascade (grupo mas grande
        o global). 30 default empirico: bajo de 30 obs el IQR es muy ruidoso.
    """

    def __init__(
        self,
        factor: float = 3.0,
        method: str = "iqr",
        lower_q: float = 0.01,
        upper_q: float = 0.99,
        numeric_cols: list[str] | None = None,
        group_col: str | list[str] | None = None,
        min_group_size: int = 30,
    ):
        self.factor = factor
        self.method = method
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.numeric_cols = numeric_cols
        self.group_col = group_col
        self.min_group_size = min_group_size

    def _resolve_cols(self, X: pd.DataFrame) -> list[str]:
        return resolve_cols(X, self.numeric_cols, NUMERIC_FEATURES, "OutlierCapper")

    def _compute_bounds(self, X: pd.DataFrame, cols: list[str]) -> dict[str, float]:
        """IQR/percentile bounds sobre un subset de filas. Devuelve dict
        plano {col: {lower, upper}} para guardar en self."""
        if self.method == "iqr":
            q1 = X[cols].quantile(0.25)
            q3 = X[cols].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr
        else:  # percentile
            lower = X[cols].quantile(self.lower_q)
            upper = X[cols].quantile(self.upper_q)
        return {"lower": lower.to_dict(), "upper": upper.to_dict()}

    @staticmethod
    def _composite_keys(X: pd.DataFrame, cols: list[str]) -> pd.Series:
        """Concatena las cols con '__' como key composite (str)."""
        if len(cols) == 1:
            return X[cols[0]].astype(str)
        return X[cols].astype(str).agg("__".join, axis=1)

    def _normalize_group_col(self, X: pd.DataFrame) -> list[str]:
        """Devuelve lista de columnas validas (presentes en X). Acepta
        str / list / None.
        """
        if self.group_col is None:
            return []
        if isinstance(self.group_col, str):
            return [self.group_col] if self.group_col in X.columns else []
        # Lista/tuple: filtra a las que existen, preservando orden
        return [c for c in self.group_col if c in X.columns]

    def fit(self, X: pd.DataFrame, y=None) -> OutlierCapper:
        if self.method not in _VALID_METHODS:
            raise ValueError(f"method debe ser uno de {_VALID_METHODS}, dado '{self.method}'")

        cols = self._resolve_cols(X)
        self.numeric_cols_ = cols

        # Bounds GLOBALES (siempre, fallback final del cascade).
        global_bounds = self._compute_bounds(X, cols)
        self.lower_ = global_bounds["lower"]
        self.upper_ = global_bounds["upper"]

        # Cascade: lista de niveles ordenados de mas especifico a menos.
        # Para group_col=['FUNDO','FORMATO']: niveles = [['FUNDO','FORMATO'],
        # ['FUNDO']]. Cada nivel guarda bounds por su key composite.
        self.group_cols_ = self._normalize_group_col(X)
        self.cascade_: list[
            tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, float]]]
        ] = []
        self.small_groups_: dict[str, list[str]] = {}  # informativo, debug

        if self.group_cols_:
            # Niveles: [primeros n], [primeros n-1], ..., [primeros 1]
            for n_level in range(len(self.group_cols_), 0, -1):
                level_cols = self.group_cols_[:n_level]
                level_label = "_".join(level_cols)
                level_lower: dict[str, dict[str, float]] = {}
                level_upper: dict[str, dict[str, float]] = {}
                small: list[str] = []
                keys = self._composite_keys(X, level_cols)
                for k, X_grp in X.groupby(keys, sort=False):
                    if len(X_grp) < self.min_group_size:
                        small.append(str(k))
                        continue
                    bnd = self._compute_bounds(X_grp, cols)
                    level_lower[str(k)] = bnd["lower"]
                    level_upper[str(k)] = bnd["upper"]
                self.cascade_.append((level_cols, level_lower, level_upper))
                if small:
                    self.small_groups_[level_label] = small

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Path legacy: sin cascade -> cap global (fila a fila igual para todos).
        if not self.cascade_:
            for c in self.numeric_cols_:
                X[c] = np.clip(X[c].astype(float), self.lower_[c], self.upper_[c])
            return X

        # Path cascade: para cada fila buscar el nivel mas especifico que
        # tenga bounds; las que no matchean en NINGUN nivel caen al global.
        # Trabajamos en arrays POSICIONALES (np) para evitar overhead de
        # .loc / index alignment cuando el dataset es grande.
        n = len(X)
        # Inicializar bounds posicionales con los GLOBALES (fallback).
        pos_lower = {c: np.full(n, self.lower_[c], dtype=float) for c in self.numeric_cols_}
        pos_upper = {c: np.full(n, self.upper_[c], dtype=float) for c in self.numeric_cols_}
        matched = np.zeros(n, dtype=bool)

        # Recorre cascade del mas especifico al menos. Filas ya 'matched' en
        # un nivel anterior NO se sobreescriben por niveles menos especificos.
        for level_cols, level_lower, level_upper in self.cascade_:
            if matched.all():
                break
            keys = self._composite_keys(X, level_cols).reset_index(drop=True)
            # `.indices` retorna dict[key, ndarray-posicional] directamente:
            # mas eficiente que `.groups.items()` (que daria pd.Index requiriendo
            # conversion a posicional con np.asarray).
            for k, idx in keys.groupby(keys, sort=False).indices.items():
                k_str = str(k)
                if k_str not in level_lower:
                    continue
                grp_mask = np.zeros(n, dtype=bool)
                grp_mask[idx] = True
                apply_mask = grp_mask & ~matched
                if not apply_mask.any():
                    continue
                lower = level_lower[k_str]
                upper = level_upper[k_str]
                for c in self.numeric_cols_:
                    pos_lower[c][apply_mask] = lower[c]
                    pos_upper[c][apply_mask] = upper[c]
                matched |= apply_mask

        # Aplicar el clip vectorizado por columna.
        for c in self.numeric_cols_:
            col_vals = X[c].astype(float).to_numpy()
            X[c] = np.clip(col_vals, pos_lower[c], pos_upper[c])
        return X

    def get_feature_names_out(self, input_features=None):
        return list(input_features) if input_features is not None else None
