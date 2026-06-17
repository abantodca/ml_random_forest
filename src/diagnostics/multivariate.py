"""Analisis multivariado: VIF, correlation matrix, mutual information.

VIF (Variance Inflation Factor):
    VIF_i = 1 / (1 - R²_i)  donde R²_i viene de regresion OLS de feature_i
    contra el resto. VIF > 10 = multicolinealidad alta. > 5 = revisar.

Mutual Information vs target:
    Captura dependencias no-lineales que correlation linear no ve. La
    implementacion de sklearn (`mutual_info_regression`) usa estimacion
    no parametrica via k-NN.

Correlation matrix:
    Spearman (rangos) por defecto: robusto a outliers y captura monotonias
    no lineales. Pearson como complemento.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CORRELATION_HIGH_THRESHOLD


@dataclass
class VIFResult:
    feature: str
    vif: float
    severity: str  # "ok"|"watch"|"high"


@dataclass
class MutualInfoResult:
    feature: str
    mi: float
    rank: int


@dataclass
class CorrelationMatrix:
    method: str  # "pearson" o "spearman"
    columns: list[str]
    matrix: list[list[float]]  # row-major
    high_pairs: list[tuple]    # (col_a, col_b, corr) con |corr| > CORRELATION_HIGH_THRESHOLD


def compute_vif(X: pd.DataFrame, threshold_high: float = 10.0,
                threshold_watch: float = 5.0) -> list[VIFResult]:
    """Calcula VIF para cada columna numerica de X.

    Excluye columnas constantes / con NaN. Usa pinv para estabilidad
    cuando la matriz X'X es casi singular (no rompe en multicolinealidad
    extrema, devuelve VIF muy alto que el caller detecta).
    """
    numeric = X.select_dtypes(include=[np.number]).copy().dropna()
    if numeric.shape[0] < 30 or numeric.shape[1] < 2:
        return []

    # Drop columnas con varianza cero (VIF indefinido)
    nonconst = [c for c in numeric.columns if numeric[c].std() > 1e-12]
    if len(nonconst) < 2:
        return []
    numeric = numeric[nonconst]

    results: list[VIFResult] = []
    cols = numeric.columns.tolist()
    X_arr = numeric.values

    # Computacion via correlation matrix inversa (mas estable que regresiones
    # individuales para muchas columnas):
    #     VIF_i = (X.corr())^-1 [i, i]
    try:
        corr = np.corrcoef(X_arr.T)
        # pinv en vez de inv para tolerar singularidades
        inv_corr = np.linalg.pinv(corr)
        for i, c in enumerate(cols):
            vif = float(inv_corr[i, i])
            if not np.isfinite(vif) or vif < 1.0:
                vif = float("inf")
            severity = (
                "high" if vif >= threshold_high
                else "watch" if vif >= threshold_watch
                else "ok"
            )
            results.append(VIFResult(feature=c, vif=vif, severity=severity))
    except Exception:
        return []

    return sorted(results, key=lambda r: r.vif, reverse=True)


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    discrete_threshold: int = 10,
) -> list[MutualInfoResult]:
    """Mutual information de cada feature numerica vs target.

    `discrete_threshold`: features con <= N unique values se tratan como
    discretas (cardinalidad baja). El resto, continuas (knn-based MI).
    """
    from sklearn.feature_selection import mutual_info_regression

    df = pd.concat([X, y.rename("__target__")], axis=1).dropna()
    if df.empty:
        return []

    numeric = df.select_dtypes(include=[np.number]).drop(columns=["__target__"], errors="ignore")
    if numeric.empty:
        return []

    discrete_mask = [
        numeric[c].nunique() <= discrete_threshold for c in numeric.columns
    ]
    try:
        mi = mutual_info_regression(
            numeric.values, df["__target__"].values,
            discrete_features=np.array(discrete_mask),
            random_state=42,
        )
    except Exception:
        return []

    pairs = sorted(zip(numeric.columns, mi, strict=True), key=lambda t: t[1], reverse=True)
    return [
        MutualInfoResult(feature=c, mi=float(v), rank=i + 1)
        for i, (c, v) in enumerate(pairs)
    ]


def correlation_matrix(
    X: pd.DataFrame,
    method: str = "spearman",
    high_threshold: float = CORRELATION_HIGH_THRESHOLD,
) -> CorrelationMatrix:
    """Matriz de correlacion + lista de pares con |r| > threshold."""
    numeric = X.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return CorrelationMatrix(method=method, columns=[], matrix=[], high_pairs=[])

    corr = numeric.corr(method=method).fillna(0.0)
    cols = corr.columns.tolist()

    high_pairs: list[tuple] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iat[i, j])
            if abs(r) >= high_threshold:
                high_pairs.append((cols[i], cols[j], r))

    return CorrelationMatrix(
        method=method,
        columns=cols,
        matrix=corr.values.tolist(),
        high_pairs=sorted(high_pairs, key=lambda t: abs(t[2]), reverse=True),
    )
