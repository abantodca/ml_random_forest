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
    high_pairs: list[tuple]  # (col_a, col_b, corr) con |corr| > CORRELATION_HIGH_THRESHOLD


def compute_vif(
    X: pd.DataFrame, threshold_high: float = 10.0, threshold_watch: float = 5.0
) -> list[VIFResult]:
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
                "high" if vif >= threshold_high else "watch" if vif >= threshold_watch else "ok"
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
    """Información mutua de features numéricas, categóricas y temporales.

    Los NaN numéricos se imputan con la mediana de cada columna; no se eliminan
    filas completas por el missing de otra variable. Las categóricas se
    codifican como enteros discretos y las fechas como días desde epoch.
    """
    from sklearn.feature_selection import mutual_info_regression

    target = pd.to_numeric(y, errors="coerce")
    valid_target = target.notna() & np.isfinite(target.to_numpy(dtype=float))
    if not valid_target.any():
        return []

    X_valid = X.loc[valid_target].reset_index(drop=True)
    target_valid = target.loc[valid_target].reset_index(drop=True)
    prepared = pd.DataFrame(index=X_valid.index)
    discrete_mask: list[bool] = []

    for column in X_valid.columns:
        series = X_valid[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            dates = pd.to_datetime(series, errors="coerce")
            fill = dates.dropna().median() if dates.notna().any() else pd.Timestamp("1970-01-01")
            prepared[column] = (
                dates.fillna(fill).astype("int64") / 86_400_000_000_000
            ).astype(float)
            discrete_mask.append(False)
        elif pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            fill = float(numeric.median()) if numeric.notna().any() else 0.0
            prepared[column] = numeric.fillna(fill).astype(float)
            discrete_mask.append(numeric.nunique(dropna=True) <= discrete_threshold)
        else:
            values = series.astype("string").fillna("__MISSING__")
            codes, _ = pd.factorize(values, sort=True)
            prepared[column] = codes.astype(float)
            discrete_mask.append(True)

    if prepared.empty:
        return []

    try:
        mi = mutual_info_regression(
            prepared.values,
            target_valid.values,
            discrete_features=np.array(discrete_mask),
            random_state=42,
        )
    except Exception:
        return []

    pairs = sorted(zip(prepared.columns, mi, strict=True), key=lambda t: t[1], reverse=True)
    return [MutualInfoResult(feature=c, mi=float(v), rank=i + 1) for i, (c, v) in enumerate(pairs)]


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
