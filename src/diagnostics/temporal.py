"""Analisis temporal: autocorrelacion, estacionariedad, drift entre anios.

Hipotesis del proyecto:
    - El target KG/JR_H es altamente estacional (cosecha anual).
    - Los lag features estan diseniados para capturar esa estacionalidad.
    - SI los lags hacen su trabajo, los residuos del champion deberian no
      tener autocorrelacion (DW≈2, Ljung-Box no rechaza).

Este modulo opera en DOS modos:
    1. RAW (sobre features pre-fit): autocorrelacion, ACF/PACF, ADF/KPSS
       sobre cada variable temporal. Util para Phase 2 EDA pre-pipeline.
    2. RESIDUAL (sobre residuos del modelo refit): DW + Ljung-Box. Se
       integra en step_05_evaluate (Phase 6) cuando el champion esta listo.

Drift entre anios:
    Population Stability Index (PSI) y Kolmogorov-Smirnov (KS) entre la
    distribucion de cada variable en anio_t vs anio_t+1. Si PSI > 0.25 o
    KS rechaza, hay drift estructural -> features pueden ser inestables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.diagnostics.statistical_tests import (
    TestResult,
    adf_test,
    durbin_watson,
    kpss_test,
    ljung_box,
)


@dataclass
class TemporalProfile:
    name: str
    acf: list[float]  # autocorrelaciones lag 0..k
    pacf: list[float]  # parciales
    significant_lags: list[int]  # lags donde |corr| > 1.96/sqrt(n)
    durbin_watson: TestResult
    ljung_box_10: TestResult
    adf: TestResult
    kpss: TestResult
    stl_trend_strength: float | None = None
    stl_seasonal_strength: float | None = None


@dataclass
class DriftReport:
    variable: str
    year_pairs: list[tuple[int, int]]
    psi_values: dict[tuple[int, int], float] = field(default_factory=dict)
    ks_pvalues: dict[tuple[int, int], float] = field(default_factory=dict)
    max_psi: float = 0.0
    drift_severity: str = "ninguno"  # "ninguno"|"moderado"|"severo"


def compute_acf_pacf(x: pd.Series, n_lags: int = 20) -> tuple[list[float], list[float], list[int]]:
    """Devuelve (acf, pacf, significant_lags).

    significant_lags = lags donde |acf| > 1.96/sqrt(n) (95% CI bajo H0=ruido blanco).
    Excluye lag 0 (siempre 1.0 trivialmente).
    """
    from statsmodels.tsa.stattools import acf, pacf

    x_clean = x.dropna()
    n = len(x_clean)
    if n < n_lags + 5:
        return [], [], []
    actual_lags = min(n_lags, n // 2 - 1)
    try:
        acf_vals = acf(x_clean.values, nlags=actual_lags, fft=True)
        pacf_vals = pacf(x_clean.values, nlags=actual_lags, method="ols")
        threshold = 1.96 / np.sqrt(n)
        sig = [int(i) for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > threshold]
        return list(map(float, acf_vals)), list(map(float, pacf_vals)), sig
    except Exception:
        return [], [], []


def stl_strengths(x: pd.Series, period: int) -> tuple[float | None, float | None]:
    """Descomposicion STL: devuelve (trend_strength, seasonal_strength).

    Definicion (Hyndman & Athanasopoulos):
        F_trend    = max(0, 1 - var(remainder) / var(trend + remainder))
        F_seasonal = max(0, 1 - var(remainder) / var(seasonal + remainder))
    Valores cercanos a 1 = componente fuerte. <0.3 = casi ruido.
    """
    from statsmodels.tsa.seasonal import STL

    x_clean = x.dropna()
    if len(x_clean) < 2 * period or period < 2:
        return None, None
    try:
        result = STL(x_clean.values, period=period, robust=True).fit()
        var_remainder = np.var(result.resid)
        var_tr = np.var(result.trend + result.resid)
        var_sr = np.var(result.seasonal + result.resid)
        ts = max(0.0, 1.0 - var_remainder / var_tr) if var_tr > 0 else 0.0
        ss = max(0.0, 1.0 - var_remainder / var_sr) if var_sr > 0 else 0.0
        return float(ts), float(ss)
    except Exception:
        return None, None


def profile_temporal(name: str, x: pd.Series, period: int = 12) -> TemporalProfile:
    """Genera el perfil temporal completo para una variable.

    `period` = perido estacional asumido (12 = mensual). Si la serie viene
    diaria, ajustar a 365; semanal, 52.
    """
    acf_v, pacf_v, sig = compute_acf_pacf(x, n_lags=20)
    ts, ss = stl_strengths(x, period=period)
    return TemporalProfile(
        name=name,
        acf=acf_v,
        pacf=pacf_v,
        significant_lags=sig,
        durbin_watson=durbin_watson(x),
        ljung_box_10=ljung_box(x, lags=10),
        adf=adf_test(x),
        kpss=kpss_test(x),
        stl_trend_strength=ts,
        stl_seasonal_strength=ss,
    )


# ---------------------------------------------------------------------------
# Drift entre anios
# ---------------------------------------------------------------------------
def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    n_bins: int = 10,
) -> float:
    """PSI entre dos distribuciones, bineado por quantiles del 'expected'.

    Heuristica de interpretacion (industria credito/risk):
        PSI < 0.10 -> sin drift
        0.10-0.25  -> drift moderado
        > 0.25     -> drift severo
    """
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if len(expected) < 50 or len(actual) < 50:
        return float("nan")
    try:
        # Bins por quantiles del expected, evitando duplicados
        breakpoints = np.unique(np.quantile(expected.values, np.linspace(0, 1, n_bins + 1)))
        if len(breakpoints) < 3:
            return float("nan")
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        e_counts, _ = np.histogram(expected.values, bins=breakpoints)
        a_counts, _ = np.histogram(actual.values, bins=breakpoints)
        e_pct = e_counts / e_counts.sum()
        a_pct = a_counts / a_counts.sum()
        # Smoothing: evitar log(0)
        eps = 1e-6
        e_pct = np.clip(e_pct, eps, 1.0)
        a_pct = np.clip(a_pct, eps, 1.0)
        psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
        return float(psi)
    except Exception:
        return float("nan")


def kolmogorov_smirnov_2sample(a: pd.Series, b: pd.Series) -> float:
    """KS test entre dos muestras. Devuelve p-value."""
    from scipy.stats import ks_2samp

    a_clean = pd.to_numeric(a, errors="coerce").dropna()
    b_clean = pd.to_numeric(b, errors="coerce").dropna()
    if len(a_clean) < 30 or len(b_clean) < 30:
        return float("nan")
    try:
        return float(ks_2samp(a_clean.values, b_clean.values).pvalue)
    except Exception:
        return float("nan")


def drift_report(
    df: pd.DataFrame,
    variable: str,
    year_col: str = "ANIO",
    date_col: str | None = None,
) -> DriftReport:
    """Reporta drift de `variable` entre anios consecutivos.

    Si `year_col` no existe pero hay `date_col`, lo deriva. Si no hay nada,
    devuelve un DriftReport vacio (sin year_pairs).
    """
    if year_col not in df.columns:
        if date_col and date_col in df.columns:
            df = df.copy()
            df[year_col] = pd.to_datetime(df[date_col], errors="coerce").dt.year
        else:
            return DriftReport(variable=variable, year_pairs=[])

    if variable not in df.columns:
        return DriftReport(variable=variable, year_pairs=[])

    years = sorted(df[year_col].dropna().unique().tolist())
    if len(years) < 2:
        return DriftReport(variable=variable, year_pairs=[])

    pairs = list(zip(years[:-1], years[1:], strict=True))
    psi_vals: dict[tuple[int, int], float] = {}
    ks_pvals: dict[tuple[int, int], float] = {}

    for y1, y2 in pairs:
        s1 = df.loc[df[year_col] == y1, variable]
        s2 = df.loc[df[year_col] == y2, variable]
        psi_vals[(int(y1), int(y2))] = population_stability_index(s1, s2)
        ks_pvals[(int(y1), int(y2))] = kolmogorov_smirnov_2sample(s1, s2)

    psi_clean = [v for v in psi_vals.values() if not np.isnan(v)]
    max_psi = max(psi_clean) if psi_clean else 0.0
    if max_psi > 0.25:
        severity = "severo"
    elif max_psi > 0.10:
        severity = "moderado"
    else:
        severity = "ninguno"

    return DriftReport(
        variable=variable,
        year_pairs=[(int(a), int(b)) for a, b in pairs],
        psi_values=psi_vals,
        ks_pvalues=ks_pvals,
        max_psi=max_psi,
        drift_severity=severity,
    )
