"""Analisis univariado por variable numerica.

Genera, por cada columna numerica + target:
    - Estadisticos descriptivos
    - Tests de normalidad (Shapiro / Anderson-Darling / Jarque-Bera)
    - Box-Cox lambda optimo (sugerencia de transformacion)
    - Outlier scoring univariado (IQR + Z-score + MAD)

Devuelve `VariableProfile` por variable, listo para que el HTML renderer
construya tarjetas visuales.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.diagnostics.statistical_tests import (
    TestResult,
    anderson_darling,
    jarque_bera,
    shapiro_wilk,
)


@dataclass
class VariableProfile:
    name: str
    n: int
    n_missing: int
    miss_ratio: float
    mean: float
    median: float
    std: float
    skew: float
    kurtosis: float
    p01: float
    p99: float
    min: float
    max: float
    n_outliers_iqr: int
    n_outliers_zscore: int
    n_outliers_mad: int
    boxcox_lambda: float | None
    boxcox_recommendation: str
    normality_tests: list[TestResult] = field(default_factory=list)


def _outliers_iqr(x: pd.Series, factor: float = 1.5) -> int:
    """Cuenta outliers por regla IQR (1.5 estandar, 3.0 conservador)."""
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((x < q1 - factor * iqr) | (x > q3 + factor * iqr)).sum())


def _outliers_zscore(x: pd.Series, threshold: float = 3.0) -> int:
    """Cuenta valores |z| > threshold."""
    if x.std() == 0 or pd.isna(x.std()):
        return 0
    z = (x - x.mean()) / x.std()
    return int((z.abs() > threshold).sum())


def _outliers_mad(x: pd.Series, threshold: float = 3.5) -> int:
    """Median Absolute Deviation outliers (mas robusto que z-score).

    Threshold 3.5 corresponde a ~3 sigmas asumiendo normalidad.
    """
    median = x.median()
    mad = (x - median).abs().median()
    if mad == 0:
        return 0
    # 0.6745 = 75-percentile de la dist normal estandar -> escala MAD a sigma
    z_robust = 0.6745 * (x - median) / mad
    return int((z_robust.abs() > threshold).sum())


def _boxcox_lambda(x: pd.Series) -> tuple[float | None, str]:
    """Estima lambda optimo de Box-Cox. Devuelve (lambda, recomendacion).

    Box-Cox requiere x > 0. Si hay no-positivos, se aplica shift x + |min| + 1.
    Lambda interpretacion (regla mnemotecnica):
        λ ≈ 1   -> NO transformar (ya simetrica)
        λ ≈ 0.5 -> sqrt
        λ ≈ 0   -> log
        λ ≈ -0.5 -> 1/sqrt
        λ ≈ -1  -> 1/x
    """
    from scipy.stats import boxcox

    x_clean = x.dropna()
    if len(x_clean) < 20 or x_clean.std() == 0:
        return None, "n insuficiente o varianza cero"
    try:
        x_pos = x_clean.copy()
        if (x_pos <= 0).any():
            x_pos = x_pos - x_pos.min() + 1.0
        _, lam = boxcox(x_pos.values)
        lam = float(lam)
        if abs(lam - 1.0) < 0.2:
            rec = "ninguna (ya simetrica)"
        elif abs(lam - 0.5) < 0.2:
            rec = "sqrt(x)"
        elif abs(lam) < 0.2:
            rec = "log(x)"
        elif abs(lam + 0.5) < 0.2:
            rec = "1/sqrt(x)"
        elif abs(lam + 1.0) < 0.2:
            rec = "1/x"
        else:
            rec = f"Box-Cox(λ={lam:.2f})"
        return lam, rec
    except Exception as e:
        return None, f"Box-Cox fallo: {e}"


def profile_variable(name: str, x: pd.Series) -> VariableProfile:
    """Construye un VariableProfile completo para una variable numerica."""
    x = pd.to_numeric(x, errors="coerce")
    x_clean = x.dropna()

    if len(x_clean) == 0:
        return VariableProfile(
            name=name,
            n=0,
            n_missing=int(x.isna().sum()),
            miss_ratio=1.0,
            mean=float("nan"),
            median=float("nan"),
            std=float("nan"),
            skew=float("nan"),
            kurtosis=float("nan"),
            p01=float("nan"),
            p99=float("nan"),
            min=float("nan"),
            max=float("nan"),
            n_outliers_iqr=0,
            n_outliers_zscore=0,
            n_outliers_mad=0,
            boxcox_lambda=None,
            boxcox_recommendation="todo NaN",
            normality_tests=[],
        )

    lam, rec = _boxcox_lambda(x_clean)

    return VariableProfile(
        name=name,
        n=len(x_clean),
        n_missing=int(x.isna().sum()),
        miss_ratio=float(x.isna().mean()),
        mean=float(x_clean.mean()),
        median=float(x_clean.median()),
        std=float(x_clean.std()),
        skew=float(x_clean.skew()),
        kurtosis=float(x_clean.kurtosis()),
        p01=float(x_clean.quantile(0.01)),
        p99=float(x_clean.quantile(0.99)),
        min=float(x_clean.min()),
        max=float(x_clean.max()),
        n_outliers_iqr=_outliers_iqr(x_clean),
        n_outliers_zscore=_outliers_zscore(x_clean),
        n_outliers_mad=_outliers_mad(x_clean),
        boxcox_lambda=lam,
        boxcox_recommendation=rec,
        normality_tests=[
            shapiro_wilk(x_clean),
            anderson_darling(x_clean),
            jarque_bera(x_clean),
        ],
    )


def profile_all_numeric(df: pd.DataFrame, cols: list[str]) -> list[VariableProfile]:
    """Profile de todas las columnas numericas indicadas."""
    return [profile_variable(c, df[c]) for c in cols if c in df.columns]
