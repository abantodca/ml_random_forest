"""Analisis de variables categoricas (FORMATO, FUNDO, ...) para el EDA.

Contraparte de `multivariate.py` (que cubre solo numericas) y
`distributions.py` (univariado numerico). Produce stats que el feature-
engineering necesita SIN ejecutar el pipeline:

  - Cardinalidad (nunique) y missing rate.
  - Top-N categorias por frecuencia + % acumulado (regla 80/20).
  - Mean target por categoria (insight para target-encoding y deteccion
    de categorias 'discriminantes').
  - Chi-square: independencia categorica vs target binarizada por mediana
    (para regresion, binarizar permite usar el mismo test que clasif).
  - Cramer's V entre pares de categoricas: asociacion. V>0.3 sugiere
    redundancia / variables co-determinadas (ej. FUNDO  determina FORMATO).

Las funciones devuelven dataclasses serializables a JSON via dataclasses.asdict;
el caller decide el formato (HTML para dashboard, JSON sidecar para que el
pipeline / LLM consuma stats sin parsear HTML).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    CARDINALITY_HIGH,
    CARDINALITY_WARN,
    CRAMERS_V_STRONG,
    CRAMERS_V_WEAK,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TopCategory:
    value: str
    count: int
    pct: float  # frecuencia relativa (0..1)
    cum_pct: float  # frecuencia acumulada (orden descendente)
    target_mean: float | None = None
    target_std: float | None = None
    target_count: int = 0


@dataclass
class CategoricalProfile:
    name: str
    n: int
    n_missing: int
    miss_ratio: float
    cardinality: int
    n_singletons: int  # categorias con count==1 (ruido potencial)
    top_categories: list[TopCategory] = field(default_factory=list)
    coverage_top10_pct: float = 0.0  # cobertura cumulativa del top-10
    chi2_statistic: float | None = None
    chi2_p_value: float | None = None
    chi2_dof: int | None = None
    cramers_v_target: float | None = None  # asociacion con target binarizado
    target_encoding_recommendation: str = ""


@dataclass
class CategoricalAssociation:
    """Asociacion (Cramer's V) entre dos categoricas. Simetrica."""

    feature_a: str
    feature_b: str
    cramers_v: float
    chi2_p_value: float
    severity: str  # "ok"|"watch"|"high"


@dataclass
class CategoricalReport:
    profiles: list[CategoricalProfile]
    associations: list[CategoricalAssociation]


# ---------------------------------------------------------------------------
# Computacion
# ---------------------------------------------------------------------------
def _cramers_v(table: np.ndarray) -> tuple[float, float, int]:
    """Cramer's V con correccion de sesgo (Bergsma & Wicher 2013).

    Devuelve (V, p_value, dof). table: contingency matrix (no normalizada).
    Para tablas con celdas esperadas <5 el chi2 es poco confiable; lo
    devolvemos igual y dejamos que el caller filtre.
    """
    from scipy.stats import chi2_contingency

    if table.size == 0 or table.sum() == 0:
        return 0.0, 1.0, 0
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.sum()
    r, k = table.shape
    if min(r, k) <= 1 or n == 0:
        return 0.0, p, dof
    # correccion de sesgo
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return 0.0, p, dof
    return float(np.sqrt(phi2_corr / denom)), float(p), int(dof)


def profile_categorical(
    df: pd.DataFrame,
    column: str,
    target: pd.Series,
    *,
    top_n: int = 10,
    min_count_for_target: int = 5,
) -> CategoricalProfile:
    """Profile completo de UNA columna categorica vs el target."""
    s = df[column]
    n = len(s)
    n_miss = int(s.isna().sum())

    # Top-N + target stats por categoria
    counts = s.value_counts(dropna=False)
    cardinality = int(counts.shape[0])
    n_singletons = int((counts == 1).sum())

    cum = 0
    total = int(counts.sum())
    top_cats: list[TopCategory] = []
    for val, c in counts.head(top_n).items():
        pct = c / total if total > 0 else 0.0
        cum += c
        cum_pct = cum / total if total > 0 else 0.0
        # target stats sobre las filas donde s == val
        mask = s == val if not pd.isna(val) else s.isna()
        sub_target = target[mask].dropna()
        tmean = float(sub_target.mean()) if len(sub_target) >= min_count_for_target else None
        tstd = float(sub_target.std()) if len(sub_target) >= min_count_for_target else None
        top_cats.append(
            TopCategory(
                value=str(val),
                count=int(c),
                pct=float(pct),
                cum_pct=float(cum_pct),
                target_mean=tmean,
                target_std=tstd,
                target_count=int(len(sub_target)),
            )
        )

    coverage_top10 = float(counts.head(10).sum() / total) if total > 0 else 0.0

    # Chi-square contra target binarizada por la mediana
    chi2_stat = chi2_p = None
    chi2_dof_val = None
    cramers_v_val = None
    target_clean = target.dropna()
    if len(target_clean) >= 30 and cardinality >= 2 and cardinality <= CARDINALITY_HIGH:
        try:
            median = float(target_clean.median())
            target_bin = (target >= median).astype(int)
            ct = pd.crosstab(s.fillna("__missing__"), target_bin)
            if ct.shape[0] >= 2 and ct.shape[1] == 2:
                v, p, dof = _cramers_v(ct.values)
                cramers_v_val = v
                chi2_p = p
                chi2_dof_val = dof
                # statistic util para el HTML
                from scipy.stats import chi2_contingency

                chi2_stat = float(chi2_contingency(ct.values)[0])
        except Exception as exc:
            # Stats inferenciales opcionales del EDA: si scipy falla (tabla
            # degenerada, frecuencias esperadas 0) quedan en None y el HTML
            # simplemente no los muestra.
            logger.debug("Chi2/Cramers V omitidos para la categorica: %s", exc)

    # Recomendacion target-encoding
    rec = _target_encoding_rec(cardinality, n - n_miss, cramers_v_val)

    return CategoricalProfile(
        name=column,
        n=n,
        n_missing=n_miss,
        miss_ratio=float(n_miss / n) if n > 0 else 0.0,
        cardinality=cardinality,
        n_singletons=n_singletons,
        top_categories=top_cats,
        coverage_top10_pct=coverage_top10,
        chi2_statistic=chi2_stat,
        chi2_p_value=chi2_p,
        chi2_dof=chi2_dof_val,
        cramers_v_target=cramers_v_val,
        target_encoding_recommendation=rec,
    )


def _target_encoding_rec(cardinality: int, n_valid: int, v: float | None) -> str:
    """Heuristica corta para sugerir tratamiento de la categorica."""
    if cardinality <= 1:
        return "constante; eliminar columna"
    if cardinality == 2:
        return "binaria; OneHot directo"
    if cardinality > CARDINALITY_WARN and n_valid / max(cardinality, 1) < 30:
        return f"alta cardinalidad ({cardinality}) con baja densidad; target-encoding con suavizado o agrupar 'Otros'"
    if v is not None and v < CRAMERS_V_WEAK:
        return f"asociacion debil con target (V={v:.2f}); candidata a drop o agrupar"
    if v is not None and v >= CRAMERS_V_STRONG:
        return f"asociacion fuerte (V={v:.2f}); target-encoding o Embedding util"
    return "OneHot estandar (cardinalidad media, asociacion moderada)"


def compute_associations(
    df: pd.DataFrame,
    columns: list[str],
    *,
    threshold_high: float = 0.5,
    threshold_watch: float = 0.3,
) -> list[CategoricalAssociation]:
    """Cramer's V entre todos los pares de categoricas en `columns`."""
    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return []
    out: list[CategoricalAssociation] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            sa = df[a].fillna("__missing__")
            sb = df[b].fillna("__missing__")
            try:
                ct = pd.crosstab(sa, sb)
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                v, p, _ = _cramers_v(ct.values)
            except Exception:
                continue
            sev = "high" if v >= threshold_high else "watch" if v >= threshold_watch else "ok"
            out.append(
                CategoricalAssociation(
                    feature_a=a,
                    feature_b=b,
                    cramers_v=float(v),
                    chi2_p_value=float(p),
                    severity=sev,
                )
            )
    return sorted(out, key=lambda x: x.cramers_v, reverse=True)


def build_categorical_report(
    df: pd.DataFrame,
    columns: list[str],
    target: pd.Series,
    *,
    top_n: int = 10,
) -> CategoricalReport:
    """Profile completo + asociaciones para todas las categoricas indicadas."""
    profiles = [profile_categorical(df, c, target, top_n=top_n) for c in columns if c in df.columns]
    associations = compute_associations(df, columns)
    return CategoricalReport(profiles=profiles, associations=associations)
