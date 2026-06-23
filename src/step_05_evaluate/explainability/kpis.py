"""KPIs ejecutivos en lenguaje natural.

Las 3 preguntas que importan al lector no-tecnico:
  1. ¿Que tan preciso es el modelo?         -> kpi_precision
  2. ¿Cuanto explica?                       -> kpi_explanatory_power
  3. ¿Vale la pena vs no usar modelo?       -> kpi_vs_baseline

Cada uno devuelve un `PlainKPI` con titulo, headline corto, detail
extendido, version tecnica y semaforo (ALTO/MEDIO/BAJO).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import (
    KPI_BASELINE_HIGH_IMPROVEMENT_PCT,
    KPI_BASELINE_MEDIUM_IMPROVEMENT_PCT,
    KPI_PRECISION_HIGH_MAPE_PCT,
    KPI_PRECISION_MEDIUM_MAPE_PCT,
    KPI_R2_HIGH_PCT,
    KPI_R2_MEDIUM_PCT,
)


@dataclass(frozen=True)
class PlainKPI:
    """KPI ejecutivo: titulo de la pregunta + respuesta en una frase."""

    question: str  # "¿Qué tan preciso es?"
    headline: str  # "8 de 10 cosechas con error ≤ 7 kg/jornal"
    detail: str  # "Mediana del error: 4.2 kg/jornal..."
    technical: str  # "MAPE OOF = 17.5% (KG/JR)"
    score_label: str  # "ALTO" | "MEDIO" | "BAJO" — semaforo opcional


def kpi_precision(abs_errors: np.ndarray, mape_pct: float) -> PlainKPI:
    """Pregunta 1: ¿Que tan preciso es?

    Reporta el percentil 80 del error absoluto en KG/JR ("8 de cada 10").
    Esa cifra es mas accionable que la mediana para gerencia: dice "el
    peor caso en 80% de las veces".
    """
    if abs_errors.size == 0:
        return PlainKPI(
            question="¿Qué tan preciso es?",
            headline="Sin datos suficientes para evaluar precisión",
            detail="No se pudo calcular el error en unidades de negocio.",
            technical="—",
            score_label="—",
        )

    p50 = float(np.percentile(abs_errors, 50))
    p80 = float(np.percentile(abs_errors, 80))

    score = (
        "ALTO"
        if mape_pct <= KPI_PRECISION_HIGH_MAPE_PCT
        else ("MEDIO" if mape_pct <= KPI_PRECISION_MEDIUM_MAPE_PCT else "BAJO")
    )

    return PlainKPI(
        question="¿Qué tan preciso es?",
        headline=f"8 de cada 10 predicciones tienen error ≤ {p80:.1f} kg/jornal",
        detail=(
            f"La mitad de las predicciones se equivoca en ≤ {p50:.1f} kg/jornal. "
            f"En promedio, cada predicción se desvía un {mape_pct:.1f}% del valor real."
        ),
        technical=f"MAPE OOF = {mape_pct:.2f}% · p50={p50:.2f} kg · p80={p80:.2f} kg",
        score_label=score,
    )


def kpi_explanatory_power(r2: float | None) -> PlainKPI:
    """Pregunta 2: ¿Cuanto explica?

    Traduce R^2 a "captura el X% de la variabilidad observada". Una
    persona no tecnica entiende mejor "explicar variabilidad" que R^2.
    """
    if r2 is None or np.isnan(r2):
        return PlainKPI(
            question="¿Cuánto explica?",
            headline="Sin información de varianza explicada",
            detail="No se pudo calcular el coeficiente de determinación.",
            technical="—",
            score_label="—",
        )
    pct = float(r2) * 100.0
    score = "ALTO" if pct >= KPI_R2_HIGH_PCT else ("MEDIO" if pct >= KPI_R2_MEDIUM_PCT else "BAJO")
    return PlainKPI(
        question="¿Cuánto explica?",
        headline=f"Captura el {pct:.0f}% de la variabilidad observada",
        detail=(
            "Un modelo perfecto explicaría el 100%; uno que solo predice el "
            f"promedio explicaría 0%. Este modelo explica {pct:.0f}% del "
            "comportamiento real de la productividad."
        ),
        technical=f"R² OOF = {r2:.4f}",
        score_label=score,
    )


def kpi_vs_baseline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> PlainKPI:
    """Pregunta 3: ¿Vale la pena vs no usar modelo?

    Baseline ingenuo = predecir siempre la media de y_true. Calcula la
    mejora porcentual del MAE del modelo sobre el MAE del baseline.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return PlainKPI(
            question="¿Vale la pena vs no usar modelo?",
            headline="Sin datos suficientes para comparar",
            detail="—",
            technical="—",
            score_label="—",
        )

    mae_model = float(np.mean(np.abs(y_pred - y_true)))
    baseline_pred = float(np.mean(y_true))
    mae_baseline = float(np.mean(np.abs(y_true - baseline_pred)))

    if mae_baseline <= 0:
        improvement_pct = 0.0
    else:
        improvement_pct = (mae_baseline - mae_model) / mae_baseline * 100.0

    if improvement_pct >= KPI_BASELINE_HIGH_IMPROVEMENT_PCT:
        score = "ALTO"
    elif improvement_pct >= KPI_BASELINE_MEDIUM_IMPROVEMENT_PCT:
        score = "MEDIO"
    else:
        score = "BAJO"

    return PlainKPI(
        question="¿Vale la pena vs no usar modelo?",
        headline=(
            f"{improvement_pct:.0f}% más preciso que predecir el promedio"
            if improvement_pct > 0
            else "Sin ventaja clara sobre predecir el promedio"
        ),
        detail=(
            f"Sin modelo, alguien que estimara siempre el promedio "
            f"({baseline_pred:.1f} kg/jornal) se equivocaría en promedio "
            f"{mae_baseline:.1f} kg. Con este modelo, el error promedio baja a "
            f"{mae_model:.1f} kg."
        ),
        technical=(
            f"MAE modelo={mae_model:.2f} · MAE baseline (media)={mae_baseline:.2f} · "
            f"mejora={improvement_pct:+.1f}%"
        ),
        score_label=score,
    )
