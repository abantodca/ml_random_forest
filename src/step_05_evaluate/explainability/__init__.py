"""Traduccion de metricas tecnicas a lenguaje ejecutivo (submodulo).

Centraliza toda la logica de "como contar el resultado a un no-tecnico":
veredicto del modelo, KPIs en lenguaje natural, comparacion contra
baseline ingenuo, deteccion de subgrupos problematicos / sesgo direccional,
glosario y contexto del entrenamiento. Lo consumen tanto
`winner_dashboard.py` (HTML) como `business_export.py` (Excel) para que
ambos digan lo mismo.

`build_winner_kit(...)` es la unica fabrica de inputs ejecutivos: arma de
una sola pasada (real, pred, abs_err, X_aligned, oof_mape/r2, verdict,
context, kpis, actions, fundo_bias) desde un `ModelResult` campeon.

Convencion: ningun string aqui adentro deberia contener jerga (MAPE, R2,
gap, OOF) sin traducirla. Si necesitas algo tecnico, llamalo desde su
funcion auxiliar y deja la version ejecutiva como output principal.

Submodulos (cada uno cohesivo en su dominio):
  - verdict.py     : Verdict + compute_verdict (4 niveles segun thresholds).
  - kpis.py        : PlainKPI + las 3 preguntas que importan.
  - bias.py        : GroupBias + residual_bias_by_group (sesgo direccional).
  - actions.py     : Action + recommended_actions (subgroups + globals).
  - context.py     : TrainingContext + build_context (descriptivo dataset).
  - glossary.py    : GLOSSARY + glossary_terms (jerga -> lenguaje claro).
  - kit.py         : WinnerKit + build_winner_kit (orquestador unificado).

Este `__init__.py` re-exporta el API publico para que los consumidores
sigan usando `from src.step_05_evaluate.explainability import X` sin
saber del split interno.
"""
from src.step_05_evaluate.explainability.actions import (
    Action,
    recommended_actions,
)
from src.step_05_evaluate.explainability.bias import (
    GroupBias,
    residual_bias_by_group,
)
from src.step_05_evaluate.explainability.context import (
    TrainingContext,
    build_context,
)
from src.step_05_evaluate.explainability.glossary import (
    GLOSSARY,
    glossary_terms,
)
from src.step_05_evaluate.explainability.kit import (
    WinnerKit,
    build_winner_kit,
)
from src.step_05_evaluate.explainability.kpis import (
    PlainKPI,
    kpi_explanatory_power,
    kpi_precision,
    kpi_vs_baseline,
)
from src.step_05_evaluate.explainability.verdict import (
    Verdict,
    compute_verdict,
)

__all__ = [
    # verdict
    "Verdict",
    "compute_verdict",
    # kpis
    "PlainKPI",
    "kpi_precision",
    "kpi_explanatory_power",
    "kpi_vs_baseline",
    # bias
    "GroupBias",
    "residual_bias_by_group",
    # actions
    "Action",
    "recommended_actions",
    # context
    "TrainingContext",
    "build_context",
    # glossary
    "GLOSSARY",
    "glossary_terms",
    # kit
    "WinnerKit",
    "build_winner_kit",
]
