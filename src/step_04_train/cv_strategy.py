"""Estrategia de Cross-Validation para el Nested CV.

Construye los splitters outer/inner y la etiqueta de estratificacion
adaptativa por variedad. Extraido de `tuning.py` (2026-06-26) para acotar el
tamano del orquestador: la decision de COMO partir los datos es una unidad
cohesiva y testeable por separado del bucle de tuning.

`step_04_train/` codifica el orden del pipeline; este modulo NO se serializa
(solo produce splitters efimeros), asi que agregarlo no afecta los .joblib.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def build_strat_label(
    X: pd.DataFrame,
    min_count: int,
) -> tuple[pd.Series | None, str]:
    """Etiqueta de estratificacion ADAPTATIVA por variedad.

    Cascada de estrategias (mas especifica primero):
        1. `FUNDO_FORMATO` compuesto, con clases n<min_count -> 'RARE'.
        2. `FUNDO` solo (idem).
        3. `FORMATO` solo (idem).
        4. None -> caller cae a `KFold` sin estratificar.

    En cada nivel se valida que tras colapsar:
        - hay >=2 clases distintas (sin variabilidad no se puede stratify),
        - cada clase final tiene n>=min_count (requisito de StratifiedKFold).

    Asi una variedad con 4x4 categoricas y desbalance moderado entra por la
    estrategia compuesta; una variedad con 1 solo FUNDO entra por FORMATO; y
    una con 1 FUNDO y 1 FORMATO degenera a KFold sin tropezar.

    Devuelve (label, strategy_name) para que el caller logue la decision.
    """
    candidates: list[tuple[str, pd.Series]] = []
    if "FUNDO" in X.columns and "FORMATO" in X.columns:
        candidates.append(
            ("FUNDO_FORMATO", X["FUNDO"].astype(str) + "_" + X["FORMATO"].astype(str))
        )
    if "FUNDO" in X.columns:
        candidates.append(("FUNDO", X["FUNDO"].astype(str)))
    if "FORMATO" in X.columns:
        candidates.append(("FORMATO", X["FORMATO"].astype(str)))

    for name, label in candidates:
        counts = label.value_counts()
        rare = counts[counts < min_count].index
        if len(rare) > 0:
            label = label.where(~label.isin(rare), other="RARE")
        final_counts = label.value_counts()
        if len(final_counts) >= 2 and (final_counts >= min_count).all():
            return label, name
    return None, "none"


def build_cv_splitters(
    X: pd.DataFrame,
    outer_folds: int,
    inner_folds: int,
    random_state: int,
):
    """Construye outer/inner CV. Outer puede ser stratified o temporal.

    Devuelve `(outer_cv, inner_cv, strat_label, strat_strategy)`.

    Outer strategy controlada por `CV_OUTER_STRATEGY` (env / config):
        - "stratified" (default): StratifiedKFold por FUNDO_FORMATO con
          fallback adaptativo a FUNDO -> FORMATO -> KFold.
        - "temporal_year": TemporalYearSplit (expanding window por ANIO).
          Resuelve drift severo: el modelo NO ve futuro en train. Necesita
          columna ANIO o DATE_COLUMN en X.

    Inner siempre stratified (dentro del outer fold el riesgo temporal ya
    se mitigo; el inner Optuna se beneficia del balance por estrato).
    """
    import math

    from src.config import CV_OUTER_STRATEGY, TEMPORAL_CV_MIN_TRAIN_YEARS

    strat_min_count = max(
        outer_folds,
        math.ceil(inner_folds * outer_folds / max(outer_folds - 1, 1)),
    )
    strat_label, strat_strategy = build_strat_label(X, min_count=strat_min_count)

    # Outer
    if CV_OUTER_STRATEGY == "temporal_year":
        from src.step_04_train.temporal_cv import TemporalYearSplit

        outer_cv = TemporalYearSplit(
            year_col="ANIO",
            n_splits=outer_folds,
            min_train_years=TEMPORAL_CV_MIN_TRAIN_YEARS,
        )
    else:
        outer_splitter_cls = StratifiedKFold if strat_label is not None else KFold
        outer_cv = outer_splitter_cls(
            n_splits=outer_folds,
            shuffle=True,
            random_state=random_state,
        )

    # Inner: siempre stratified (cuando hay strat_label) — el outer fold
    # contiene multiples anios mezclados, el balance por FUNDO_FORMATO
    # estabiliza la inner CV de Optuna.
    inner_splitter_cls = StratifiedKFold if strat_label is not None else KFold
    inner_cv = inner_splitter_cls(
        n_splits=inner_folds,
        shuffle=True,
        random_state=random_state,
    )
    return outer_cv, inner_cv, strat_label, strat_strategy
