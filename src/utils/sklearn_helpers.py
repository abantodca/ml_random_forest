"""Helpers compartidos por la capa de training (tuning / oof).

Centraliza patrones que se repetian bit-for-bit en multiples archivos:

    `fit_with_optional_sample_weight` : el namespace `regressor__sample_weight`
        del Pipeline depende de la profundidad del wrapper (TTR + modelo).
        Tener UN solo punto de definicion permite refactorear el namespace
        si en el futuro se cambia el wrapper raw -> wrapped.

    `index_or_none` : `arr[idx] if arr is not None else None` aparecia 4
        veces para slicear sample_weight / strat_label por fold.

    `dump_json_artifact` : `json.dumps(..., indent=2, ensure_ascii=False) +
        write_text(encoding="utf-8")` aparecia 5+ veces para summaries y
        decision JSONs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def fit_with_optional_sample_weight(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> None:
    """Fit que pasa `sample_weight` al regressor del Pipeline si no es None.

    El namespace `regressor__sample_weight` asume que el ultimo step del
    Pipeline se llama 'regressor' (convencion en build_pipeline.py /
    tuning._build_pipeline). Si el estimador NO es un Pipeline (p.ej.
    OOFEnsembleRegressor), se pasa `sample_weight` directamente.
    """
    if sample_weight is not None:
        if isinstance(pipeline, Pipeline):
            pipeline.fit(X, y, regressor__sample_weight=sample_weight)
        else:
            pipeline.fit(X, y, sample_weight=sample_weight)
    else:
        pipeline.fit(X, y)


def index_or_none(arr: np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
    """`arr[idx]` si `arr` no es None; None en caso contrario.

    Util para slicear sample_weight / strat_label por fold sin un if/else
    expansivo en cada call site.
    """
    if arr is None:
        return None
    return arr[idx]


def dump_json_artifact(
    path: Path,
    data: dict,
    *,
    indent: int = 2,
    default=str,
) -> Path:
    """Serializa `data` a `path` con la convencion del proyecto.

    Convencion (consistente con todos los summaries del proyecto):
        - indent=2, ensure_ascii=False (caracteres latinos legibles).
        - default=str para tipos no-serializables (Path, datetime, np.float).
        - encoding utf-8.

    Devuelve `path` para encadenar (`log_artifact(dump_json_artifact(...))`).
    """
    Path(path).write_text(
        json.dumps(data, indent=indent, ensure_ascii=False, default=default),
        encoding="utf-8",
    )
    return Path(path)
