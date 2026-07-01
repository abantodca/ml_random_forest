"""Contexto del entrenamiento: datos descriptivos del dataset para mostrar al lector."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.config import DATE_COLUMN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingContext:
    """Datos descriptivos del dataset para mostrar al lector."""

    n_rows: int
    date_min: str | None
    date_max: str | None
    n_fundos: int
    fundos_top: list[str]  # primeros 5 alfabeticamente
    n_formatos: int
    formatos_top: list[str]  # primeros 5 alfabeticamente


def build_context(
    X_raw: pd.DataFrame | None,
    date_col: str = DATE_COLUMN,
) -> TrainingContext:
    """Extrae el contexto presentable desde el dataset original."""
    n_rows = int(len(X_raw)) if X_raw is not None else 0
    date_min = date_max = None
    n_fundos = n_formatos = 0
    fundos_top: list[str] = []
    formatos_top: list[str] = []

    if X_raw is not None:
        if date_col in X_raw.columns:
            try:
                d = pd.to_datetime(X_raw[date_col], errors="coerce").dropna()
                if not d.empty:
                    date_min = d.min().strftime("%Y-%m")
                    date_max = d.max().strftime("%Y-%m")
            except Exception as exc:
                # Contexto descriptivo best-effort: sin rango de fechas el
                # reporte sigue siendo valido (date_min/max quedan None).
                logger.debug("Rango de fechas del contexto omitido: %s", exc)
        if "FUNDO" in X_raw.columns:
            uniq = sorted(X_raw["FUNDO"].dropna().astype(str).unique())
            n_fundos = len(uniq)
            fundos_top = uniq[:5]
        if "FORMATO" in X_raw.columns:
            uniq = sorted(X_raw["FORMATO"].dropna().astype(str).unique())
            n_formatos = len(uniq)
            formatos_top = uniq[:5]

    return TrainingContext(
        n_rows=n_rows,
        date_min=date_min,
        date_max=date_max,
        n_fundos=n_fundos,
        fundos_top=fundos_top,
        n_formatos=n_formatos,
        formatos_top=formatos_top,
    )
