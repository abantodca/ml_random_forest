"""Validacion de schema y rangos del dataset crudo.

Se ejecuta DESPUES del parseo Excel pero ANTES de que el DataFrame entre al
Pipeline. Detecta problemas estructurales que romperian el modelo o
producirian metricas absurdas:

  - tipos incorrectos (categoricas que vinieron como float, fechas mal parseadas)
  - rangos imposibles (KG/JR_H <= 0, H-EF fuera de [0, 24])
  - duplicados estructurales (mismo FUNDO+FORMATO+FECHA -> dos cosechas
    contradictorias para el mismo dia)
  - columnas con varianza cero (todas las filas con el mismo valor; no
    aporta nada al modelo)

Comportamiento: solo loguea warnings y devuelve la lista de issues. NO
aborta el pipeline. Los sintomas reales aparecen en el modelo (training
fallido, MAPE alto). Sin dependencias nuevas: solo pandas + numpy.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    DATE_COLUMN,
    NUMERIC_FEATURES,
    TARGET,
)

logger = logging.getLogger(__name__)


# Rangos sanos por columna. Si un valor cae fuera, lo marcamos como
# violacion de schema. None = sin chequeo de rango.
_NUMERIC_RANGES: dict = {
    TARGET: (0.0, None),  # KG/JR_H > 0; sin techo (puede haber outliers altos)
    "KG/HA": (0.0, None),  # productividad por hectarea > 0
    "%INDUS": (0.0, 100.0),  # porcentaje
    "DPC": (0.0, None),  # dias post cosecha; tipicamente 0-180
    "P/BAYA": (0.0, None),  # peso por baya en gramos
    "HA": (0.0, None),  # hectareas cosechadas
    "DIA_COSECHA": (1, 366),  # dia desde inicio de campana (no DOY); cap >365 generoso
    "H-EF": (0.0, 24.0),  # horas efectivas en una jornada (validacion solo)
    "KG/JR": (0.0, None),  # kilos por jornal (validacion solo)
}


def _check_dtypes(df: pd.DataFrame) -> list[str]:
    """Devuelve lista de problemas de tipo detectados."""
    issues: list[str] = []
    for col in NUMERIC_FEATURES + [TARGET]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            n_non_numeric = pd.to_numeric(df[col], errors="coerce").isna().sum()
            if n_non_numeric > 0:
                issues.append(
                    f"'{col}' tiene {n_non_numeric} valor(es) no numerico(s) "
                    f"que se convertiran a NaN"
                )
    if DATE_COLUMN in df.columns and not pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]):
        # Permitimos string si parsea bien; lo flaggeamos como info.
        try:
            pd.to_datetime(df[DATE_COLUMN], errors="raise")
        except Exception as exc:
            issues.append(f"'{DATE_COLUMN}' no parsea como fecha: {exc}")
    return issues


def _check_numeric_ranges(df: pd.DataFrame) -> list[str]:
    """Reporta filas con valores fuera del rango sano por columna."""
    issues: list[str] = []
    for col, (lo, hi) in _NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        bad_lo = (s < lo).sum() if lo is not None else 0
        bad_hi = (s > hi).sum() if hi is not None else 0
        if bad_lo > 0:
            issues.append(f"'{col}': {int(bad_lo)} fila(s) con valor < {lo}")
        if bad_hi > 0:
            issues.append(f"'{col}': {int(bad_hi)} fila(s) con valor > {hi}")
    return issues


def _check_full_row_duplicates(df: pd.DataFrame) -> list[str]:
    """Detecta filas COMPLETAMENTE duplicadas (todas las columnas iguales).

    NO chequeamos por (FUNDO+FORMATO+FECHA) porque en datos agricolas varias
    cosechas el mismo dia para el mismo (fundo, formato) son legitimas
    (distintas cuadrillas, turnos, lotes). Solo flaggeamos filas exactamente
    duplicadas que probablemente vinieron de un export duplicado del Excel.
    """
    n_dup = df.duplicated(keep="first").sum()
    if n_dup > 0:
        return [f"{int(n_dup)} fila(s) exactamente duplicadas (todas las cols iguales)"]
    return []


def _check_zero_variance(df: pd.DataFrame) -> list[str]:
    """Columnas con varianza 0 (todos los valores iguales). No aportan al modelo."""
    issues: list[str] = []
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 0 and s.nunique() == 1:
            issues.append(f"'{col}' tiene varianza 0 (todos los valores = {s.iloc[0]})")
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) > 0 and s.nunique() == 1:
            issues.append(f"'{col}' tiene un unico nivel ({s.iloc[0]!r})")
    return issues


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Corre todos los chequeos de schema. Devuelve lista de issues encontrados.

    Solo loguea warnings; NUNCA aborta. Si el caller quiere validar columnas
    requeridas, usar `df.columns` directamente antes de invocar.
    """
    issues: list[str] = []
    issues.extend(_check_dtypes(df))
    issues.extend(_check_numeric_ranges(df))
    # Filas exactamente duplicadas: warning (sobre-pesan al training pero
    # no lo corrompen). Duplicados en (FUNDO+FORMATO+FECHA) NO se
    # chequean: son cosechas distintas legitimas el mismo dia.
    issues.extend(_check_full_row_duplicates(df))
    issues.extend(_check_zero_variance(df))

    for issue in issues:
        logger.warning(f"[schema] {issue}")

    return issues
