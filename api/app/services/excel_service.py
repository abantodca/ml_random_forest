"""
Servicio de procesamiento de archivos Excel
============================================
Parsea y valida archivos Excel para convertirlos en `ForecastCreate`.

Las columnas requeridas espejan los inputs RAW del modelo MLflow. Las
opcionales (`%INDUS`, `P/BAYA`) son los `required: false` de la signature
del pipeline; pueden venir vacias y la pipeline las imputa internamente.
`HORAS_EFECTIVAS` es metadato del request (NO entra al modelo, solo se
usa para calcular KGJN_PRED post-prediccion).
"""

import logging

import pandas as pd

from app.core import (
    FORMATO_DEFAULT,
    parse_date_value,
    read_excel_dataframe,
    validate_excel_file,
)
from app.schemas import ForecastCreate

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"FECHA", "KG/HA", "DPC", "HA", "DIA_COSECHA", "FORMATO", "FUNDO"}
)


def parse_excel_to_forecasts(contents: bytes, filename: str) -> list[ForecastCreate]:
    """Parsea Excel y devuelve forecasts validados.

    Raises:
        ValueError: archivo invalido, excede tamano o faltan columnas.
    """
    validate_excel_file(contents, filename)

    df = read_excel_dataframe(contents, REQUIRED_COLUMNS)

    forecasts = [_row_to_forecast(row) for _, row in df.iterrows()]
    logger.debug("Excel parsed: file=%s rows=%d", filename, len(forecasts))
    return forecasts


# ============================================================================
# Helpers privados
# ============================================================================


def _safe_float(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    return float(val)


def _safe_str(row: pd.Series, col: str) -> str | None:
    if col not in row.index:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    return str(val).strip()


def _row_to_forecast(row: pd.Series) -> ForecastCreate:
    """Convierte una fila del DataFrame a un ForecastCreate.

    `populate_by_name=True` en el schema permite construir con los nombres
    snake_case (kg_ha, indus_pct, ...) en vez de los alias del Excel.
    """
    formato = _safe_str(row, "FORMATO") or FORMATO_DEFAULT

    return ForecastCreate.model_validate(
        {
            "FECHA": parse_date_value(row["FECHA"]),
            "EXTERNAL_ID": _safe_str(row, "EXTERNAL_ID"),
            "KG/HA": float(row["KG/HA"]),
            "%INDUS": _safe_float(row, "%INDUS"),
            "DPC": float(row["DPC"]),
            "P/BAYA": _safe_float(row, "P/BAYA"),
            "HA": float(row["HA"]),
            "DIA_COSECHA": int(row["DIA_COSECHA"]),
            "FORMATO": formato,
            "FUNDO": _safe_str(row, "FUNDO") or "",
            "HORAS_EFECTIVAS": _safe_float(row, "HORAS_EFECTIVAS"),
        }
    )
