"""
Router /history - administracion del historial usado por el feature engineering
=================================================================================
Las observaciones historicas (KG/HA + KG/JR_H reales) son la materia
prima de los lag features. Tipicamente se siembran una vez desde el
Excel de cosechas; opcionalmente se siguen alimentando con observaciones
nuevas para mantener fresca la mediana rolling.

Endpoints:
----------
POST   /history/{variety}/upload   - Sube Excel y reemplaza el historial
GET    /history/{variety}          - Lista paginada
DELETE /history/{variety}          - Borra todas las observaciones
"""

import logging

import pandas as pd
from fastapi import APIRouter, File, Query, UploadFile

from app import crud
from app.core import (
    parse_date_value,
    read_excel_dataframe,
    validate_excel_file,
    validate_upload_size,
)
from app.dependencies import DbSession, ValidatedVariety
from app.schemas import (
    DeletedCountResponse,
    HistoricalObservationCreate,
    HistoricalObservationListResponse,
    HistoricalObservationResponse,
    HistoryImportResponse,
)

router = APIRouter(prefix="/history", tags=["history"])
logger = logging.getLogger(__name__)


REQUIRED_HISTORY_COLUMNS: frozenset[str] = frozenset(
    {"FUNDO", "FORMATO", "FECHA", "KG/HA", "KG/JR_H"}
)


# ============================================================================
# Helpers
# ============================================================================


def _opt_float(row: pd.Series, col: str) -> float | None:
    """Lee una columna numérica opcional; None si falta o es NaN."""
    if col not in row.index:
        return None
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    return float(val)


def _parse_history_excel(
    contents: bytes,
) -> tuple[list[HistoricalObservationCreate], int]:
    """Devuelve (filas validas, filas descartadas).

    Columnas requeridas: FUNDO, FORMATO, FECHA, KG/HA, KG/JR_H.
    Columnas opcionales (features reales, espejan el Excel de pronósticos):
    DPC, %INDUS, P/BAYA, HA, DIA_COSECHA. Cuando vienen, habilitan la
    descomposición exacta de error en el UI.

    Filas con NaN en columnas requeridas o con valores invalidos se
    descartan en silencio: el operador puede preferir un seed parcial
    a fallar el import completo. El conteo se devuelve para auditar.
    """
    df = read_excel_dataframe(contents, REQUIRED_HISTORY_COLUMNS)

    rows: list[HistoricalObservationCreate] = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            kg_ha = float(row["KG/HA"])
            kg_jr_h = float(row["KG/JR_H"])
            if pd.isna(kg_ha) or pd.isna(kg_jr_h) or kg_ha <= 0 or kg_jr_h <= 0:
                skipped += 1
                continue
            dia_raw = _opt_float(row, "DIA_COSECHA")
            rows.append(
                HistoricalObservationCreate.model_validate(
                    {
                        "FUNDO": str(row["FUNDO"]).strip(),
                        "FORMATO": str(row["FORMATO"]).strip(),
                        "FECHA": parse_date_value(row["FECHA"]),
                        "KG/HA": kg_ha,
                        "KG/JR_H": kg_jr_h,
                        "DPC": _opt_float(row, "DPC"),
                        "%INDUS": _opt_float(row, "%INDUS"),
                        "P/BAYA": _opt_float(row, "P/BAYA"),
                        "HA": _opt_float(row, "HA"),
                        "DIA_COSECHA": int(dia_raw) if dia_raw is not None else None,
                    }
                )
            )
        except (ValueError, TypeError, KeyError):
            skipped += 1

    return rows, skipped


# ============================================================================
# Endpoints
# ============================================================================


@router.post(
    "/{variety}/upload",
    response_model=HistoryImportResponse,
    status_code=201,
)
async def upload_history(
    variety: ValidatedVariety,
    db: DbSession,
    file: UploadFile = File(..., description="Excel con FUNDO, FORMATO, FECHA, KG/HA, KG/JR_H"),
    replace: bool = Query(
        True,
        description="Si es True, borra el historial existente antes de insertar",
    ),
) -> HistoryImportResponse:
    """Sube observaciones historicas para una variedad.

    Comportamiento:
    - `replace=True` (default): borra el historial actual y reemplaza.
    - `replace=False`: agrega al historial existente.

    Raises:
        400: archivo invalido o sin columnas requeridas.
        404: variedad inexistente.
    """
    # Rechaza archivos sobredimensionados ANTES de leerlos a memoria
    # (mismo patrón que forecasts/upload). Ver excel_helpers.validate_upload_size.
    validate_upload_size(file.size, file.filename)

    contents = await file.read()
    validate_excel_file(contents, file.filename)
    rows, skipped = _parse_history_excel(contents)

    if replace:
        deleted = await crud.historical_observation.delete_by_variety(db, variety)
        logger.info("History replaced: variety=%s deleted=%d", variety, deleted)

    inserted = await crud.historical_observation.bulk_insert(db, variety, rows)
    logger.info(
        "History upload: variety=%s inserted=%d skipped=%d", variety, inserted, skipped
    )

    return HistoryImportResponse(
        variety=variety,
        inserted=inserted,
        skipped_invalid_rows=skipped,
        message=(
            f"{inserted} observaciones cargadas para {variety} "
            f"({'reemplazo total' if replace else 'append'}); "
            f"{skipped} filas descartadas por valores invalidos."
        ),
    )


@router.get("/{variety}", response_model=HistoricalObservationListResponse)
async def list_history(
    variety: ValidatedVariety,
    db: DbSession,
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> HistoricalObservationListResponse:
    """Lista paginada del historial cargado para una variedad."""
    rows, total = await crud.historical_observation.list_history(
        db, variety, limit=limit, offset=offset,
    )
    items = [HistoricalObservationResponse.model_validate(r) for r in rows]
    return HistoricalObservationListResponse(
        items=items, total=total, limit=limit, offset=offset,
    )


@router.delete("/{variety}", response_model=DeletedCountResponse)
async def delete_history(
    variety: ValidatedVariety,
    db: DbSession,
) -> DeletedCountResponse:
    """Borra todo el historial de una variedad. Operacion irreversible."""
    deleted = await crud.historical_observation.delete_by_variety(db, variety)
    logger.warning("History wiped: variety=%s deleted=%d", variety, deleted)
    return DeletedCountResponse(
        deleted=deleted,
        message=f"{deleted} observaciones historicas eliminadas para {variety}",
    )
