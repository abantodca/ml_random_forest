"""
Router de Forecasts (Pronósticos)
==================================
Endpoints para crear, listar, actualizar y eliminar pronósticos.

Endpoints:
----------
POST   /forecasts/{variety}          - Crear pronóstico individual
POST   /forecasts/{variety}/predict  - Predecir sin persistir (dry-run)
POST   /forecasts/{variety}/batch    - Crear múltiples pronósticos
POST   /forecasts/{variety}/upload   - Cargar desde Excel
GET    /forecasts                    - Listar con filtros
GET    /forecasts/{id}               - Obtener por ID
PATCH  /forecasts/{id}               - Actualizar parcialmente
DELETE /forecasts/{id}               - Eliminar por ID
DELETE /forecasts/fecha/{fecha}      - Eliminar por fecha
"""

import logging
from datetime import date, datetime

from fastapi import APIRouter, File, Query, UploadFile

from app import crud
from app.core import validate_upload_size
from app.dependencies import (
    DbSession,
    ForecastSvc,
    ValidatedVariety,
    validate_optional_variety,
)
from app.schemas import (
    DeletedCountResponse,
    ForecastBatchCreate,
    ForecastCreate,
    ForecastListResponse,
    ForecastResponse,
    ForecastUpdate,
    PredictionResponse,
)
from app.services import parse_excel_to_forecasts

router = APIRouter(prefix="/forecasts", tags=["forecasts"])

logger = logging.getLogger(__name__)


# ============================================================================
# CREATE - Crear pronósticos
# ============================================================================
#
# La orquestación (build features → predecir → drift → persistir → ensamblar)
# vive en `ForecastService` (app/services/forecast_service.py). Estos handlers
# quedan finos: solo traducen HTTP ⇆ servicio.


@router.post("/{variety}", response_model=ForecastResponse, status_code=201)
async def create_forecast(
    variety: ValidatedVariety,
    forecast_data: ForecastCreate,
    db: DbSession,
    forecasts: ForecastSvc,
) -> ForecastResponse:
    """Crea un pronostico individual para una variedad.

    Raises:
        404: variedad inexistente.
        503: modelo no disponible en MLflow.
        500: la prediccion fallo.
    """
    return await forecasts.create_one(db, variety, forecast_data)


@router.post("/{variety}/predict", response_model=PredictionResponse)
async def predict_forecast(
    variety: ValidatedVariety,
    forecast_data: ForecastCreate,
    forecasts: ForecastSvc,
) -> PredictionResponse:
    """Predice KGHORA para una variedad SIN persistir (dry-run).

    Útil para predicción exploratoria (no ensucia la tabla forecasts) y para
    la descomposición de error: re-predecir sobre los inputs reales y comparar
    contra el pronóstico original y el valor realizado.

    Raises:
        404: variedad inexistente.
        503: modelo no disponible en MLflow.
        500: la prediccion fallo.
    """
    return await forecasts.predict_only(variety, forecast_data)


@router.post("/{variety}/batch", response_model=ForecastListResponse, status_code=201)
async def create_forecasts_batch(
    variety: ValidatedVariety,
    batch_data: ForecastBatchCreate,
    db: DbSession,
    forecasts: ForecastSvc,
) -> ForecastListResponse:
    """Crea multiples pronosticos en una sola transaccion.

    Maximo 1000 registros por llamada.
    """
    result = await forecasts.create_batch(db, variety, batch_data.forecasts)
    logger.info("Batch created: variety=%s count=%d", variety, result.total)
    return result


@router.post("/{variety}/upload", response_model=ForecastListResponse, status_code=201)
async def upload_excel_forecasts(
    variety: ValidatedVariety,
    db: DbSession,
    forecasts: ForecastSvc,
    file: UploadFile = File(..., description="Archivo Excel (.xlsx o .xls)"),
) -> ForecastListResponse:
    """
    Carga pronosticos desde un archivo Excel.

    Columnas requeridas: FECHA, KG/HA, DPC, HA, DIA_COSECHA, FORMATO, FUNDO
    Columnas opcionales: EXTERNAL_ID, %INDUS, P/BAYA, HORAS_EFECTIVAS

    Raises:
        400: archivo invalido o faltan columnas requeridas
        404: variedad inexistente
        503: modelo no disponible
    """
    logger.info("Upload started: variety=%s file=%s", variety, file.filename)

    # Rechaza archivos sobredimensionados ANTES de leerlos a memoria.
    # `UploadFile.size` es el Content-Length declarado por el cliente (puede
    # ser None si no viene en el part multipart). `validate_upload_size` falla
    # barato cuando el tamaño está disponible; si es None, el chequeo se
    # posterga a `parse_excel_to_forecasts` → `validate_excel_file`, que opera
    # sobre los bytes ya leídos (comportamiento anterior intacto).
    validate_upload_size(file.size, file.filename)

    contents = await file.read()
    forecasts_data = parse_excel_to_forecasts(contents, file.filename)

    result = await forecasts.create_batch(db, variety, forecasts_data)
    logger.info("Upload completed: variety=%s count=%d", variety, result.total)
    return result


# ============================================================================
# READ - Listar y obtener pronósticos
# ============================================================================


@router.get("", response_model=ForecastListResponse)
async def list_forecasts(
    db: DbSession,
    variety: str | None = Query(None, description="Filtrar por variedad"),
    fecha: date | None = Query(None, description="Filtrar por fecha (YYYY-MM-DD)"),
    external_id: str | None = Query(None, description="Filtrar por ID externo"),
    date_from: datetime | None = Query(None, description="Desde fecha de creación"),
    date_to: datetime | None = Query(None, description="Hasta fecha de creación"),
    limit: int = Query(500, ge=1, le=5000, description="Límite de resultados"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
) -> ForecastListResponse:
    """
    Lista pronósticos con filtros opcionales y paginación.

    Soporta filtrado por variedad, fecha, external_id y rangos de creación.
    Resultados ordenados por fecha de creación descendente.
    """
    variety = validate_optional_variety(variety)

    forecasts, total = await crud.forecast.get_forecasts(
        db=db,
        variety=variety,
        fecha=fecha,
        external_id=external_id,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )

    items = [ForecastResponse.model_validate(f) for f in forecasts]
    return ForecastListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/{forecast_id}", response_model=ForecastResponse)
async def get_forecast(
    forecast_id: int,
    db: DbSession,
) -> ForecastResponse:
    """
    Obtiene un pronóstico por su ID.

    Raises:
        404: Si no se encuentra el pronóstico
    """
    forecast = await crud.forecast.get_forecast_by_id(db, forecast_id)
    return ForecastResponse.model_validate(forecast)


# ============================================================================
# UPDATE - Actualizar pronósticos
# ============================================================================


@router.patch("/{forecast_id}", response_model=ForecastResponse)
async def update_forecast(
    forecast_id: int,
    forecast_update: ForecastUpdate,
    db: DbSession,
) -> ForecastResponse:
    """
    Actualiza parcialmente un pronóstico existente.

    Solo actualiza los campos enviados en el body.
    Si se actualiza HORAS_EFECTIVAS, recalcula KGJN_PRED automáticamente.

    Raises:
        404: Si no se encuentra el pronóstico
    """
    forecast = await crud.forecast.update_forecast(db, forecast_id, forecast_update)
    return ForecastResponse.model_validate(forecast)


# ============================================================================
# DELETE - Eliminar pronósticos
# ============================================================================


@router.delete("/fecha/{fecha}", response_model=DeletedCountResponse)
async def delete_forecasts_by_fecha(
    fecha: date,
    db: DbSession,
) -> DeletedCountResponse:
    """
    Elimina todos los pronósticos de una fecha específica.

    Returns:
        Número de pronósticos eliminados
    """
    count = await crud.forecast.delete_forecasts_by_fecha(db, fecha)
    logger.info("Deleted by fecha=%s count=%d", fecha, count)
    return DeletedCountResponse(deleted=count, message=f"{count} pronósticos eliminados de {fecha}")


@router.delete("/{forecast_id}", response_model=DeletedCountResponse)
async def delete_forecast(
    forecast_id: int,
    db: DbSession,
) -> DeletedCountResponse:
    """
    Elimina un pronóstico por su ID.

    Raises:
        404: Si no se encuentra el pronóstico
    """
    await crud.forecast.delete_forecast(db, forecast_id)
    logger.info("Deleted forecast id=%d", forecast_id)
    return DeletedCountResponse(deleted=1, message=f"Pronóstico {forecast_id} eliminado")
