"""
Operaciones CRUD para Forecasts
================================
Persiste y consulta predicciones. Las features RAW del modelo se
guardan junto con la prediccion para auditar despues que entrada
produjo que salida.
"""

import logging
from datetime import date, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import ForecastNotFoundError
from app.models import Forecast
from app.schemas import ForecastCreate, ForecastUpdate

logger = logging.getLogger(__name__)


def calc_kgjn(kghora_pred: float, horas_efectivas: float | None) -> float | None:
    """KGJN_PRED = kghora_pred * horas_efectivas. None si no hay horas.

    Única fuente de la fórmula: la usan create/update (persistencia) y
    `ForecastService.predict_only` (dry-run en memoria).
    """
    if horas_efectivas is not None:
        return round(kghora_pred * horas_efectivas, 4)
    return None


def _to_orm_kwargs(forecast_data: ForecastCreate) -> dict:
    """Mapea ForecastCreate (campos snake) a kwargs del ORM Forecast.

    Centraliza el mapeo para que `create_forecast` y `create_forecasts_batch`
    no se desincronicen cuando cambia el schema.
    """
    return {
        "fecha": forecast_data.fecha,
        "external_id": forecast_data.external_id,
        "kg_ha": forecast_data.kg_ha,
        "indus_pct": forecast_data.indus_pct,
        "dpc": forecast_data.dpc,
        "p_baya": forecast_data.p_baya,
        "ha": forecast_data.ha,
        "dia_cosecha": forecast_data.dia_cosecha,
        "formato": forecast_data.formato,
        "fundo": forecast_data.fundo,
        "horas_efectivas": forecast_data.horas_efectivas,
    }


# ============================================================================
# READ
# ============================================================================


async def get_forecast_by_id(db: AsyncSession, forecast_id: int) -> Forecast:
    result = await db.execute(select(Forecast).where(Forecast.id == forecast_id))
    forecast = result.scalar_one_or_none()
    if forecast is None:
        raise ForecastNotFoundError(forecast_id)
    return forecast


async def get_forecasts(
    db: AsyncSession,
    variety: str | None = None,
    fecha: date | None = None,
    external_id: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    limit: int = 500,
    offset: int = 0,
) -> tuple[list[Forecast], int]:
    query = select(Forecast)

    if variety:
        query = query.where(Forecast.variety == variety.upper())
    if fecha:
        query = query.where(Forecast.fecha == fecha)
    if external_id:
        query = query.where(Forecast.external_id == external_id)
    if date_from:
        query = query.where(Forecast.created_at >= date_from)
    if date_to:
        query = query.where(Forecast.created_at <= date_to)

    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar_one()

    query = query.order_by(Forecast.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all()), total


# ============================================================================
# CREATE
# ============================================================================


async def create_forecast(
    db: AsyncSession,
    variety: str,
    forecast_data: ForecastCreate,
    kghora_pred: float,
) -> Forecast:
    db_forecast = Forecast(
        variety=variety.upper(),
        kghora_pred=kghora_pred,
        kgjn_pred=calc_kgjn(kghora_pred, forecast_data.horas_efectivas),
        **_to_orm_kwargs(forecast_data),
    )
    db.add(db_forecast)
    await db.commit()
    await db.refresh(db_forecast)
    return db_forecast


async def create_forecasts_batch(
    db: AsyncSession,
    variety: str,
    forecasts_data: list[ForecastCreate],
    kghora_preds: list[float],
) -> list[Forecast]:
    db_forecasts = [
        Forecast(
            variety=variety.upper(),
            kghora_pred=kghora_pred,
            kgjn_pred=calc_kgjn(kghora_pred, forecast_data.horas_efectivas),
            **_to_orm_kwargs(forecast_data),
        )
        for forecast_data, kghora_pred in zip(forecasts_data, kghora_preds, strict=True)
    ]

    db.add_all(db_forecasts)
    await db.flush()

    ids = [f.id for f in db_forecasts]
    # order_by(id) es OBLIGATORIO: los id son autoincrementales en orden de
    # inserción, así que ordenar por id devuelve las filas en el mismo orden
    # que `forecasts_data`/`kghora_preds`. Sin esto, PostgreSQL no garantiza
    # el orden del IN (...) y el reporte de drift por fila (que el caller
    # empareja por índice) podría adjuntarse al pronóstico equivocado.
    # El re-fetch ocurre ANTES del commit para que todo el batch sea una sola
    # transacción: si algo falla, no quedan filas persistidas a medias.
    result = await db.execute(select(Forecast).where(Forecast.id.in_(ids)).order_by(Forecast.id))
    forecasts = list(result.scalars().all())
    await db.commit()
    return forecasts


# ============================================================================
# UPDATE
# ============================================================================


async def update_forecast(
    db: AsyncSession,
    forecast_id: int,
    forecast_update: ForecastUpdate,
) -> Forecast:
    forecast = await get_forecast_by_id(db, forecast_id)

    update_data = forecast_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(forecast, key, value)

    if forecast.kghora_pred is not None:
        forecast.kgjn_pred = calc_kgjn(forecast.kghora_pred, forecast.horas_efectivas)

    await db.commit()
    await db.refresh(forecast)
    return forecast


# ============================================================================
# DELETE
# ============================================================================


async def delete_forecast(db: AsyncSession, forecast_id: int) -> None:
    forecast = await get_forecast_by_id(db, forecast_id)
    await db.delete(forecast)
    await db.commit()


async def delete_forecasts_by_fecha(db: AsyncSession, fecha: date) -> int:
    result = await db.execute(delete(Forecast).where(Forecast.fecha == fecha))
    await db.commit()
    return result.rowcount
