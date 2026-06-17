"""CRUD HistoricalObservation
==============================
Persistencia y consulta del historial de observaciones reales (KG/HA +
KG/JR_H). El historial se usa para mostrar al usuario la base que el
modelo memorizo internamente durante el entrenamiento (via
`LagFeatureTransformer.history_` empaquetado en el pickle MLflow), no
para alimentar lags en runtime.
"""

import logging

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import HistoricalObservation
from app.schemas import HistoricalObservationCreate

logger = logging.getLogger(__name__)


# ============================================================================
# Lectura listado
# ============================================================================


async def list_history(
    db: AsyncSession,
    variety: str,
    limit: int = 500,
    offset: int = 0,
) -> tuple[list[HistoricalObservation], int]:
    base = select(HistoricalObservation).where(
        HistoricalObservation.variety == variety.upper()
    )
    total = (
        await db.execute(select(func.count()).select_from(base.subquery()))
    ).scalar_one()
    rows = (
        await db.execute(
            base.order_by(HistoricalObservation.fecha.desc())
            .offset(offset)
            .limit(limit)
        )
    ).scalars().all()
    return list(rows), total


# ============================================================================
# Insercion
# ============================================================================


async def bulk_insert(
    db: AsyncSession,
    variety: str,
    rows: list[HistoricalObservationCreate],
) -> int:
    """Inserta N observaciones en una sola transaccion. Devuelve el conteo."""
    if not rows:
        return 0

    db.add_all(
        [
            HistoricalObservation(
                variety=variety.upper(),
                fundo=r.fundo,
                formato=r.formato,
                fecha=r.fecha,
                kg_ha=r.kg_ha,
                kg_jr_h=r.kg_jr_h,
                dpc=r.dpc,
                indus_pct=r.indus_pct,
                p_baya=r.p_baya,
                ha=r.ha,
                dia_cosecha=r.dia_cosecha,
            )
            for r in rows
        ]
    )
    await db.commit()
    return len(rows)


# ============================================================================
# Borrado
# ============================================================================


async def delete_by_variety(db: AsyncSession, variety: str) -> int:
    """Borra todas las observaciones de una variedad. Util para re-seed."""
    result = await db.execute(
        delete(HistoricalObservation).where(
            HistoricalObservation.variety == variety.upper()
        )
    )
    await db.commit()
    return result.rowcount
