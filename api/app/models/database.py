"""
Configuración de la base de datos PostgreSQL (async)
=====================================================
Gestiona la conexión y las sesiones de SQLAlchemy
"""

import logging
from collections.abc import AsyncGenerator
from urllib.parse import urlsplit, urlunsplit

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class para todos los modelos ORM"""

    pass


# Variables globales de motor y sesión
_engine: AsyncEngine | None = None
_async_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Retorna el motor de base de datos (debe estar inicializado)"""
    if _engine is None:
        raise RuntimeError(
            "El motor de base de datos no está inicializado. "
            "Llama a init_db() primero."
        )
    return _engine


async def ensure_database(database_url: str) -> None:
    """Crea la base de datos destino si no existe (idempotente).

    En local el initdb de Postgres ya crea ``forecasts``; en producción el
    RDS de MLflow solo trae la DB ``mlflow``, así que en el primer arranque la
    API crea aquí su propia base. Se conecta a la DB de mantenimiento
    ``postgres`` con las mismas credenciales. Tolerante: si no hay permisos o
    no existe ``postgres``, se asume que la DB destino ya está creada y se
    continúa (init_db fallará con un error claro si realmente no existe).
    """
    raw = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    parts = urlsplit(raw)
    target_db = parts.path.lstrip("/")
    if not target_db:
        return

    maintenance_dsn = urlunsplit(parts._replace(path="/postgres"))
    try:
        conn = await asyncpg.connect(dsn=maintenance_dsn, timeout=10)
    except Exception as exc:  # noqa: BLE001 — best-effort bootstrap
        logger.warning(
            "ensure_database: no se pudo conectar a 'postgres' (%s); "
            "se asume que '%s' ya existe.",
            exc,
            target_db,
        )
        return

    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", target_db
        )
        if not exists:
            # CREATE DATABASE no admite parámetros y no puede ir en transacción;
            # el nombre proviene de nuestra config, no de input externo.
            await conn.execute(f'CREATE DATABASE "{target_db}"')
            logger.info("✅ Base de datos '%s' creada", target_db)
    finally:
        await conn.close()


async def init_db(database_url: str) -> None:
    """
    Inicializa la conexión a PostgreSQL.

    Args:
        database_url: URL de conexión de PostgreSQL

    Raises:
        ConnectionError: Si no se puede conectar a la base de datos
    """
    global _engine, _async_session_maker

    try:
        # Crea la DB destino si falta (no-op si ya existe).
        await ensure_database(database_url)

        # Convertir postgresql:// a postgresql+asyncpg://
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )

        # Crear motor async
        _engine = create_async_engine(
            database_url,
            echo=False,  # Cambia a True para debug SQL
            pool_pre_ping=True,  # Verifica conexión antes de usar
            pool_size=5,
            max_overflow=10,
        )

        # Crear session maker
        _async_session_maker = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Verificar conexión
        async with _engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        logger.info("✅ Conexión a PostgreSQL establecida")

        # Crear tablas si no existen — el import explicito registra los
        # mapeos en Base.metadata antes del create_all.
        from app.models.forecast import Forecast  # noqa: F401
        from app.models.historical_observation import HistoricalObservation  # noqa: F401

        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Migración aditiva idempotente: create_all NO agrega columnas a
            # tablas ya existentes. Para DBs creadas antes de enriquecer
            # historical_observations con las features reales, las añadimos con
            # ADD COLUMN IF NOT EXISTS (no-op en DBs nuevas). Nombres/tipos son
            # literales fijos del código, no input externo.
            for _col, _ddl_type in (
                ("dpc", "DOUBLE PRECISION"),
                ("indus_pct", "DOUBLE PRECISION"),
                ("p_baya", "DOUBLE PRECISION"),
                ("ha", "DOUBLE PRECISION"),
                ("dia_cosecha", "INTEGER"),
            ):
                await conn.execute(
                    text(
                        "ALTER TABLE historical_observations "
                        f"ADD COLUMN IF NOT EXISTS {_col} {_ddl_type}"
                    )
                )

        logger.info("✅ Tablas verificadas/creadas")

    except Exception as e:
        logger.error("❌ Error conectando a PostgreSQL: %s", e)
        raise ConnectionError(f"No se pudo conectar a PostgreSQL: {e}") from e


async def dispose_engine() -> None:
    """Cierra la conexión a la base de datos"""
    global _engine, _async_session_maker

    if _engine:
        await _engine.dispose()
        logger.info("Database connection closed")
        _engine = None
        _async_session_maker = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency de FastAPI: produce una sesión y hace rollback ante error."""
    if _async_session_maker is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with _async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
