"""
Servicio de Health Check
=========================
Encapsula la lógica de verificación del estado del servicio:
cache con TTL (cachetools), ping a PostgreSQL y determinación del
estado general.
"""

import logging

from cachetools import TTLCache
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import settings

logger = logging.getLogger(__name__)


class HealthCache:
    """Cache TTL de un solo slot para resultados de health check.

    Wrapper minimo sobre `cachetools.TTLCache`: esconde la clave interna
    y expone `get()` / `set(...)` con la tripleta (mlflow_ok, db_ok,
    models_available). Evita que checks frecuentes (load balancers)
    saturen las conexiones a PostgreSQL y MLflow.
    """

    _KEY = "current"

    def __init__(self, ttl_seconds: int = settings.health_cache_ttl_seconds) -> None:
        self._cache: TTLCache[str, tuple[bool, bool, int]] = TTLCache(maxsize=1, ttl=ttl_seconds)

    def get(self) -> tuple[bool, bool, int] | None:
        """Retorna la tripleta cacheada o None si no hay valor vigente."""
        return self._cache.get(self._KEY)

    def set(self, mlflow_ok: bool, db_ok: bool, models_available: int) -> None:
        """Almacena el resultado del health check (TTL aplica desde aquí)."""
        self._cache[self._KEY] = (mlflow_ok, db_ok, models_available)


async def check_database(db: AsyncSession) -> tuple[bool, str | None]:
    """
    Verifica la conexión a PostgreSQL.

    Returns:
        (connected, error_message)
    """
    try:
        await db.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return False, str(exc)


def determine_status(mlflow_ok: bool, db_ok: bool) -> str:
    """
    Determina el estado general del servicio.

    Returns:
        "healthy"  — todo operativo
        "degraded" — DB ok pero MLflow caído (predicciones fallan)
        "unhealthy"— DB caída (servicio crítico)
    """
    if mlflow_ok and db_ok:
        return "healthy"
    if db_ok:
        return "degraded"
    return "unhealthy"


# Instancia compartida del cache (por proceso/worker)
health_cache = HealthCache()
