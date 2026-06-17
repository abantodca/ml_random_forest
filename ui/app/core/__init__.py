"""
core - Fachada de la capa base
=================================
Centraliza configuración, constantes, excepciones y logging para que
el resto del frontend importe `from app.core import ...` sin acoplarse
a archivos internos.
"""

from app.core.config import Configuracion
from app.core.constants import (
    CACHE_TTL_DASHBOARD_HTML,
    COLUMNAS_OPCIONALES,
    COLUMNAS_REQUERIDAS,
    DEFAULT_API_URL,
    DEFAULT_CACHE_TTL_FORECASTS,
    DEFAULT_CACHE_TTL_HEALTH,
    DEFAULT_CACHE_TTL_VARIETIES,
    DEFAULT_LOG_LEVEL,
    DEFAULT_TIMEOUT_BATCH,
    DEFAULT_TIMEOUT_HEALTH,
    DEFAULT_TIMEOUT_READ,
    DEFAULT_TIMEOUT_WRITE,
    FORMATO_DEFAULT_FALLBACK,
    FORMATOS_FALLBACK,
    FUNDOS_FALLBACK,
    LOGGER_NAME,
    LONGITUD_VISIBLE_API_URL,
    PALETA_SERIES,
    TEMA,
    WORKERS_VARIETY_DETAIL_MAX,
    WORKERS_VARIETY_ROOT,
)
from app.core.exceptions import ApiConnectionError, ApiResponseError
from app.core.logger import logger

__all__ = [
    # Configuración
    "Configuracion",
    # Constantes - red / IO
    "DEFAULT_API_URL",
    "DEFAULT_TIMEOUT_HEALTH",
    "DEFAULT_TIMEOUT_READ",
    "DEFAULT_TIMEOUT_WRITE",
    "DEFAULT_TIMEOUT_BATCH",
    # Constantes - caché
    "DEFAULT_CACHE_TTL_HEALTH",
    "DEFAULT_CACHE_TTL_VARIETIES",
    "DEFAULT_CACHE_TTL_FORECASTS",
    "CACHE_TTL_DASHBOARD_HTML",
    # Constantes - logging
    "DEFAULT_LOG_LEVEL",
    "LOGGER_NAME",
    # Constantes - workers / UI
    "WORKERS_VARIETY_ROOT",
    "WORKERS_VARIETY_DETAIL_MAX",
    "LONGITUD_VISIBLE_API_URL",
    # Constantes - tema y datos
    "TEMA",
    "PALETA_SERIES",
    "COLUMNAS_REQUERIDAS",
    "COLUMNAS_OPCIONALES",
    "FORMATOS_FALLBACK",
    "FORMATO_DEFAULT_FALLBACK",
    "FUNDOS_FALLBACK",
    # Excepciones de dominio
    "ApiConnectionError",
    "ApiResponseError",
    # Logging
    "logger",
]
