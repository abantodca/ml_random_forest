"""Configuración inmutable derivada de variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.core.constants import (
    DEFAULT_API_URL,
    DEFAULT_CACHE_TTL_FORECASTS,
    DEFAULT_CACHE_TTL_HEALTH,
    DEFAULT_CACHE_TTL_VARIETIES,
    DEFAULT_LOG_LEVEL,
    DEFAULT_TIMEOUT_BATCH,
    DEFAULT_TIMEOUT_HEALTH,
    DEFAULT_TIMEOUT_READ,
    DEFAULT_TIMEOUT_WRITE,
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"La variable {name} debe ser entera, recibido {raw!r}"
        ) from exc


@dataclass(frozen=True)
class Configuracion:
    api_url: str
    timeout_health: int
    timeout_read: int
    timeout_write: int
    timeout_batch: int
    cache_ttl_health: int
    cache_ttl_varieties: int
    cache_ttl_forecasts: int
    log_level: str

    @classmethod
    def desde_entorno(cls) -> Configuracion:
        return cls(
            api_url=os.getenv("API_URL", DEFAULT_API_URL),
            timeout_health=_env_int("TIMEOUT_HEALTH", DEFAULT_TIMEOUT_HEALTH),
            timeout_read=_env_int("TIMEOUT_READ", DEFAULT_TIMEOUT_READ),
            timeout_write=_env_int("TIMEOUT_WRITE", DEFAULT_TIMEOUT_WRITE),
            timeout_batch=_env_int("TIMEOUT_BATCH", DEFAULT_TIMEOUT_BATCH),
            cache_ttl_health=_env_int("CACHE_TTL_HEALTH", DEFAULT_CACHE_TTL_HEALTH),
            cache_ttl_varieties=_env_int(
                "CACHE_TTL_VARIETIES", DEFAULT_CACHE_TTL_VARIETIES
            ),
            cache_ttl_forecasts=_env_int(
                "CACHE_TTL_FORECASTS", DEFAULT_CACHE_TTL_FORECASTS
            ),
            log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
        )
