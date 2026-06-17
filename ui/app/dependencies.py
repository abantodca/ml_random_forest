"""Wrappers `@st.cache_*` sobre los servicios — única fuente de cache."""

from __future__ import annotations

import streamlit as st

from app.client.api_client import ApiClient
from app.core import (
    CACHE_TTL_DASHBOARD_HTML,
    FORMATO_DEFAULT_FALLBACK,
    FORMATOS_FALLBACK,
    FUNDOS_FALLBACK,
    ApiConnectionError,
    ApiResponseError,
    Configuracion,
    logger,
)
from app.schemas import AccuracyPoint, Catalogs, ServiceHealth, VarietyViewModel
from app.services.forecast_service import ForecastService
from app.services.health_service import HealthService
from app.services.tracking_service import TrackingService
from app.services.variety_service import VarietyService

_cfg = Configuracion.desde_entorno()

_CATALOGS_FALLBACK = Catalogs(
    formatos=FORMATOS_FALLBACK,
    formato_default=FORMATO_DEFAULT_FALLBACK,
    fundos=FUNDOS_FALLBACK,
)


def get_config() -> Configuracion:
    return _cfg


@st.cache_resource(show_spinner=False)
def get_api_client() -> ApiClient:
    return ApiClient(_cfg)


def get_health_service() -> HealthService:
    return HealthService(get_api_client())


def get_variety_service() -> VarietyService:
    return VarietyService(get_api_client())


def get_forecast_service() -> ForecastService:
    return ForecastService(get_api_client())


def get_tracking_service() -> TrackingService:
    return TrackingService(get_api_client())


@st.cache_data(ttl=_cfg.cache_ttl_health, show_spinner=False)
def get_cached_health() -> ServiceHealth | None:
    return get_health_service().get()


@st.cache_data(ttl=_cfg.cache_ttl_varieties, show_spinner=False)
def get_cached_varieties() -> list[VarietyViewModel]:
    return get_variety_service().list_all()


def get_loaded_variety_names() -> list[str]:
    return [v.name for v in get_cached_varieties() if v.model_loaded]


def get_all_variety_names() -> list[str]:
    return [v.name for v in get_cached_varieties()]


@st.cache_data(ttl=CACHE_TTL_DASHBOARD_HTML, show_spinner=False)
def get_cached_dashboard_html(variety: str) -> str:
    return get_variety_service().get_dashboard_html(variety)


@st.cache_data(ttl=_cfg.cache_ttl_varieties, show_spinner=False)
def get_cached_catalogs() -> Catalogs:
    """Catálogos (FORMATO, FUNDO) — fallback local si el backend no responde."""
    try:
        return get_variety_service().get_catalogs()
    except (ApiConnectionError, ApiResponseError) as exc:
        logger.warning("Usando catálogos fallback: %s", exc)
        return _CATALOGS_FALLBACK


@st.cache_data(ttl=_cfg.cache_ttl_varieties, show_spinner=False)
def get_cached_accuracy(
    variety: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    with_decomposition: bool = True,
) -> list[AccuracyPoint]:
    """Pares proyectado↔real (con descomposición de error) cacheados.

    Se cachea porque `with_decomposition=True` dispara una predicción dry-run
    por punto; sin cache, cada rerun de Streamlit re-pegaría al backend.
    """
    try:
        return get_tracking_service().build_accuracy(
            variety,
            date_from=date_from,
            date_to=date_to,
            with_decomposition=with_decomposition,
        )
    except (ApiConnectionError, ApiResponseError) as exc:
        logger.warning("No se pudo construir accuracy de %s: %s", variety, exc)
        return []


def reload_models_and_clear_cache() -> dict:
    """Recarga modelos en el backend e invalida los caches relacionados."""
    result = get_health_service().reload_models()
    if "error" not in result:
        get_cached_health.clear()
        get_cached_varieties.clear()
        get_cached_catalogs.clear()
        get_cached_accuracy.clear()
    return result
