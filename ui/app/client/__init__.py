"""
api - Fachada de la capa HTTP hacia el backend FastAPI
=======================================================
Reúne el cliente HTTP y las constantes/builders de endpoints.

NOTA: los `mapeadores` (`to_forecast`, `to_health`, ...) NO se
re-exportan en este barrel a propósito: solo los consumen los
servicios internos y exponerlos aquí dispararía un ciclo
de imports client ↔ schemas. Quien los necesite los importa con
`from app.client.mappers import ...` (path directo).
"""

from app.client import endpoints
from app.client.api_client import ApiClient
from app.client.endpoints import (
    CATALOGS,
    FORECASTS,
    HEALTH,
    RELOAD_MODELS,
    VARIETIES,
    VARIETIES_AVAILABLE,
    forecast_batch,
    forecast_by_id,
    forecast_create,
    forecast_predict,
    forecast_predict_batch,
    forecast_upload,
    forecasts_by_fecha,
    history_list,
    history_upload,
    variety_dashboard,
    variety_detail,
)

__all__ = [
    "ApiClient",
    "HEALTH",
    "RELOAD_MODELS",
    "VARIETIES",
    "VARIETIES_AVAILABLE",
    "CATALOGS",
    "FORECASTS",
    "variety_detail",
    "variety_dashboard",
    "forecast_create",
    "forecast_batch",
    "forecast_predict",
    "forecast_predict_batch",
    "forecast_upload",
    "forecast_by_id",
    "forecasts_by_fecha",
    "history_list",
    "history_upload",
    "endpoints",
]
