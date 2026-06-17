"""Constantes y constructores de paths del backend FastAPI."""

from __future__ import annotations

HEALTH = "/api/health"
RELOAD_MODELS = "/api/health/models/reload"

VARIETIES = "/api/varieties"
VARIETIES_AVAILABLE = "/api/varieties/available"
CATALOGS = "/api/varieties/catalogs"

FORECASTS = "/api/forecasts"


def history_list(variety: str) -> str:
    """Observaciones reales (KG/JR_H realizado) de una variedad."""
    return f"/api/history/{variety}"


def history_upload(variety: str) -> str:
    """Carga Excel de datos REALES (mismas columnas del pronóstico + KG/JR_H)."""
    return f"/api/history/{variety}/upload"


def forecast_predict(variety: str) -> str:
    """Predicción dry-run (no persiste) — usada en seguimiento y exploración."""
    return f"/api/forecasts/{variety}/predict"


def variety_detail(name: str) -> str:
    return f"/api/varieties/{name}"


def variety_dashboard(name: str) -> str:
    return f"/api/varieties/{name}/dashboard"


def forecast_create(variety: str) -> str:
    return f"/api/forecasts/{variety}"


def forecast_batch(variety: str) -> str:
    return f"/api/forecasts/{variety}/batch"


def forecast_upload(variety: str) -> str:
    return f"/api/forecasts/{variety}/upload"


def forecast_by_id(forecast_id: int) -> str:
    return f"/api/forecasts/{forecast_id}"


def forecasts_by_fecha(fecha: str) -> str:
    return f"/api/forecasts/fecha/{fecha}"
