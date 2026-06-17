"""Adaptadores JSON → modelos Pydantic.

Sólo aplica cuando el JSON tiene un alias o requiere un fallback que el
modelo Pydantic no maneja por defecto. Para el resto, los servicios
llaman a `Model.model_validate(d)` directamente.
"""

from __future__ import annotations

from app.schemas import (
    ForecastListResult,
    ForecastRecord,
    ServiceHealth,
    VarietyViewModel,
)


def to_forecast(d: dict) -> ForecastRecord:
    return ForecastRecord.model_validate(d)


def to_forecast_list(d: dict) -> ForecastListResult:
    """Acepta tanto `{"items": [...]}` como `{"records": [...]}`."""
    payload = {**d, "items": d.get("items", d.get("records", []))}
    return ForecastListResult.model_validate(payload)


def to_health(d: dict) -> ServiceHealth:
    return ServiceHealth.model_validate(d)


def to_variety(d: dict, fallback_name: str) -> VarietyViewModel:
    """Si el backend no envía `name`, lo reemplaza por `fallback_name`."""
    payload = {**d, "name": d.get("name", fallback_name)}
    return VarietyViewModel.model_validate(payload)
