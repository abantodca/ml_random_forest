"""services - Lógica de negocio del frontend (equivalente a api/app/services).

Orden de import deliberado (servicios puros primero, validación al final):
los servicios y builders puros primero; `batch_validation` al final porque
necesita el composition root (`app.dependencies`) — que importa estos
servicios — y lo resuelve con un import local para no crear un ciclo.
"""

from app.services.batch_validation import (
    BatchValidationError,
    ValidationIssue,
    validate_batch_upload,
)
from app.services.forecast_service import ForecastService
from app.services.health_service import HealthService
from app.services.payload_builder import (
    build_forecast_payload,
    build_prediction_payload,
    row_to_record,
)
from app.services.tracking_service import TrackingService, forecast_verdict
from app.services.variety_service import VarietyService

__all__ = [
    # Servicios
    "ForecastService",
    "HealthService",
    "TrackingService",
    "VarietyService",
    "forecast_verdict",
    # Constructores de payload
    "build_forecast_payload",
    "build_prediction_payload",
    "row_to_record",
    # Validación de lotes
    "BatchValidationError",
    "ValidationIssue",
    "validate_batch_upload",
]
