"""
app.core - Fachada pública del paquete core
============================================
Centraliza la API pública (config, excepciones, handlers, helpers,
logging, middleware y catálogos) para que el resto de la app importe
solo desde `app.core` y no dependa de la estructura interna de archivos.
"""

from app.core.catalogs import (
    FORMATO_DEFAULT,
    Formato,
    Fundo,
    normalize_formato,
    normalize_fundo,
)
from app.core.config import Settings, settings
from app.core.excel_helpers import (
    parse_date_value,
    read_excel_dataframe,
    validate_excel_file,
    validate_upload_size,
)
from app.core.exception_handlers import (
    forecast_not_found_handler,
    generic_exception_handler,
    model_not_available_handler,
    validation_exception_handler,
    value_error_handler,
    variety_not_found_handler,
)
from app.core.exceptions import (
    ForecastNotFoundError,
    ModelNotAvailableError,
    PredictionError,
    VarietyNotFoundError,
)
from app.core.logger import setup_logger
from app.core.middleware import RequestLoggingMiddleware
from app.core.varieties import Variety, validate_variety

__all__ = [
    # Config
    "Settings",
    "settings",
    # Catálogos (StrEnum + helpers)
    "Formato",
    "Fundo",
    "Variety",
    "FORMATO_DEFAULT",
    "normalize_formato",
    "normalize_fundo",
    "validate_variety",
    # Excel helpers
    "parse_date_value",
    "read_excel_dataframe",
    "validate_excel_file",
    "validate_upload_size",
    # Excepciones de dominio
    "ForecastNotFoundError",
    "ModelNotAvailableError",
    "PredictionError",
    "VarietyNotFoundError",
    # Handlers FastAPI
    "forecast_not_found_handler",
    "generic_exception_handler",
    "model_not_available_handler",
    "validation_exception_handler",
    "value_error_handler",
    "variety_not_found_handler",
    # Logging y middleware
    "setup_logger",
    "RequestLoggingMiddleware",
]
