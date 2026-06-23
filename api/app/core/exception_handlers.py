"""
Manejadores de excepciones de FastAPI
======================================
Centraliza todos los handlers de errores del dominio y de HTTP.
Se registran en main.py usando app.add_exception_handler().
"""

import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    ForecastNotFoundError,
    ModelNotAvailableError,
    VarietyNotFoundError,
)

logger = logging.getLogger(__name__)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Maneja errores de validación de Pydantic (422)."""
    errors = [
        {
            "field": ".".join(str(x) for x in error["loc"][1:]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]

    logger.warning(
        "Validation error on %s %s: %s",
        request.method,
        request.url.path,
        errors,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Error de validación en los datos de entrada",
            "errors": errors,
            "path": request.url.path,
        },
    )


async def variety_not_found_handler(request: Request, exc: VarietyNotFoundError) -> JSONResponse:
    """Maneja errores de variedad no encontrada (404)."""
    logger.warning(
        "Variety not found: '%s' on %s %s",
        exc.variety,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "variety_not_found",
            "message": str(exc),
            "variety": exc.variety,
            "path": request.url.path,
        },
    )


async def model_not_available_handler(
    request: Request, exc: ModelNotAvailableError
) -> JSONResponse:
    """Maneja errores de modelo no disponible en MLflow (503)."""
    logger.error(
        "Model not available: '%s' on %s %s",
        exc.variety,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "model_not_available",
            "message": str(exc),
            "variety": exc.variety,
            "path": request.url.path,
        },
    )


async def forecast_not_found_handler(request: Request, exc: ForecastNotFoundError) -> JSONResponse:
    """Maneja errores de pronóstico no encontrado (404)."""
    logger.warning(
        "Forecast not found: id=%s on %s %s",
        exc.forecast_id,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "forecast_not_found",
            "message": str(exc),
            "forecast_id": exc.forecast_id,
            "path": request.url.path,
        },
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Maneja errores de validación de dominio (400)."""
    logger.warning(
        "Value error on %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "bad_request",
            "message": str(exc),
            "path": request.url.path,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Maneja excepciones no capturadas (500)."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "Error interno del servidor",
            "path": request.url.path,
        },
    )
