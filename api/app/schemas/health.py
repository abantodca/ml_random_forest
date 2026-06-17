"""
Schemas de Pydantic para Health Check
======================================
Define los modelos para verificar el estado del servicio
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check básico (con cache)"""

    status: str = Field(..., description="Estado general: healthy, degraded, unhealthy")
    database_connected: bool = Field(..., description="Estado de PostgreSQL")
    mlflow_connected: bool = Field(..., description="Estado de MLflow")
    models_loaded: int = Field(..., description="Modelos cargados en memoria")
    models_available: int = Field(..., description="Modelos disponibles en MLflow")
    total_varieties: int = Field(..., description="Total de variedades soportadas")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "healthy",
                    "database_connected": True,
                    "mlflow_connected": True,
                    "models_loaded": 32,
                    "models_available": 32,
                    "total_varieties": 32,
                }
            ]
        }
    )


class HealthDetailedResponse(BaseModel):
    """Health check detallado (sin cache)"""

    status: str
    timestamp: datetime
    database: dict = Field(..., description="Estado y error de PostgreSQL si existe")
    mlflow: dict = Field(..., description="Estado y URI de MLflow")
    models: dict = Field(
        ...,
        description="Modelos cargados, disponibles, no cargados y total de variedades",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2026-04-13T10:00:00Z",
                    "database": {"connected": True, "error": None},
                    "mlflow": {
                        "connected": True,
                        "tracking_uri": "http://3.85.150.197:5000",
                    },
                    "models": {
                        "loaded": ["ATLAS", "BIANCA", "POP"],
                        "available": 32,
                        "not_loaded": [],
                        "total_varieties": 32,
                    },
                }
            ]
        }
    )


class ModelInfo(BaseModel):
    """Información de un modelo individual"""

    variety: str
    version: str
    status: str = Field(..., description="loaded, updated, failed, unchanged")
    message: str | None = None
    training_params: dict | None = Field(
        default=None,
        description="Parámetros y métricas de entrenamiento desde MLflow",
    )


class ModelReloadResponse(BaseModel):
    """Respuesta al recargar modelos"""

    status: str
    models_loaded: int
    models_available: int
    summary: dict[str, int] = Field(
        ..., description="Resumen: loaded, updated, unchanged, failed"
    )
    models: list[ModelInfo]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "success",
                    "models_loaded": 33,
                    "models_available": 33,
                    "summary": {
                        "loaded": 30,
                        "updated": 3,
                        "unchanged": 0,
                        "failed": 0,
                    },
                    "models": [
                        {
                            "variety": "ATLAS",
                            "version": "1",
                            "status": "loaded",
                            "message": None,
                        }
                    ],
                }
            ]
        }
    )
