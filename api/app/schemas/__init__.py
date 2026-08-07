"""
app.schemas - Fachada pública de los schemas Pydantic
======================================================
Centraliza los modelos de validación (request / response) de toda la
API. Los routers importan desde `app.schemas` y no necesitan conocer
en qué archivo concreto vive cada schema.
"""

from app.schemas.forecast import (
    BatchDriftReport,
    BatchFeatureDrift,
    DeletedCountResponse,
    DriftPerFeature,
    DriftReport,
    DriftStatus,
    ForecastBatchCreate,
    ForecastCreate,
    ForecastListResponse,
    ForecastResponse,
    ForecastUpdate,
    PredictionBatchResponse,
    PredictionResponse,
    RowStatusCounts,
    TrainingWindow,
)
from app.schemas.health import (
    HealthDetailedResponse,
    HealthResponse,
    ModelInfo,
    ModelReloadResponse,
)
from app.schemas.historical_observation import (
    HistoricalObservationCreate,
    HistoricalObservationListResponse,
    HistoricalObservationResponse,
    HistoryImportResponse,
)
from app.schemas.variety import CatalogsResponse, VarietyInfo, VarietyList

__all__ = [
    "BatchDriftReport",
    "BatchFeatureDrift",
    "DeletedCountResponse",
    "DriftPerFeature",
    "DriftReport",
    "DriftStatus",
    "ForecastBatchCreate",
    "ForecastCreate",
    "ForecastListResponse",
    "ForecastResponse",
    "ForecastUpdate",
    "PredictionBatchResponse",
    "PredictionResponse",
    "RowStatusCounts",
    "TrainingWindow",
    "HealthDetailedResponse",
    "HealthResponse",
    "ModelInfo",
    "ModelReloadResponse",
    "HistoricalObservationCreate",
    "HistoricalObservationListResponse",
    "HistoricalObservationResponse",
    "HistoryImportResponse",
    "CatalogsResponse",
    "VarietyInfo",
    "VarietyList",
]
