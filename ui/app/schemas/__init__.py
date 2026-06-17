"""schemas - DTOs de presentación (equivalente a api/app/schemas).

Modelos puros (sin dependencias de otras capas) que viajan entre los
servicios y las vistas. Romper la regla "schemas no importa client/services"
reintroduciría el ciclo `client.mappers ⇄ schemas`.
"""

from app.schemas.models import (
    AccuracyPoint,
    BatchDriftReport,
    BatchFeatureDrift,
    Catalogs,
    DriftPerFeature,
    DriftReport,
    DriftStatus,
    ForecastListResult,
    ForecastRecord,
    HistoricalObservation,
    PredictionResult,
    RowStatusCounts,
    ServiceHealth,
    TrainingWindow,
    VarietyViewModel,
    WeekAggregate,
)

__all__ = [
    "AccuracyPoint",
    "BatchDriftReport",
    "BatchFeatureDrift",
    "Catalogs",
    "DriftPerFeature",
    "DriftReport",
    "DriftStatus",
    "ForecastListResult",
    "ForecastRecord",
    "HistoricalObservation",
    "PredictionResult",
    "RowStatusCounts",
    "ServiceHealth",
    "TrainingWindow",
    "VarietyViewModel",
    "WeekAggregate",
]
