"""
app.services - Fachada pública de la capa de servicios
=======================================================
Centraliza los servicios de negocio (Excel, FeaturePipeline, MLflow,
HealthCache) detrás de un único punto de import. Los routers pueden
hacer `from app.services import MLflowService, FeaturePipeline, ...`
sin saber en qué archivo concreto está implementado cada uno.
"""

from app.services.drift_baseline import DriftBaselineExtractor
from app.services.drift_service import DriftService
from app.services.excel_service import parse_excel_to_forecasts
from app.services.feature_pipeline import MODEL_INPUT_COLUMNS, FeaturePipeline
from app.services.forecast_service import ForecastService
from app.services.health_service import (
    HealthCache,
    check_database,
    determine_status,
    health_cache,
)
from app.services.mlflow_service import MLflowService, ModelVersionInfo

__all__ = [
    # Excel
    "parse_excel_to_forecasts",
    # Feature engineering
    "FeaturePipeline",
    "MODEL_INPUT_COLUMNS",
    # Orquestación de pronósticos
    "ForecastService",
    # Drift
    "DriftService",
    "DriftBaselineExtractor",
    # Health
    "HealthCache",
    "check_database",
    "determine_status",
    "health_cache",
    # MLflow
    "MLflowService",
    "ModelVersionInfo",
]
