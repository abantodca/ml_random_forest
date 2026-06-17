"""
Router de Health Check
======================
Endpoints para verificar el estado del servicio.

Endpoints:
----------
GET  /health          - Health check básico (con cache de 30s)
GET  /health/detailed - Health check detallado (sin cache)
POST /health/models/reload - Recarga modelos ML desde MLflow
"""

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter

from app.core import Variety
from app.dependencies import DbSession, MLflow
from app.schemas import (
    HealthDetailedResponse,
    HealthResponse,
    ModelInfo,
    ModelReloadResponse,
)
from app.services import check_database, determine_status, health_cache

router = APIRouter(prefix="/health", tags=["health"])

logger = logging.getLogger(__name__)


# ============================================================================
# Health Checks
# ============================================================================


@router.get("", response_model=HealthResponse)
async def health_check(
    db: DbSession,
    mlflow: MLflow,
) -> HealthResponse:
    """
    Health check básico con cache de 30 segundos.

    Retorna el estado del servicio de forma rápida sin sobrecargar
    las conexiones a PostgreSQL y MLflow en cada request.

    Returns:
        Estado del servicio, base de datos, MLflow y modelos cargados
    """
    cached = health_cache.get()
    if cached is not None:
        mlflow_ok, db_ok, models_available = cached
    else:
        mlflow_ok = await mlflow.check_connection()
        db_ok, _ = await check_database(db)
        # get_available_models() es una llamada de red BLOQUEANTE a MLflow;
        # se ejecuta en el threadpool para no congelar el event loop.
        loop = asyncio.get_running_loop()
        available = await loop.run_in_executor(None, mlflow.get_available_models)
        models_available = len(available)
        health_cache.set(mlflow_ok, db_ok, models_available)

    return HealthResponse(
        status=determine_status(mlflow_ok, db_ok),
        database_connected=db_ok,
        mlflow_connected=mlflow_ok,
        models_loaded=mlflow.models_loaded,
        models_available=models_available,
        total_varieties=len(Variety),
    )


@router.get("/detailed", response_model=HealthDetailedResponse)
async def health_check_detailed(
    db: DbSession,
    mlflow: MLflow,
) -> HealthDetailedResponse:
    """
    Health check detallado sin cache.

    Proporciona información completa sobre el estado del servicio,
    incluyendo listas de modelos cargados y pendientes.

    Returns:
        Estado detallado con timestamp y estado de cada componente
    """
    db_ok, db_error = await check_database(db)
    mlflow_ok = await mlflow.check_connection()

    # Red bloqueante → threadpool (ver health_check). is_loaded() es lookup
    # en memoria, no necesita offload.
    loop = asyncio.get_running_loop()
    available_models = await loop.run_in_executor(None, mlflow.get_available_models)
    loaded_models = [v.value for v in Variety if mlflow.is_loaded(v)]
    not_loaded = [v.value for v in Variety if not mlflow.is_loaded(v)]

    return HealthDetailedResponse(
        status=determine_status(mlflow_ok, db_ok),
        timestamp=datetime.now(),
        database={"connected": db_ok, "error": db_error},
        mlflow={
            "connected": mlflow_ok,
            "tracking_uri": mlflow.tracking_uri,
        },
        models={
            "loaded": loaded_models,
            "available": len(available_models),
            "not_loaded": not_loaded,
            "total_varieties": len(Variety),
        },
    )


# ============================================================================
# Model Management
# ============================================================================


@router.post("/models/reload", response_model=ModelReloadResponse, tags=["models"])
async def reload_models(
    mlflow: MLflow,
) -> ModelReloadResponse:
    """
    Recarga modelos ML desde MLflow sin reiniciar el servidor.

    Verifica si hay nuevas versiones de modelos y solo actualiza
    los que tienen versiones más recientes.

    Útil cuando:
    - Se entrena y registra un nuevo modelo en MLflow
    - Se detectan modelos desactualizados en producción

    Returns:
        Resumen de la operación con modelos actualizados/cargados/fallidos
    """
    result = await mlflow.reload_models()

    logger.info(
        "Models reload: loaded=%d updated=%d failed=%d",
        result["summary"]["loaded"],
        result["summary"]["updated"],
        result["summary"]["failed"],
    )

    models = [
        ModelInfo(
            variety=m["variety"],
            version=m["version"],
            status=m["status"],
            message=m.get("message"),
            training_params=m.get("training_params"),
        )
        for m in result["models"]
    ]

    return ModelReloadResponse(
        status="success" if result["summary"]["failed"] == 0 else "partial",
        models_loaded=result["models_loaded"],
        models_available=result["models_available"],
        summary=result["summary"],
        models=models,
    )
