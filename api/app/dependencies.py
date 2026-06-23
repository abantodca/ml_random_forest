"""
Dependencias compartidas de FastAPI
====================================
Fuente única de verdad para todas las dependencias inyectables.
Los routers importan los Annotated aliases desde aquí en lugar
de redefinirlos localmente.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Path, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import VarietyNotFoundError, validate_variety
from app.models import get_session
from app.services import (
    DriftService,
    FeaturePipeline,
    ForecastService,
    MLflowService,
)

# ============================================================================
# Database
# ============================================================================


DbSession = Annotated[AsyncSession, Depends(get_session)]


# ============================================================================
# MLflow Service
# ============================================================================


def get_mlflow_service(request: Request) -> MLflowService:
    """
    Dependency de FastAPI para obtener el servicio MLflow desde app.state.

    El servicio se inicializa en el lifespan de la aplicación y se almacena
    en app.state, evitando variables globales a nivel de módulo.

    Raises:
        RuntimeError: Si el servicio no está inicializado (startup fallido)
    """
    service: MLflowService | None = getattr(request.app.state, "mlflow_service", None)
    if service is None:
        raise RuntimeError(
            "MLflow service no inicializado. "
            "Verifica que la aplicación haya arrancado correctamente."
        )
    return service


MLflow = Annotated[MLflowService, Depends(get_mlflow_service)]


# ============================================================================
# Drift Service
# ============================================================================


def get_drift_service(request: Request) -> DriftService:
    """Devuelve el DriftService singleton inicializado en el lifespan.

    Mismo patrón que `get_mlflow_service`: el servicio se construye una
    vez al arrancar (compartiendo `MLflowService` para reutilizar la
    conexión y el cache de versiones) y se persiste en `app.state`.
    """
    service: DriftService | None = getattr(request.app.state, "drift_service", None)
    if service is None:
        # Fallback defensivo: si el lifespan no lo inicializó (e.g. en tests),
        # lo construimos al vuelo con el MLflowService disponible.
        mlflow_service: MLflowService | None = getattr(request.app.state, "mlflow_service", None)
        if mlflow_service is None:
            raise RuntimeError(
                "Drift service no inicializado y MLflow service tampoco. "
                "Verifica que la aplicación haya arrancado correctamente."
            )
        service = DriftService(mlflow_service)
        request.app.state.drift_service = service
    return service


Drift = Annotated[DriftService, Depends(get_drift_service)]


# ============================================================================
# Feature Pipeline
# ============================================================================


@lru_cache(maxsize=1)
def get_feature_pipeline() -> FeaturePipeline:
    """`FeaturePipeline` es stateless; lru_cache garantiza un único
    instance por proceso sin globales explícitas."""
    return FeaturePipeline()


Features = Annotated[FeaturePipeline, Depends(get_feature_pipeline)]


# ============================================================================
# Forecast Service (orquestación)
# ============================================================================


def get_forecast_service(
    mlflow: MLflow,
    features: Features,
    drift: Drift,
) -> ForecastService:
    """Construye el `ForecastService` por request a partir de los servicios
    ya inyectados (MLflow, FeaturePipeline, DriftService).

    El servicio es un orquestador liviano (solo guarda referencias), así que
    instanciarlo por request es barato y evita estado compartido mutable.
    """
    return ForecastService(mlflow=mlflow, features=features, drift=drift)


ForecastSvc = Annotated[ForecastService, Depends(get_forecast_service)]


# ============================================================================
# Variety Validation
# ============================================================================


def validate_optional_variety(variety: str | None) -> str | None:
    """Valida/normaliza una variedad opcional (p. ej. query param).

    Fuente única de la conversión "valida o lanza 404": `None` pasa sin
    tocar; un valor inválido se traduce a `VarietyNotFoundError`. Tanto el
    dependency de path (`get_validated_variety`) como los endpoints que
    reciben la variedad por query reutilizan esta función en vez de repetir
    el bloque try/except.

    Raises:
        VarietyNotFoundError: Si la variedad no está en el catálogo.
    """
    if variety is None:
        return None
    try:
        return validate_variety(variety)
    except ValueError as exc:
        raise VarietyNotFoundError(variety) from exc


def get_validated_variety(
    variety: str = Path(..., description="Nombre de la variedad (ej: ATLAS, BIANCA)"),
) -> str:
    """
    Dependency que valida y normaliza el nombre de una variedad desde el path.

    Elimina la necesidad de repetir el bloque try/except en cada endpoint
    que recibe {variety} como parámetro de ruta.

    Returns:
        Nombre de variedad normalizado en mayúsculas

    Raises:
        VarietyNotFoundError: Si la variedad no está en el catálogo
    """
    # variety nunca es None aquí (Path obligatorio); el cast tranquiliza al
    # type checker sobre el retorno str | None de la función compartida.
    return validate_optional_variety(variety)  # type: ignore[return-value]


ValidatedVariety = Annotated[str, Depends(get_validated_variety)]
