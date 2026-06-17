"""
Router de Varieties (Variedades)
=================================
Endpoints para información sobre variedades y sus modelos.

Endpoints:
----------
GET /varieties                       - Lista todas las variedades del catálogo
GET /varieties/available             - Variedades con modelos disponibles en MLflow
GET /varieties/{variety}             - Info de una variedad específica
GET /varieties/{variety}/dashboard   - Winner_<VARIETY>.html (reporte gerencial)
"""

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

from app.core import (
    FORMATO_DEFAULT,
    Formato,
    Fundo,
    ModelNotAvailableError,
    Variety,
)
from app.dependencies import MLflow, ValidatedVariety
from app.schemas import CatalogsResponse, VarietyInfo, VarietyList

router = APIRouter(prefix="/varieties", tags=["varieties"])

logger = logging.getLogger(__name__)


# ============================================================================
# Varieties Information
# ============================================================================


@router.get("/catalogs", response_model=CatalogsResponse)
async def get_catalogs() -> CatalogsResponse:
    """Catálogos cerrados (FORMATO, FUNDO) consumidos por el frontend."""
    return CatalogsResponse(
        formatos=list(Formato),
        formato_default=FORMATO_DEFAULT,
        fundos=list(Fundo),
    )


@router.get("", response_model=VarietyList)
async def list_varieties() -> VarietyList:
    """
    Lista todas las variedades soportadas por el sistema.

    Incluye variedades del catálogo, estén o no sus modelos
    cargados actualmente en MLflow.

    Returns:
        Lista completa de variedades y total
    """
    varieties = [v.value for v in Variety]
    return VarietyList(varieties=varieties, total=len(varieties))


@router.get("/available", response_model=VarietyList)
def list_available_varieties(
    mlflow: MLflow,
) -> VarietyList:
    """
    Lista variedades con modelos disponibles en MLflow.

    Solo retorna variedades que tienen modelos registrados
    y accesibles en el servidor MLflow activo.

    Definido como `def` (no `async`): `get_available_models` hace una
    llamada de red BLOQUEANTE a MLflow, así que Starlette lo corre en el
    threadpool y no congela el event loop.

    Returns:
        Lista de variedades con modelos disponibles
    """
    available = mlflow.get_available_models()
    return VarietyList(varieties=available, total=len(available))


@router.get("/{variety}", response_model=VarietyInfo)
def get_variety_info(
    variety: ValidatedVariety,
    mlflow: MLflow,
) -> VarietyInfo:
    """
    Obtiene información detallada de una variedad específica.

    Métricas + hiperparámetros del campeón vienen del REGISTRY (no del estado
    in-memory): así el dashboard no queda vacío tras un reinicio aunque el
    modelo aún no esté cargado por lazy-load.

    Definido como `def`: consulta MLflow (red bloqueante) en el primer acceso;
    corre en el threadpool de Starlette.

    Raises:
        404: Si la variedad no existe en el catálogo
    """
    info = mlflow.get_model_info(variety)  # {version, model_type, metrics, best_params}
    best_params = (
        {"model_type": info.get("model_type"), **info.get("best_params", {})}
        if info
        else {}
    )

    return VarietyInfo(
        name=variety,
        model_loaded=mlflow.is_loaded(variety),
        metrics=info.get("metrics", {}),
        version=info.get("version"),
        best_params=best_params,
    )


# ============================================================================
# Winner Dashboard (reporte gerencial HTML del modelo)
# ============================================================================


@router.get(
    "/{variety}/dashboard",
    response_class=HTMLResponse,
    responses={
        200: {"content": {"text/html": {}}, "description": "Reporte HTML"},
        404: {"description": "Reporte no encontrado para la variedad"},
        503: {"description": "Modelo MLflow no disponible"},
    },
)
def get_variety_dashboard(
    variety: ValidatedVariety,
    mlflow: MLflow,
) -> HTMLResponse:
    """
    Devuelve el reporte HTML `Winner_<VARIETY>.html` artifact del modelo.

    Pensado para que el frontend lo embeba en un iframe sin que el público
    necesite acceso directo al servidor MLflow. El backend descarga el
    artifact y cachea el contenido por (variety, run_id).

    Definido como `def`: la descarga del artifact y la lectura de archivo
    son BLOQUEANTES; corre en el threadpool de Starlette para no congelar
    el event loop.
    """
    try:
        html = mlflow.get_winner_dashboard_html(variety)
    except ModelNotAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reporte de '{variety}' no disponible en MLflow.",
        ) from exc
    return HTMLResponse(content=html, status_code=200)
