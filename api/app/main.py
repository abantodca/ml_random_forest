"""
Main - Punto de entrada de la aplicación FastAPI
=================================================
Configura y crea la instancia de FastAPI con todos los componentes.
La lógica de handlers y middlewares está en app/core/ para mantener
este archivo enfocado solo en la composición de la aplicación.
"""

import logging
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

# Suprimir warnings de MLflow antes de cualquier importación
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
warnings.filterwarnings("ignore", message=".*Python.*differs.*")

# Configurar logging ANTES de importar otros módulos
from app.core import settings, setup_logger

setup_logger("rnd-forest-backend", level=settings.log_level)

from app.core import (
    ForecastNotFoundError,
    ModelNotAvailableError,
    RequestLoggingMiddleware,
    VarietyNotFoundError,
    forecast_not_found_handler,
    generic_exception_handler,
    model_not_available_handler,
    validation_exception_handler,
    value_error_handler,
    variety_not_found_handler,
)
from app.models import dispose_engine, init_db
from app.routers import forecasts, health, history, varieties
from app.services import DriftService, MLflowService

logger = logging.getLogger(__name__)


# ============================================================================
# Lifecycle Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación (startup y shutdown)."""
    logger.info("🚀 Iniciando RND Forest Backend...")

    # 1. Conexión a PostgreSQL
    try:
        await init_db(settings.database_url)
    except ConnectionError:
        logger.error("═" * 60)
        logger.error("❌ ERROR DE CONEXIÓN A POSTGRESQL")
        logger.error("   El backend no puede iniciar sin PostgreSQL.")
        logger.error("   Solución: docker-compose up -d")
        logger.error("═" * 60)
        raise

    # 2. Inicializar servicio MLflow y almacenarlo en app.state
    mlflow_service = MLflowService(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_prefix=settings.experiment_prefix,
        preload=settings.mlflow_preload_models,
    )
    app.state.mlflow_service = mlflow_service

    try:
        if not await mlflow_service.check_connection():
            logger.warning(
                "⚠️  MLflow no accesible en: %s — las predicciones fallarán.",
                settings.mlflow_tracking_uri,
            )
        mlflow_service.startup()
    except Exception as exc:
        logger.error("❌ Error al conectar con MLflow: %s", exc)
        logger.warning("   El backend iniciará sin modelos precargados.")

    # 3. DriftService: comparte MLflowService (reutiliza versiones y cache).
    #    Lazy build de baselines: la primera predicción de cada variedad
    #    paga el costo de leer history_/scaler una vez.
    app.state.drift_service = DriftService(mlflow_service)

    logger.info("✅ Backend iniciado | Docs: /docs | Health: /api/health")

    yield

    # Shutdown
    logger.info("⏹️  Deteniendo backend...")
    mlflow_service.shutdown()
    await dispose_engine()
    logger.info("✅ Backend detenido correctamente")


# ============================================================================
# Application Factory
# ============================================================================


def create_app() -> FastAPI:
    """Factory que crea y configura la aplicación FastAPI."""

    app = FastAPI(
        title=settings.app_name,
        description="API de predicción de KGHORA para variedades de bayas usando MLflow",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Middlewares (el último registrado se ejecuta primero en el pipeline)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    # Exception handlers (orden importa: más específico primero)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(VarietyNotFoundError, variety_not_found_handler)
    app.add_exception_handler(ModelNotAvailableError, model_not_available_handler)
    app.add_exception_handler(ForecastNotFoundError, forecast_not_found_handler)
    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Routers
    app.include_router(health.router, prefix="/api")
    app.include_router(varieties.router, prefix="/api")
    app.include_router(forecasts.router, prefix="/api")
    app.include_router(history.router, prefix="/api")

    @app.get("/", tags=["root"], include_in_schema=False)
    async def root():
        """Endpoint raíz con información del servicio."""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


# ============================================================================
# App Instance (para ASGI servers)
# ============================================================================

app = create_app()

# Para desarrollo: fastapi dev app/main.py
# Para producción: uvicorn app.main:app --host 0.0.0.0 --port 8000
