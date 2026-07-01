"""
Configuración del servicio usando Pydantic Settings
====================================================
Lee variables de entorno con fallback a archivo .env
"""

from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración centralizada de la aplicación"""

    # Información del servicio
    app_name: str = "RND Forest Backend"
    app_version: str = "2.0.0"
    debug: bool = False

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_preload_models: bool = True
    # Prefijo COMPLETO del registered model. Debe coincidir con
    # MODEL_REGISTRY_PREFIX del trainer (src/config.py), default 'rnd-forest-'.
    experiment_prefix: str = "rnd-forest-"

    # Base de datos PostgreSQL (obligatorio via DATABASE_URL en .env o entorno)
    database_url: str

    # CORS
    # NoDecode: evita que pydantic-settings intente parsear el env var como JSON;
    # delegamos el split por comas al validator parse_cors.
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:8501",
        "http://localhost:3000",
    ]

    # Servidor
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Health check
    health_cache_ttl_seconds: int = 30

    # MLflow limits
    mlflow_max_registered_models: int = 1000

    # Excel upload
    max_excel_file_size_mb: int = 10

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: object) -> list[str]:
        """Convierte string separado por comas en lista"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("health_cache_ttl_seconds")
    @classmethod
    def validate_ttl(cls, v: int) -> int:
        """Un TTL <= 0 haría que el cache de health expirara de inmediato
        (siempre miss), anulando su propósito. Fail-fast al boot."""
        if v <= 0:
            raise ValueError("health_cache_ttl_seconds debe ser > 0")
        return v

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Singleton de configuración
settings = Settings()
