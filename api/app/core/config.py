"""
Configuración del servicio usando Pydantic Settings
====================================================
Lee variables de entorno con fallback a archivo .env
"""

from typing import Annotated
from urllib.parse import urlsplit

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración centralizada de la aplicación"""

    app_name: str = "RND Forest Backend"
    app_version: str = "2.0.0"
    debug: bool = False

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_preload_models: bool = True
    experiment_prefix: str = "rnd-forest-"

    database_url: str

    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:8501",
        "http://localhost:3000",
    ]

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    health_cache_ttl_seconds: int = 30

    mlflow_max_registered_models: int = 1000

    max_excel_file_size_mb: int = 10

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v: object) -> list[str]:
        """Convierte string separado por comas en lista"""
        if isinstance(v, str):
            v = [origin.strip() for origin in v.split(",") if origin.strip()]
        if not isinstance(v, list) or not v:
            raise ValueError("cors_origins debe contener al menos un origen")
        origins: list[str] = []
        for origin in v:
            candidate = str(origin).strip().rstrip("/")
            parsed = urlsplit(candidate)
            if candidate == "*" or parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"Origen CORS inválido: {candidate!r}")
            origins.append(candidate)
        return list(dict.fromkeys(origins))

    @field_validator(
        "health_cache_ttl_seconds",
        "mlflow_max_registered_models",
        "max_excel_file_size_mb",
        "port",
    )
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Los límites y TTL no admiten cero ni valores negativos."""
        if v <= 0:
            raise ValueError("el valor debe ser > 0")
        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"postgresql", "postgresql+asyncpg"} or not parsed.hostname:
            raise ValueError("database_url debe ser una URL PostgreSQL válida")
        if not parsed.path or parsed.path == "/":
            raise ValueError("database_url debe incluir el nombre de la base")
        return value

    @field_validator("mlflow_tracking_uri")
    @classmethod
    def validate_mlflow_uri(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("mlflow_tracking_uri debe ser una URL HTTP(S) válida")
        return value.rstrip("/")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"debug", "info", "warning", "error", "critical"}:
            raise ValueError("log_level inválido")
        return normalized

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
