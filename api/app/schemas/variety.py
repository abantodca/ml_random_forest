"""
Schemas de Pydantic para Varieties
===================================
Define los modelos para información de variedades.
`CatalogsResponse` reutiliza los `StrEnum` de `app.core` para que
OpenAPI documente los valores válidos automáticamente.
"""

from pydantic import BaseModel, ConfigDict, Field

from app.core import Formato, Fundo


class VarietyInfo(BaseModel):
    """Información de una variedad"""

    name: str
    model_loaded: bool
    metrics: dict = Field(default_factory=dict)
    version: int | None = Field(default=None, description="Versión del modelo en el registry")
    best_params: dict = Field(
        default_factory=dict,
        description="Hiperparámetros del campeón (incluye model_type) para el dashboard MLOps",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "ATLAS",
                    "model_loaded": True,
                    "metrics": {
                        "r2_score": 0.95,
                        "rmse": 12.3,
                        "mae": 8.5,
                    },
                }
            ]
        }
    )


class VarietyList(BaseModel):
    """Lista de variedades disponibles"""

    varieties: list[str]
    total: int

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "varieties": ["ATLAS", "BIANCA", "POP"],
                    "total": 3,
                }
            ]
        }
    )


class CatalogsResponse(BaseModel):
    """Catálogos cerrados consumidos por el frontend."""

    formatos: list[Formato]
    formato_default: Formato
    fundos: list[Fundo]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "formatos": [
                        "CLAMSHELL 4.4 OZ",
                        "CLAMSHELL 11 OZ",
                        "CLAMSHELL 18 OZ",
                        "GRANEL",
                        "OTROS",
                    ],
                    "formato_default": "CLAMSHELL 4.4 OZ",
                    "fundos": ["C5", "A9", "LN", "C6"],
                }
            ]
        }
    )
