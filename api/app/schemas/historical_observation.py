"""Schemas Pydantic para HistoricalObservation."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core import (
    FORMATO_DEFAULT,
    Formato,
    Fundo,
    normalize_formato,
    normalize_fundo,
)


class HistoricalObservationCreate(BaseModel):
    """Schema para insertar UNA observacion historica (dato REAL).

    Las features reales (DPC, %INDUS, P/BAYA, HA, DIA_COSECHA) son opcionales:
    cuando vienen, habilitan la descomposición exacta de error en el UI; si el
    Excel trae solo el formato mínimo (KG/HA + KG/JR_H), quedan en None.
    """

    fundo: Fundo = Field(..., alias="FUNDO")
    formato: Formato = Field(default=FORMATO_DEFAULT, alias="FORMATO")
    fecha: date = Field(..., alias="FECHA")
    kg_ha: float = Field(..., alias="KG/HA", gt=0.0, le=100_000.0)
    kg_jr_h: float = Field(..., alias="KG/JR_H", gt=0.0)
    dpc: float | None = Field(default=None, alias="DPC", ge=0.0, le=400.0)
    indus_pct: float | None = Field(default=None, alias="%INDUS", ge=0.0, le=100.0)
    p_baya: float | None = Field(default=None, alias="P/BAYA", gt=0.0, le=100.0)
    ha: float | None = Field(default=None, alias="HA", gt=0.0, le=10_000.0)
    dia_cosecha: int | None = Field(default=None, alias="DIA_COSECHA", ge=0, le=365)

    @field_validator("formato", mode="before")
    @classmethod
    def _validate_formato(cls, value: object) -> Formato:
        return normalize_formato(value)

    @field_validator("fundo", mode="before")
    @classmethod
    def _validate_fundo(cls, value: object) -> Fundo:
        if not isinstance(value, str):
            raise ValueError("FUNDO es obligatorio")
        return normalize_fundo(value)

    model_config = ConfigDict(populate_by_name=True)


class HistoricalObservationResponse(BaseModel):
    id: int
    variety: str
    fundo: str
    formato: str
    fecha: date
    kg_ha: float
    kg_jr_h: float
    dpc: float | None = None
    indus_pct: float | None = None
    p_baya: float | None = None
    ha: float | None = None
    dia_cosecha: int | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HistoricalObservationListResponse(BaseModel):
    items: list[HistoricalObservationResponse]
    total: int
    limit: int
    offset: int


class HistoryImportResponse(BaseModel):
    """Resumen del import desde Excel."""

    variety: str
    inserted: int
    skipped_invalid_rows: int
    message: str
