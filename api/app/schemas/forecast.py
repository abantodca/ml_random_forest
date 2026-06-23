"""
Schemas de Pydantic para Forecasts
===================================
Define los modelos de validacion de datos para requests y responses.

Las columnas de input al modelo MLflow estan dictadas por la signature
del pipeline serializado en `ml_training`. El backend solo envia
las columnas RAW (FECHA, KG/HA, %INDUS, DPC, P/BAYA, HA, DIA_COSECHA,
FORMATO, FUNDO); el `LagFeatureTransformer` que vive dentro del pipeline
calcula los 31 lag features en transform usando el historial memorizado
en fit.

`HORAS_EFECTIVAS` y `HECTA` NO van al modelo: se usan despues para
KGJN_PRED = KGHORA_PRED * HORAS_EFECTIVAS y como metadato del request.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core import (
    FORMATO_DEFAULT,
    Formato,
    Fundo,
    normalize_formato,
    normalize_fundo,
)

DriftStatus = Literal["ok", "warning", "alert"]


# ============================================================================
# Request Schemas (Input)
# ============================================================================


class ForecastCreate(BaseModel):
    """Schema para crear un pronostico individual."""

    fecha: date = Field(..., alias="FECHA", description="Fecha del pronostico (YYYY-MM-DD)")
    external_id: str | None = Field(
        default=None,
        alias="EXTERNAL_ID",
        max_length=100,
        description="ID externo opcional para identificar pronosticos",
    )
    kg_ha: float = Field(..., alias="KG/HA", gt=0.0, description="Kilogramos por hectarea")
    indus_pct: float | None = Field(
        default=None,
        alias="%INDUS",
        ge=0.0,
        le=100.0,
        description="Porcentaje industrial (opcional)",
    )
    dpc: float = Field(..., alias="DPC", description="Dias post cuaja")
    p_baya: float | None = Field(
        default=None,
        alias="P/BAYA",
        gt=0.0,
        description="Peso de baya (opcional)",
    )
    ha: float = Field(..., alias="HA", gt=0.0, description="Hectareas")
    dia_cosecha: int = Field(
        ..., alias="DIA_COSECHA", ge=0, description="Dia dentro de la temporada de cosecha"
    )
    formato: Formato = Field(
        default=FORMATO_DEFAULT,
        alias="FORMATO",
        description="Formato del producto (catálogo cerrado)",
    )
    fundo: Fundo = Field(
        ...,
        alias="FUNDO",
        description="Fundo / parcela (catálogo cerrado)",
    )
    horas_efectivas: float | None = Field(
        default=None,
        alias="HORAS_EFECTIVAS",
        ge=0,
        description="Horas efectivas trabajadas (opcional, para KGJN_PRED)",
    )

    @field_validator("formato", mode="before")
    @classmethod
    def _validate_formato(cls, v: object) -> Formato:
        if v is None or v == "":
            return FORMATO_DEFAULT
        return normalize_formato(v)

    @field_validator("fundo", mode="before")
    @classmethod
    def _validate_fundo(cls, v: object) -> Fundo:
        if not isinstance(v, str):
            raise ValueError("FUNDO es obligatorio")
        return normalize_fundo(v)

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "FECHA": "2026-04-10",
                    "EXTERNAL_ID": "FECV",
                    "KG/HA": 5000.0,
                    "%INDUS": 5.0,
                    "DPC": 120.0,
                    "P/BAYA": 2.5,
                    "HA": 10.0,
                    "DIA_COSECHA": 30,
                    "FORMATO": FORMATO_DEFAULT,
                    "FUNDO": "C5",
                    "HORAS_EFECTIVAS": 8.0,
                }
            ]
        },
    )


class ForecastBatchCreate(BaseModel):
    """Schema para crear multiples pronosticos."""

    forecasts: list[ForecastCreate] = Field(..., min_length=1, max_length=1000)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "forecasts": [
                        {
                            "FECHA": "2026-04-10",
                            "EXTERNAL_ID": "FECV-1",
                            "KG/HA": 5000.0,
                            "%INDUS": 5.0,
                            "DPC": 120.0,
                            "P/BAYA": 2.5,
                            "HA": 10.0,
                            "DIA_COSECHA": 30,
                            "FORMATO": FORMATO_DEFAULT,
                            "FUNDO": "C5",
                        },
                    ]
                }
            ]
        }
    )


class ForecastUpdate(BaseModel):
    """Schema para actualizar campos de un pronostico existente."""

    fecha: date | None = Field(default=None, alias="FECHA")
    external_id: str | None = Field(default=None, alias="EXTERNAL_ID")
    kg_ha: float | None = Field(default=None, alias="KG/HA", gt=0.0)
    indus_pct: float | None = Field(default=None, alias="%INDUS", ge=0.0, le=100.0)
    dpc: float | None = Field(default=None, alias="DPC")
    p_baya: float | None = Field(default=None, alias="P/BAYA", gt=0.0)
    ha: float | None = Field(default=None, alias="HA", gt=0.0)
    dia_cosecha: int | None = Field(default=None, alias="DIA_COSECHA", ge=0)
    formato: Formato | None = Field(default=None, alias="FORMATO")
    fundo: Fundo | None = Field(default=None, alias="FUNDO")
    horas_efectivas: float | None = Field(default=None, alias="HORAS_EFECTIVAS", ge=0)

    @field_validator("formato", mode="before")
    @classmethod
    def _validate_formato(cls, v: object) -> Formato | None:
        if v is None:
            return None
        return normalize_formato(v)

    @field_validator("fundo", mode="before")
    @classmethod
    def _validate_fundo(cls, v: object) -> Fundo | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("FUNDO debe ser texto")
        return normalize_fundo(v)

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"examples": [{"EXTERNAL_ID": "FECV-UPDATED", "HORAS_EFECTIVAS": 9.0}]},
    )


# ============================================================================
# Response Schemas (Output)
# ============================================================================


class DriftPerFeature(BaseModel):
    """Drift de una sola feature en una sola fila.

    Campos numéricos y categóricos comparten el mismo schema (con valores
    None para los campos no aplicables a su tipo) para que el frontend
    pueda renderizarlas en una sola tabla sin discriminar.
    """

    feature: str
    value: float | None = None
    value_str: str | None = None
    baseline_median: float | None = None
    baseline_iqr: float | None = None
    baseline_p05: float | None = None
    baseline_p95: float | None = None
    baseline_freq: float | None = None
    z_score: float | None = None
    is_unseen_category: bool = False
    status: DriftStatus = "ok"
    source: str = "history"


class TrainingWindow(BaseModel):
    """Ventana temporal del baseline (rango y tamaño).

    Usamos `start`/`end` (en lugar de `from`/`to`) porque `from` es
    palabra reservada en Python y obligaría a tener alias por todos
    lados (campo `from_` con alias='from'); el JSON queda más limpio
    con nombres directos.
    """

    start: str = ""
    end: str = ""
    n_samples: int = 0


class DriftReport(BaseModel):
    """Reporte de drift adjunto a una predicción individual."""

    score: float = 0.0
    status: DriftStatus = "ok"
    verdict: str = ""
    training_window: TrainingWindow = Field(default_factory=TrainingWindow)
    per_feature: list[DriftPerFeature] = Field(default_factory=list)


class ForecastResponse(BaseModel):
    """Schema de respuesta con todos los datos del pronostico."""

    id: int
    variety: str
    fecha: date
    external_id: str | None
    kg_ha: float
    indus_pct: float | None
    dpc: float
    p_baya: float | None
    ha: float
    dia_cosecha: int
    formato: Formato
    fundo: Fundo
    horas_efectivas: float | None
    kghora_pred: float
    kgjn_pred: float | None
    created_at: datetime
    updated_at: datetime
    # Metadata opcional, no persistida en DB. Se inyecta en el router
    # después de model_validate(forecast). Default None preserva la
    # respuesta legacy para clientes que no entienden el campo.
    drift: DriftReport | None = None
    # Incertidumbre del ensemble (std entre los K=5 pipelines internos del
    # OOFEnsembleRegressor). No persistida. None con modelos legacy.
    # kghora_lo/hi = banda ±1.96·std; confidence por std relativa:
    # alta (<10%), media (10-20%), baja (>20% -> revisar manualmente).
    kghora_std: float | None = None
    kghora_lo: float | None = None
    kghora_hi: float | None = None
    confidence: Literal["alta", "media", "baja"] | None = None

    model_config = ConfigDict(from_attributes=True)


class PredictionResponse(BaseModel):
    """Resultado de una predicción dry-run (NO se persiste en la DB).

    Usado por `POST /forecasts/{variety}/predict` para casos donde el cliente
    necesita el KGHORA estimado sin crear un registro: predicción exploratoria
    en el UI y la descomposición de error (re-predecir sobre inputs reales).
    """

    variety: str
    kghora_pred: float
    kgjn_pred: float | None = None
    drift: DriftReport | None = None
    # Banda de confianza del ensemble (ver ForecastResponse).
    kghora_std: float | None = None
    kghora_lo: float | None = None
    kghora_hi: float | None = None
    confidence: Literal["alta", "media", "baja"] | None = None


class BatchFeatureDrift(BaseModel):
    """Drift de una feature a nivel de lote (PSI + K-S + Chi²).

    Los campos numéricos y categóricos comparten el schema con `None`
    en los que no aplican (Chi² no aplica a numéricas, K-S no aplica
    a categóricas).
    """

    feature: str
    kind: Literal["numeric", "categorical"]
    psi: float = 0.0
    psi_status: DriftStatus = "ok"
    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    chi2_statistic: float | None = None
    chi2_pvalue: float | None = None
    method: str = "psi"
    status: DriftStatus = "ok"
    n_baseline: int | None = None
    n_observed: int = 0
    source: str = "history"
    unseen_categories: int = 0


class RowStatusCounts(BaseModel):
    """Conteo de filas por estado de drift individual."""

    ok: int = 0
    warning: int = 0
    alert: int = 0


class BatchDriftReport(BaseModel):
    """Reporte agregado de drift para un lote de pronósticos.

    Solo se calcula cuando el lote tiene ≥30 filas (umbral mínimo para
    que PSI/K-S/Chi² sean confiables). Por debajo, queda en `None` y el
    frontend usa solo el drift por fila.
    """

    n_rows: int = 0
    score: float = 0.0
    status: DriftStatus = "ok"
    verdict: str = ""
    training_window: TrainingWindow = Field(default_factory=TrainingWindow)
    per_feature: list[BatchFeatureDrift] = Field(default_factory=list)
    row_status_counts: RowStatusCounts = Field(default_factory=RowStatusCounts)


class ForecastListResponse(BaseModel):
    """Schema de respuesta para listas paginadas de pronosticos."""

    items: list[ForecastResponse]
    total: int
    limit: int
    offset: int
    # Drift agregado del lote: presente solo en endpoints batch/upload
    # con N>=30. None en endpoints de lectura (GET /forecasts).
    batch_drift: BatchDriftReport | None = None


class DeletedCountResponse(BaseModel):
    """Schema de respuesta para operaciones de eliminacion masiva."""

    deleted: int
    message: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"deleted": 5, "message": "5 pronosticos eliminados"}]}
    )
