"""Entidades del dominio (mirror de los DTOs del backend).

Se modelan como Pydantic v2 BaseModel con `frozen=True` para mantener
la inmutabilidad que daba `@dataclass(frozen=True)` y, al mismo tiempo,
ganar parsing JSON → instancia con `Model.model_validate(d)` sin escribir
mappers manuales.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

DriftStatus = Literal["ok", "warning", "alert"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)


# El backend (`/varieties/{variety}`) devuelve las métricas con sus nombres
# CRUDOS de MLflow (nested_cv_*_mean, full_model_r2, ...), no `test_mae`/
# `test_r2`. Resolvemos sobre una lista de candidatos para que el dashboard
# muestre el valor real out-of-fold en vez de 0.
_MAE_KEYS: tuple[str, ...] = ("test_mae", "mae", "nested_cv_mae_mean")
_R2_KEYS: tuple[str, ...] = ("test_r2", "r2", "nested_cv_r2_mean", "full_model_r2")


def _resolve_metric(metrics: dict, candidates: tuple[str, ...]) -> float:
    """Primer valor numérico presente entre los nombres candidatos; 0.0 si ninguno."""
    for key in candidates:
        val = metrics.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 0.0


class VarietyViewModel(_Frozen):
    name: str
    model_loaded: bool = True
    metrics: dict = Field(default_factory=dict)
    version: int | None = None
    best_params: dict = Field(default_factory=dict)

    @computed_field  # type: ignore[misc]
    @property
    def mae(self) -> float:
        return _resolve_metric(self.metrics, _MAE_KEYS)

    @computed_field  # type: ignore[misc]
    @property
    def r2(self) -> float:
        return _resolve_metric(self.metrics, _R2_KEYS)

    @computed_field  # type: ignore[misc]
    @property
    def mape(self) -> float:
        return _resolve_metric(self.metrics, ("test_mape", "mape", "business_oof_mape"))

    @computed_field  # type: ignore[misc]
    @property
    def model_type(self) -> str:
        return str(self.best_params.get("model_type", "—"))


class PredictionResult(_Frozen):
    variety: str
    kghora: float
    kgjn: float | None = None
    inputs: dict = Field(default_factory=dict)
    # Drift report copiado del ForecastRecord subyacente. None cuando el
    # backend no expuso baseline para esa variedad.
    drift: DriftReport | None = None


class DriftPerFeature(_Frozen):
    """Drift de una sola feature en una sola fila (mirror del backend)."""

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

    @computed_field  # type: ignore[misc]
    @property
    def display_value(self) -> str:
        """Texto a mostrar en la tabla — formato adecuado por tipo."""
        if self.value_str is not None:
            return self.value_str
        if self.value is None:
            return "—"
        return f"{self.value:,.2f}"


class TrainingWindow(_Frozen):
    """Ventana temporal del baseline."""

    start: str = ""
    end: str = ""
    n_samples: int = 0


class DriftReport(_Frozen):
    """Reporte de drift adjunto a un ForecastRecord (puede ser None)."""

    score: float = 0.0
    status: DriftStatus = "ok"
    verdict: str = ""
    training_window: TrainingWindow = Field(default_factory=TrainingWindow)
    per_feature: tuple[DriftPerFeature, ...] = ()


class ForecastRecord(_Frozen):
    """Espejo de ForecastResponse del backend (snake_case)."""

    id: int
    variety: str
    fecha: str = ""
    external_id: str | None = None
    kg_ha: float = 0.0
    indus_pct: float | None = None
    dpc: float = 0.0
    p_baya: float | None = None
    ha: float = 0.0
    dia_cosecha: int = 0
    formato: str = "FRESCO"
    fundo: str = ""
    horas_efectivas: float | None = None
    kghora_pred: float = 0.0
    kgjn_pred: float | None = None
    created_at: str = ""
    updated_at: str = ""
    drift: DriftReport | None = None


class BatchFeatureDrift(_Frozen):
    """Drift de una feature a nivel de lote (mirror del backend)."""

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


class RowStatusCounts(_Frozen):
    ok: int = 0
    warning: int = 0
    alert: int = 0


class BatchDriftReport(_Frozen):
    """Reporte agregado de drift para un lote (mirror del backend)."""

    n_rows: int = 0
    score: float = 0.0
    status: DriftStatus = "ok"
    verdict: str = ""
    training_window: TrainingWindow = Field(default_factory=TrainingWindow)
    per_feature: tuple[BatchFeatureDrift, ...] = ()
    row_status_counts: RowStatusCounts = Field(default_factory=RowStatusCounts)


class ForecastListResult(_Frozen):
    items: tuple[ForecastRecord, ...] = ()
    total: int = 0
    limit: int = 0
    offset: int = 0
    batch_drift: BatchDriftReport | None = None


class HistoricalObservation(_Frozen):
    """Espejo de HistoricalObservationResponse del backend (dato REAL).

    Las features reales (dpc/indus_pct/p_baya/ha/dia_cosecha) son opcionales:
    cuando el Excel de reales las trae, habilitan la descomposición de error
    100% exacta; si no, el seguimiento usa solo KG/HA real.
    """

    id: int = 0
    variety: str = ""
    fundo: str = ""
    formato: str = ""
    fecha: str = ""
    kg_ha: float = 0.0
    kg_jr_h: float = 0.0
    dpc: float | None = None
    indus_pct: float | None = None
    p_baya: float | None = None
    ha: float | None = None
    dia_cosecha: int | None = None
    created_at: str = ""


class AccuracyPoint(_Frozen):
    """Pronóstico emparejado con su valor realizado + descomposición de error.

    Identidad exacta: `error_total = error_data + error_model`, donde
    `pred_on_real` es la predicción del modelo re-evaluada con el KG/HA REAL
    (resto de features = las del pronóstico). Así el `error_data` aísla el
    efecto de la mala proyección de KG/HA y el `error_model` queda como el
    residual del modelo dado el input correcto. `pred_on_real=None` cuando no
    se pudo recalcular (la UI degrada a solo error total).
    """

    variety: str
    fundo: str
    formato: str
    fecha: str
    pred_original: float
    real: float
    pred_on_real: float | None = None

    @computed_field  # type: ignore[misc]
    @property
    def error_total(self) -> float:
        return self.pred_original - self.real

    @computed_field  # type: ignore[misc]
    @property
    def error_model(self) -> float | None:
        if self.pred_on_real is None:
            return None
        return self.pred_on_real - self.real

    @computed_field  # type: ignore[misc]
    @property
    def error_data(self) -> float | None:
        if self.pred_on_real is None:
            return None
        return self.pred_original - self.pred_on_real

    @computed_field  # type: ignore[misc]
    @property
    def abs_pct_error(self) -> float | None:
        if self.real == 0:
            return None
        return abs(self.error_total) / self.real * 100.0


class WeekAggregate(_Frozen):
    """Cierre semanal (ISO): suma proyectada vs suma real de una semana."""

    week: str
    proj_sum: float = 0.0
    real_sum: float = 0.0
    n: int = 0

    @computed_field  # type: ignore[misc]
    @property
    def pct_diff(self) -> float | None:
        if self.real_sum == 0:
            return None
        return (self.proj_sum - self.real_sum) / self.real_sum * 100.0


class Catalogs(_Frozen):
    """Catálogos cerrados consumidos por formularios y validación."""

    formatos: tuple[str, ...] = ()
    formato_default: str = ""
    fundos: tuple[str, ...] = ()


class ServiceHealth(_Frozen):
    status: str
    mlflow_connected: bool = False
    database_connected: bool = False
    models_loaded: int = 0
    models_available: int = 0
    total_varieties: int = 0

    @computed_field  # type: ignore[misc]
    @property
    def is_healthy(self) -> bool:
        return self.status in ("healthy", "ok")


# Resuelve forward references: PredictionResult.drift apunta a DriftReport,
# definida más abajo en el archivo. Llamamos model_rebuild() para que Pydantic
# resuelva el string `"DriftReport | None"` ahora que la clase ya existe.
PredictionResult.model_rebuild()
