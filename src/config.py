"""Configuracion global: rutas, esquema de datos, CV/tuning, gates y MLflow.

Ningun modulo debe hardcodear estas constantes; se leen siempre desde aqui.

Backend MLflow: SIEMPRE un server con Postgres + S3 (ADR-001 / ADR-003).
No hay file://mlruns, sqlite ni LocalStack. En local lo sirve `docker compose up`;
en produccion se apunta `MLFLOW_TRACKING_URI` al server real.

Esquema de modelado (decidido tras EDA):
    Target       : KG/JR_H (kg por jornal-hora)
    Numericas    : KG/HA, %INDUS, DPC, P/BAYA, HA, DIA_COSECHA
    Categoricas  : FORMATO, FUNDO
    Derivadas    : ciclicas de FECHA + ratios intra-fila (FeatureGenerator)
    Lags         : 35 cols por (FUNDO+FORMATO, FUNDO, FORMATO) (lag_features.py)
    Excluidas    : KG/JR y H-EF por leakage (target = KG/JR / H-EF);
                   VARIEDAD, CALIBRADO, DIA_SEM por nula informacion.

El razonamiento largo de cada umbral vive en README.md, docs/adr/ y
docs/05-auditoria-ml-2026-08-07.md, no en este archivo.
"""

import os
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    """Lee bool de env var: '1', 'true', 'yes', 'on' -> True."""
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
LOGS_DIR: Path = BASE_DIR / "logs"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
REPORTS_DIR: Path = BASE_DIR / "reports"

S3_ARTIFACTS_BUCKET: str = os.environ.get("S3_ARTIFACTS_BUCKET", "")
S3_ARTIFACTS_PREFIX: str = os.environ.get("S3_ARTIFACTS_PREFIX", "ml-training")
S3_REPORTS_PREFIX: str = os.environ.get("S3_REPORTS_PREFIX", "ml-training/reports")


def init_dirs() -> None:
    """Crea los directorios de salida. Idempotente.

    Explicito (no al importar) para que un test que solo lea TARGET no cree
    `logs/`, `artifacts/`, etc.
    """
    for d in (LOGS_DIR, ARTIFACTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


ACCUMULATED_FILE: Path = DATA_DIR / "BD_HISTORICO_ACUMULADO.xlsx"
TRAINING_FILE: Path = DATA_DIR / "training" / "DB-HISTORICA.xlsx"
DEFAULT_VARIETIES: str = "POP"
MIN_ROWS_PER_VARIETY: int = 100

TARGET: str = "KG/JR_H"

NUMERIC_FEATURES: list[str] = ["KG/HA", "%INDUS", "DPC", "P/BAYA", "HA", "DIA_COSECHA"]

_EXTRA_CATEGORICALS: list[str] = ["CALIBRE", "TIPO DE COSECHA"]
_ENABLE_EXTRA_CATEGORICALS: bool = os.environ.get(
    "ENABLE_EXTRA_CATEGORICALS", ""
).strip().lower() in ("1", "true", "yes", "on")
CATEGORICAL_FEATURES: list[str] = ["FORMATO", "FUNDO"] + (
    _EXTRA_CATEGORICALS if _ENABLE_EXTRA_CATEGORICALS else []
)

DATE_COLUMN: str = "FECHA"
RAW_FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [DATE_COLUMN]
LEAKAGE_COLUMNS: list[str] = ["KG/JR", "H-EF"]
USELESS_COLUMNS: list[str] = ["VARIEDAD", "DIA_SEM", "MES"]

MISSING_FLAG_COLS: list[str] = ["%INDUS", "P/BAYA"]

SEASON_AUTODETECT: bool = bool(int(os.environ.get("SEASON_AUTODETECT", "1")))
SEASON_HIGH_PCTL: float = float(os.environ.get("SEASON_HIGH_PCTL", "66"))
SEASON_LOW_PCTL: float = float(os.environ.get("SEASON_LOW_PCTL", "33"))
SEASON_MIN_MONTH_OBS: int = int(os.environ.get("SEASON_MIN_MONTH_OBS", "5"))

SKEW_AUTO_DETECT: bool = True
SKEW_THRESHOLD: float = 1.5
SKEW_KURT_THRESHOLD: float = 50.0

EDA_KURT_WARN: float = 5.0
EDA_KURT_HIGH: float = 10.0
EDA_SKEW_HIGH: float = 3.0
OUTLIER_FRACTION_WARN: float = 0.05
CORRELATION_HIGH_THRESHOLD: float = 0.85
CARDINALITY_HIGH: int = 200
CARDINALITY_WARN: int = 50
CRAMERS_V_WEAK: float = 0.05
CRAMERS_V_STRONG: float = 0.3

RANDOM_STATE: int = int(os.environ.get("SEED", "42"))

MAPE_MIN_DENOM: float = float(os.environ.get("MAPE_MIN_DENOM", "1.0"))

CHAMPION_MAX_MAPE: float = float(os.environ.get("CHAMPION_MAX_MAPE", "25.0"))
CHAMPION_MAX_GAP: float = float(os.environ.get("CHAMPION_MAX_GAP", "18.0"))
CHAMPION_MAX_GAP_REL: float = float(os.environ.get("CHAMPION_MAX_GAP_REL", "0.40"))
CHAMPION_MIN_BASELINE_SKILL: float = float(os.environ.get("CHAMPION_MIN_BASELINE_SKILL", "0.05"))

CHAMPION_WARN_TEMPORAL_MAPE: float = float(os.environ.get("CHAMPION_WARN_TEMPORAL_MAPE", "30.0"))
CHAMPION_WARN_TEMPORAL_R2: float = float(os.environ.get("CHAMPION_WARN_TEMPORAL_R2", "0.20"))

REQUIRE_TEMPORAL_GATE: bool = _env_bool("REQUIRE_TEMPORAL_GATE", False)
MIN_TEMPORAL_FOLDS: int = int(os.environ.get("MIN_TEMPORAL_FOLDS", "2"))
CHAMPION_MAX_TEMPORAL_MAPE: float = float(os.environ.get("CHAMPION_MAX_TEMPORAL_MAPE", "35.0"))
CHAMPION_MIN_TEMPORAL_BASELINE_SKILL: float = float(
    os.environ.get("CHAMPION_MIN_TEMPORAL_BASELINE_SKILL", "0.0")
)

OOF_MAPE_TIE_TOLERANCE: float = 0.5

ADAPT_FOLDS_TO_N: bool = bool(int(os.environ.get("ADAPT_FOLDS_TO_N", "1")))
ADAPT_FOLDS_ROWS_PER_OUTER: int = int(os.environ.get("ADAPT_FOLDS_ROWS_PER_OUTER", "120"))
ADAPT_FOLDS_ROWS_PER_INNER: int = int(os.environ.get("ADAPT_FOLDS_ROWS_PER_INNER", "150"))

ENABLE_PRUNER: bool = bool(int(os.environ.get("ENABLE_PRUNER", "1")))
PRUNER_STARTUP_TRIALS: int = int(os.environ.get("PRUNER_STARTUP_TRIALS", "8"))
PRUNER_WARMUP_STEPS: int = int(os.environ.get("PRUNER_WARMUP_STEPS", "1"))

OPTUNA_STORAGE_URL: str = os.environ.get("OPTUNA_STORAGE_URL", "")

TUNING_PROFILES: dict[str, dict[str, int]] = {
    "smoke": {"n_trials": 5, "final_trials": 3, "outer_folds": 2, "inner_folds": 2},
    "dev": {"n_trials": 20, "final_trials": 10, "outer_folds": 3, "inner_folds": 3},
    "prod": {"n_trials": 60, "final_trials": 30, "outer_folds": 5, "inner_folds": 3},
    "prod_xl": {"n_trials": 100, "final_trials": 50, "outer_folds": 6, "inner_folds": 3},
}
BACKEND_BUDGET_FRACTION: dict[str, float] = {}

DEFAULT_TUNING: str = "dev"

OUTER_CV_FOLDS: int = TUNING_PROFILES[DEFAULT_TUNING]["outer_folds"]
INNER_CV_FOLDS: int = TUNING_PROFILES[DEFAULT_TUNING]["inner_folds"]

EARLY_STOPPING_ROUNDS: int = int(os.environ.get("EARLY_STOPPING_ROUNDS", "50"))
EARLY_STOPPING_VAL_FRACTION: float = 0.1
EARLY_STOPPING_MIN_ROWS: int = int(os.environ.get("EARLY_STOPPING_MIN_ROWS", "60"))
EARLY_STOPPING_MIN_VAL: int = int(os.environ.get("EARLY_STOPPING_MIN_VAL", "12"))
EARLY_STOPPING_REFIT_FULL: bool = _env_bool("EARLY_STOPPING_REFIT_FULL", False)
N_ESTIMATORS_MAX: int = 1200

OOF_ENSEMBLE_K: int = 5

OPTUNA_OBJECTIVE_STD_PENALTY: float = float(os.environ.get("OPTUNA_OBJECTIVE_STD_PENALTY", "0.0"))
OPTUNA_OBJECTIVE_GAP_PENALTY: float = float(os.environ.get("OPTUNA_OBJECTIVE_GAP_PENALTY", "0.0"))

SAMPLE_WEIGHT_BINS: int = 10
SAMPLE_WEIGHT_CAP: float = 5.0

OPTUNA_OBJECTIVE_METRIC: str = os.environ.get("OPTUNA_OBJECTIVE_METRIC", "mae").strip().lower()

SAMPLE_WEIGHT_HIGH_SEASON: bool = bool(int(os.environ.get("SAMPLE_WEIGHT_HIGH_SEASON", "0")))
SAMPLE_WEIGHT_HIGH_SEASON_MONTHS: tuple = tuple(
    int(m) for m in os.environ.get("SAMPLE_WEIGHT_HIGH_SEASON_MONTHS", "8,9,10").split(",")
)
SAMPLE_WEIGHT_HIGH_SEASON_BOOST: float = float(
    os.environ.get("SAMPLE_WEIGHT_HIGH_SEASON_BOOST", "1.5")
)

SAMPLE_WEIGHT_INV_Y: bool = bool(int(os.environ.get("SAMPLE_WEIGHT_INV_Y", "0")))
SAMPLE_WEIGHT_INV_Y_CAP: float = float(os.environ.get("SAMPLE_WEIGHT_INV_Y_CAP", "5.0"))


ENABLE_OUTLIER_CASCADE_FF: bool = _env_bool("ENABLE_OUTLIER_CASCADE_FF", False)

ENABLE_SIMPLE_LAGS: bool = _env_bool("ENABLE_SIMPLE_LAGS", False)

ENABLE_FUNDO_FORMATO_INTERACTION: bool = _env_bool(
    "ENABLE_FUNDO_FORMATO_INTERACTION",
    True,
)

ENABLE_LOF_BEFORE_CAPPER: bool = _env_bool("ENABLE_LOF_BEFORE_CAPPER", False)

SKEW_DROP_RAW: bool = _env_bool("SKEW_DROP_RAW", False)

LAG_LOG_DERIVED: bool = _env_bool("LAG_LOG_DERIVED", False)
IMPUTER_GROUP_MEDIAN: bool = _env_bool("IMPUTER_GROUP_MEDIAN", False)
ENABLE_FEATURE_LAGS: bool = _env_bool("ENABLE_FEATURE_LAGS", False)
ENABLE_TARGET_VOLATILITY: bool = _env_bool("ENABLE_TARGET_VOLATILITY", False)
ENABLE_SEASONAL_2Y: bool = _env_bool("ENABLE_SEASONAL_2Y", False)
ENABLE_CALENDAR_EXTRA: bool = _env_bool("ENABLE_CALENDAR_EXTRA", False)

EXANTE_MODE: bool = _env_bool("EXANTE_MODE", False)

DUAL_CV_REPORT: bool = _env_bool("DUAL_CV_REPORT", True)
DUAL_CV_FOLDS: int = int(os.environ.get("DUAL_CV_FOLDS", "3"))
TEMPORAL_MAPE_REL_FLOOR: float = float(os.environ.get("TEMPORAL_MAPE_REL_FLOOR", "0.05"))

CV_OUTER_STRATEGY: str = os.environ.get("CV_OUTER_STRATEGY", "stratified")
TEMPORAL_CV_MIN_TRAIN_YEARS: int = int(os.environ.get("TEMPORAL_CV_MIN_TRAIN_YEARS", "2"))
TEMPORAL_CV_GAP_PERIODS: int = int(os.environ.get("TEMPORAL_CV_GAP_PERIODS", "1"))

MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_PREFIX: str = os.environ.get("MLFLOW_EXPERIMENT_PREFIX", "")
MODEL_REGISTRY_PREFIX: str = os.environ.get("MODEL_REGISTRY_PREFIX", "rnd-forest-")

WARM_START_FROM_REGISTRY: bool = bool(int(os.environ.get("WARM_START_FROM_REGISTRY", "1")))

REGISTER_ENABLED: bool = _env_bool("REGISTER_ENABLED", True)

REPORT_PROJECT_NAME: str = "Pronostico de productividad de cosecha (POP)"
REPORT_BUSINESS_UNIT: str = "Operaciones Agricolas"

REPORT_R2_TARGET: float = 0.90
REPORT_MAE_TARGET: float = 0.20

REPORT_MODEL_DESCRIPTION: str = (
    "Predice la productividad por jornal (kilogramos cosechados por "
    "jornada de trabajo) usando datos historicos de cosecha, formato del "
    "producto, fundo y fechas. Permite anticipar el rendimiento esperado "
    "para planificar logistica, equipos y compromisos comerciales."
)

REPORT_VERDICT_THRESHOLDS: dict = {
    "alta_confianza": {"max_mape_pct": 15.0, "max_abs_gap": 0.10},
    "confianza_aceptable": {"max_mape_pct": 22.0, "max_abs_gap": 0.18},
    "confianza_limitada": {"max_mape_pct": 35.0, "max_abs_gap": 0.30},
}

REPORT_SUBGROUP_WARN_RATIO: float = 1.5
REPORT_SUBGROUP_MIN_N: int = 10

KPI_PRECISION_HIGH_MAPE_PCT: float = 15.0
KPI_PRECISION_MEDIUM_MAPE_PCT: float = 25.0
KPI_R2_HIGH_PCT: float = 80.0
KPI_R2_MEDIUM_PCT: float = 60.0
KPI_BASELINE_HIGH_IMPROVEMENT_PCT: float = 50.0
KPI_BASELINE_MEDIUM_IMPROVEMENT_PCT: float = 25.0

ABS_GAP_WARN: float = 0.20
FULL_MAPE_CRITICAL_PCT: float = 25.0

RARE_MIN_COUNT: int = 50
RARE_GROUP_COLS: list[str] = ["FORMATO"]
ADAPT_RARE_MIN_COUNT: bool = bool(int(os.environ.get("ADAPT_RARE_MIN_COUNT", "1")))
RARE_MIN_COUNT_FRAC: float = float(os.environ.get("RARE_MIN_COUNT_FRAC", "0.03"))
RARE_MIN_COUNT_FLOOR: int = int(os.environ.get("RARE_MIN_COUNT_FLOOR", "15"))

REPORT_R2_AMBER_THRESHOLD: float = 0.70
REPORT_MAE_AMBER_RATIO: float = 2.0

REPORT_PLOTLY_OFFLINE: bool = os.environ.get("REPORT_PLOTLY_OFFLINE", "1") != "0"
