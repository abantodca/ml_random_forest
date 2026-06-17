"""Configuracion global del proyecto.

Centraliza rutas, esquema de datos, hiperparametros de CV y URI de MLflow.
Cualquier modulo debe leer constantes desde aqui en vez de hardcodearlas.

Backend MLflow:
    El proyecto SIEMPRE usa un MLflow server (Postgres + S3 detras).
    En local lo sirve `docker compose up` (servicio mlflow en :5000,
    backend Postgres + S3 real parametrizado via S3_MLFLOW_BUCKET).
    En produccion apuntas la misma env var `MLFLOW_TRACKING_URI` a tu
    server real (ECS Fargate detras de ALB). No hay backend file://mlruns
    ni sqlite local ni LocalStack (ADR-001 / ADR-003).

Variables de entorno reconocidas (todas opcionales, con fallback sano):
    MLFLOW_TRACKING_URI       : URI del tracking server. Default:
                                http://localhost:5000 (servicio Docker
                                expuesto al host). En el container del
                                trainer se sobreescribe a http://mlflow:5000
                                via docker-compose.yml.
    MLFLOW_EXPERIMENT_PREFIX  : prefijo de experimentos MLflow.
                                Default vacio -> el experimento es la variedad.
    MODEL_REGISTRY_PREFIX     : prefijo del Model Registry.
                                Default 'rnd-forest-'.
    REPORT_PLOTLY_OFFLINE     : 1 = embeber plotly.js gzip (default, autocontenido), 0 = CDN.

Esquema de modelado (decidido tras EDA):
    Target          : KG/JR_H (kg cosechados por jornal-hora)
    Numericas (raw) : KG/HA, %INDUS, DPC, P/BAYA, HA, DIA_COSECHA
    Categoricas     : FORMATO, FUNDO
    Date-derived    : ANIO, MES_SIN/COS (orden 1-3), SEMANA_SIN/COS,
                      TEMPORADA_ALTA/BAJA  (creadas en FeatureGenerator)
                      DIA_SEM_SIN/COS removidas (auditoria 2026-05-05: corr ~0).
    Structural      : KG_TOTAL, INDUS_KG_HA, KG_PER_BAYA, KG_HA_PER_DPC
                      (ratios intra-fila en FeatureGenerator)
    Lag features    : 35 cols rolling/seasonal/std/slope/ratios + tenure + cadencia
                      por (FUNDO+FORMATO, FUNDO, FORMATO) en step_03_features/
                      lag_features.py (ver LAG_OUTPUT_COLUMNS).

Excluidas por LEAKAGE (target = KG/JR / H-EF, demostrado con max_abs_diff = 0):
    KG/JR, H-EF

Excluidas por NULA INFORMACION (1 unico valor o MI = 0 en EDA):
    VARIEDAD, CALIBRADO, DIA_SEM
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto (resueltas desde la raiz)
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
LOGS_DIR: Path = BASE_DIR / "logs"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
REPORTS_DIR: Path = BASE_DIR / "reports"

# ---------------------------------------------------------------------------
# S3 — artifacts remotos (activo solo si S3_ARTIFACTS_BUCKET esta definido)
# ---------------------------------------------------------------------------
# Apunta SIEMPRE a un bucket S3 real (ADR-003: no usamos LocalStack).
# En local lo configurás vía .env (S3_ARTIFACTS_BUCKET=<tu-bucket>).
# En AWS Batch lo inyecta la job-def definida en GUIA_MLOPS_AWS.md #4.4.
# El upload ocurre al final de main.py si el bucket esta configurado;
# scripts/s3_sync.py es defensivo: si S3 falla, el training termina OK
# igual y los artefactos quedan en disco local del container.
S3_ARTIFACTS_BUCKET: str = os.environ.get("S3_ARTIFACTS_BUCKET", "")
S3_ARTIFACTS_PREFIX: str = os.environ.get("S3_ARTIFACTS_PREFIX", "ml-training")
S3_REPORTS_PREFIX: str = os.environ.get("S3_REPORTS_PREFIX", "ml-training/reports")


def init_dirs() -> None:
    """Crea los directorios de salida en disco. Idempotente.

    Se invoca explicitamente desde `main.py` / workers para evitar
    side-effects al importar `src.config` (un test que solo importe TARGET
    no debe crear `logs/`, `artifacts/`, etc.).
    """
    for d in (LOGS_DIR, ARTIFACTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Datos de entrada
# ---------------------------------------------------------------------------
ACCUMULATED_FILE: Path = DATA_DIR / "BD_HISTORICO_ACUMULADO.xlsx"
TRAINING_FILE: Path = DATA_DIR / "training" / "DB-HISTORICA.xlsx"
DEFAULT_VARIETIES: str = "POP"  # comma-separated; "all" expande a todas las hojas
MIN_ROWS_PER_VARIETY: int = 100  # umbral usado por scripts/prepare_data.py

# ---------------------------------------------------------------------------
# Esquema de modelado
# ---------------------------------------------------------------------------
TARGET: str = "KG/JR_H"

# Columnas numericas conservadas tal cual del Excel
NUMERIC_FEATURES: list[str] = ["KG/HA", "%INDUS", "DPC", "P/BAYA", "HA", "DIA_COSECHA"]

# Categoricas a one-hot
CATEGORICAL_FEATURES: list[str] = ["FORMATO", "FUNDO"]

# Columna de fecha (se transforma a derivadas ciclicas en FeatureGenerator)
DATE_COLUMN: str = "FECHA"

# Columnas que el data_loader debe traer del Excel (sin TARGET)
RAW_FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [DATE_COLUMN]

# Columnas a descartar explicitamente (leakage o nula informacion) si existieran
LEAKAGE_COLUMNS: list[str] = ["KG/JR", "H-EF"]
USELESS_COLUMNS: list[str] = ["VARIEDAD", "DIA_SEM", "MES"]

# Missing flags: columnas con missing significativo cuyo NaN es informativo.
# `MissingFlagger` agrega `<col>__MISS` antes del imputer para que el modelo
# reciba la senal de "esta fila tenia ese valor faltante". Decision basada
# en EDA POP (filas con P/BAYA NaN -> MAPE 17.3% vs 15.6% observadas;
# %INDUS similar). Si entrenas otra variedad con patrones distintos,
# pasar `cols=` explicito al constructor de `MissingFlagger` o ajustar aqui.
MISSING_FLAG_COLS: list[str] = ["%INDUS", "P/BAYA"]

# Skew mitigation: thresholds para auto-deteccion en FeatureGenerator.fit.
# Reemplazo las listas hardcoded por variedad (eran fragiles cuando se entrena
# variedades nuevas con distribuciones distintas). FeatureGenerator decide
# por columna si log1p / sqrt aplica, basado en skew y kurtosis del fit data.
#
# Politica de transformacion (additive: no reemplaza la columna raw):
#   - kurt > SKEW_KURT_THRESHOLD       -> agrega <col>_SQRT
#   - |skew| > SKEW_THRESHOLD          -> agrega <col>_LOG1P
#   - else                              -> nada (distribucion sana)
#
# El shift por columna se memoiza en fit -> transform usa el mismo shift.
# Sin esto, la misma fila podia dar valores distintos en train vs inference
# cuando los rangos diferian (bug latente de la version anterior).
#
# SKEW_AUTO_DETECT permite desactivar para comparar contra baseline sin
# transformaciones (A/B test informativo).
SKEW_AUTO_DETECT: bool = True
SKEW_THRESHOLD: float = 1.5      # |skew| above this -> log1p
SKEW_KURT_THRESHOLD: float = 50.0  # kurtosis above this -> sqrt (mas agresivo)

# ---------------------------------------------------------------------------
# EDA thresholds (diagnostics/*.py)
# ---------------------------------------------------------------------------
# Reportar findings de skew/kurt en EDA. Independientes de SKEW_THRESHOLD
# y SKEW_KURT_THRESHOLD (que rigen transformacion en FeatureGenerator):
# aqui solo se trata de avisar al humano en el reporte de auditoria.
EDA_KURT_WARN: float = 5.0          # kurt > 5 -> finding "medium"
EDA_KURT_HIGH: float = 10.0         # kurt > 10 -> escala el finding a "high"
EDA_SKEW_HIGH: float = 3.0          # |skew| > 3 -> escala el finding a "high"

# Fraccion de outliers IQR sobre n_total que dispara warning en EDA.
OUTLIER_FRACTION_WARN: float = 0.05

# Threshold para considerar dos numericas "muy correlacionadas". Se usa en
# diagnostics/multivariate.py:correlation_matrix y en eda.py al llamarla.
CORRELATION_HIGH_THRESHOLD: float = 0.85

# Cardinalidad de variables categoricas (diagnostics/categorical.py).
#   CARDINALITY_HIGH : por encima de esto, NO se calcula chi2/Cramer's V
#                      (tabla de contingencia se vuelve poco confiable).
#   CARDINALITY_WARN : aviso para considerar target-encoding / agrupar.
CARDINALITY_HIGH: int = 200
CARDINALITY_WARN: int = 50

# Bandas de interpretacion de V de Cramer (asociacion categorica-target).
#   < CRAMERS_V_WEAK   : asociacion debil, candidata a drop / agrupar.
#   >= CRAMERS_V_STRONG: asociacion fuerte, target-encoding util.
CRAMERS_V_WEAK: float = 0.05
CRAMERS_V_STRONG: float = 0.3

# ---------------------------------------------------------------------------
# Hiperparametros de CV y tuning
# ---------------------------------------------------------------------------
RANDOM_STATE: int = int(os.environ.get("SEED", "42"))
# El pipeline siempre entrena TODOS los backends del registry (XGB + LGB
# hoy) cada uno con su Optuna study independiente, y `champion.select_champion`
# elige el mejor por variedad usando lex-order (gap -> full_mape -> tiempo).
# Si en el futuro se agrega un nuevo backend al BACKEND_REGISTRY, queda
# incluido automaticamente.

# ---------------------------------------------------------------------------
# Quality gates del campeon (umbrales minimos para considerar un modelo util)
# ---------------------------------------------------------------------------
# Un campeon que no supere estos umbrales se considera inutilizable y NO
# se registra ni promueve en MLflow Registry. Los logs y artefactos se
# guardan igual para auditoria.
#
# CHAMPION_MAX_MAPE: MAPE OOF maximo aceptable (out-of-fold, honesto).
#   Valor en % (ej: 25.0 = 25%). Si el campeon supera este umbral,
#   el modelo no se promueve. Comparado contra `champion.oof_mape`
#   (cada fila predicha por un modelo que NO la vio en train).
#   25% es ~8pp arriba del MAPE_oof observado en POP (~17%); deja
#   holgura para variedades mas dificiles pero filtra modelos rotos.
#   Antes era 30% comparado contra full_mape (in-sample, optimista).
# CHAMPION_MAX_GAP: brecha maxima Train-Test aceptable (overfitting).
#   Valor en % (18.0 = 18pp de diferencia entre MAE_train y MAE_test).
#   Subido de 15 -> 18 tras evidencia empirica: con search spaces rev. 7.1
#   (LGB) y rev. 6 (XGB) -- capacidad capada y regularizacion estricta
#   forzada -- el suelo realista del gap para POP (10k filas, 16 estratos
#   FUNDO_FORMATO, target con cola larga) ronda 0.13-0.18pp. 15pp era
#   conservador sin restricciones de capacidad, y rechazaba modelos que
#   ya combatieron el overfit pero rebotan contra el techo del dataset.
#   Un overfit "real" (Optuna gaming el search space) deja gaps de 20+pp,
#   que este threshold sigue rechazando correctamente.
CHAMPION_MAX_MAPE: float = float(os.environ.get("CHAMPION_MAX_MAPE", "25.0"))
CHAMPION_MAX_GAP: float = float(os.environ.get("CHAMPION_MAX_GAP", "18.0"))
# Gate de gap RELATIVO (fix multi-variedad 2026-06-11, hallazgo revision
# experta): CHAMPION_MAX_GAP multiplica kilos por 100 y lo llama "pp" — para
# una variedad con target chico (CLAMSHELL mu=2.5) el mismo umbral es 2x mas
# laxo que para GRANEL (mu=5.4). El gate del campeon ahora usa
# gap_rel = (MAE_test - MAE_train) / MAE_test (adimensional, comparable
# entre variedades). 0.40 ~ "el MAE de train no puede ser menos del 60% del
# de test". El absoluto queda como tag informativo. Para POP el campeon
# actual da 0.30 -> pasa igual que antes (sin cambio de decision).
CHAMPION_MAX_GAP_REL: float = float(os.environ.get("CHAMPION_MAX_GAP_REL", "0.40"))

# ---------------------------------------------------------------------------
# Decision lex-order del champion (champion.select_champion)
# ---------------------------------------------------------------------------
# Estos umbrales gobiernan el desempate entre modelos (XGB vs LGB) en
# `select_champion`. Centralizados aqui para tunear sin tocar codigo.
#
# Revision 2026-06-10: el lex-order anterior era |gap| -> full_mape -> tiempo.
# Eso optimizaba "poco overfitting" como objetivo primario (premia subajuste)
# y usaba una metrica IN-SAMPLE (full_mape) como desempate. El orden nuevo:
#   1. Gate de gap: |gap|*100 <= CHAMPION_MAX_GAP (restriccion, no objetivo).
#   2. Menor MAPE OOF de negocio (generalizacion honesta).
#   3. Menor tiempo de entrenamiento ante empate practico.

# Tolerancia sobre MAPE OOF (en %). Dos modelos cuyo MAPE_oof difiere en
# menos de esto se consideran empate de rendimiento -> desempata por tiempo.
# 0.5 pp de MAPE es ruido tipico entre seeds distintas.
OOF_MAPE_TIE_TOLERANCE: float = 0.5


# ---------------------------------------------------------------------------
# Tuning profiles (presupuesto de Optuna: cuantos trials, cuantos folds).
# NO confundir con entornos (local vs aws): el tuning es ortogonal al entorno.
#
# Tiempo estimado por variedad (~10k filas) ENTRENANDO LOS 3 BACKENDS
# (gpb + lgb + xgb). gpb domina ~60% del wall-time: su componente GP
# (efectos aleatorios) no paraleliza. Medido en dev/POP 2026-06-15;
# prod/prod_xl escalados por nº de trials (~4.7x / ~9.3x dev):
#   smoke   : ~3-5 min    (humo; 5 trials, NO registra)
#   dev     : ~75-80 min  (gpb ~50m + lgb ~12m + xgb ~18m)
#   prod    : ~5-7 h      (modelo a promover; 60 trials, 5 outer folds)
#   prod_xl : ~11-15 h    (overnight; 100 trials, 6 outer folds)
TUNING_PROFILES: dict[str, dict[str, int]] = {
    "smoke": {
        "n_trials": 5,
        "final_trials": 3,
        "outer_folds": 2,
        "inner_folds": 2,
    },
    "dev": {
        "n_trials": 20,
        "final_trials": 10,
        "outer_folds": 3,
        "inner_folds": 3,
    },
    "prod": {
        "n_trials": 60,
        "final_trials": 30,
        "outer_folds": 5,
        "inner_folds": 3,
    },
    "prod_xl": {
        "n_trials": 100,
        "final_trials": 50,
        "outer_folds": 6,
        "inner_folds": 3,
    },
}
# Override de presupuesto POR BACKEND (fraccion del perfil de tuning, aplicado
# a n_trials/final_trials/outer_folds; inner_folds queda intacto). gpb es el
# backend mas lento (componente GP no paraleliza) y NUNCA gana el campeonato:
# pierde por MAPE OOF incluso a presupuesto JUSTO (validado en
# scripts/experiments/lgb_vs_gpb_control.py y en 2 corridas dev). Se corre como
# backend de REFERENCIA a presupuesto reducido para no inflar el wall-time
# (~3.5-4h gpb full en prod -> ~2h). lgb/xgb (candidatos reales) van completos.
# Vaciar el dict ({}) para volver a paridad total entre los 3 backends.
BACKEND_BUDGET_FRACTION: dict[str, float] = {"gpb": 0.5}

DEFAULT_TUNING: str = "dev"

# Defaults de CV (usados por tuning.py cuando el caller no override-a)
OUTER_CV_FOLDS: int = TUNING_PROFILES[DEFAULT_TUNING]["outer_folds"]
INNER_CV_FOLDS: int = TUNING_PROFILES[DEFAULT_TUNING]["inner_folds"]

# ---------------------------------------------------------------------------
# Early stopping interno de los regresores (step_04_train/early_stopping.py)
# ---------------------------------------------------------------------------
# Cada fit (inner CV, outer refit, ensemble final) carva un holdout interno
# del train y corta los arboles cuando la metrica (MAE en espacio log1p)
# deja de mejorar EARLY_STOPPING_ROUNDS rondas. Esto reemplaza el cap de
# n_estimators<=1000 del search space: el numero de arboles ya no se tunea,
# se fija alto (N_ESTIMATORS_MAX) y el early stopping decide el corte por
# fold/trial. Permite reabrir la grilla (capacidad) sin que arboles extra
# memoricen el train.
EARLY_STOPPING_ROUNDS: int = int(os.environ.get("EARLY_STOPPING_ROUNDS", "50"))
# Fraccion del train usada como holdout interno para early stopping.
EARLY_STOPPING_VAL_FRACTION: float = 0.1
# Bajo este n de filas NO se hace early stopping (holdout demasiado chico
# para ser senal; caso tipico: folds internos del perfil smoke).
EARLY_STOPPING_MIN_ROWS: int = 200
# Techo de arboles cuando early stopping esta activo (el corte real lo
# decide el holdout; esto es solo un fusible).
N_ESTIMATORS_MAX: int = 2000

# Refit final: K pipelines en folds del KFold; predict promedia las K.
# Reduce varianza del modelo de produccion (~5-10%) a costa de +(K-1)x
# tiempo del refit final, que es despreciable vs el nested CV.
# K=1 = legacy (refit unico sobre todo el dataset).
OOF_ENSEMBLE_K: int = 5

# Penalizacion por varianza entre inner folds en el objective de Optuna
# (robustez del tuning, 2026-06-13): score = mean(MAE) + lambda * std(MAE).
# Con lambda>0, TPE prefiere configs ESTABLES sobre configs con buen
# promedio pero alta dispersion entre folds (que generalizan peor).
# Default 0.0 = comportamiento historico bit-identico (solo media).
# Valores razonables para activar: 0.5-1.0.
OPTUNA_OBJECTIVE_STD_PENALTY: float = float(
    os.environ.get("OPTUNA_OBJECTIVE_STD_PENALTY", "0.0")
)

# Sample weights por densidad del target (compute_sample_weights).
# Centralizado aqui para tunear sin tocar codigo. n_bins=10 fija el valor
# que el caller usaba de facto (antes hardcoded en tuning.py overrideando
# el default 20 de la funcion); cap=5.0 alinea con el default historico.
SAMPLE_WEIGHT_BINS: int = 10
SAMPLE_WEIGHT_CAP: float = 5.0

# Peso extra para filas de TEMPORADA ALTA (autopsia OOF 2026-06-11: el peor
# 5% de errores esta 2-3x sobre-representado en ago-oct — picos de cosecha
# sub-predichos). Boost multiplicativo a los meses listados; se combina con
# los demas pesos y se renormaliza a media=1. Default OFF hasta A/B.
# (no usa _env_bool: se define mas abajo en este modulo)
SAMPLE_WEIGHT_HIGH_SEASON: bool = bool(int(os.environ.get("SAMPLE_WEIGHT_HIGH_SEASON", "0")))
# Meses pico POR VARIEDAD: el default (8,9,10) es el pico de POP. Otras
# variedades tienen calendarios distintos — override con env var CSV, p.ej.
# SAMPLE_WEIGHT_HIGH_SEASON_MONTHS="11,12,1". Al escalar a multi-variedad
# en un mismo run, esto necesitara un override por variedad (no global).
SAMPLE_WEIGHT_HIGH_SEASON_MONTHS: tuple = tuple(
    int(m) for m in os.environ.get(
        "SAMPLE_WEIGHT_HIGH_SEASON_MONTHS", "8,9,10"
    ).split(",")
)
SAMPLE_WEIGHT_HIGH_SEASON_BOOST: float = float(
    os.environ.get("SAMPLE_WEIGHT_HIGH_SEASON_BOOST", "1.5")
)

# Pesos adicionales ∝ 1/y (compute_inv_target_weights): alinean la loss MAE
# con el MAPE de negocio dando mas peso a filas de target bajo, donde el
# MAPE OOF es peor (quintil bajo 22% vs alto 14%, diagnostico 2026-06-10).
# Se MULTIPLICAN con los pesos por bins y se renormalizan a media=1.
# Default OFF hasta validar via A/B (Fase B.4 del plan 2026-06-11).
SAMPLE_WEIGHT_INV_Y: bool = bool(int(os.environ.get("SAMPLE_WEIGHT_INV_Y", "0")))
SAMPLE_WEIGHT_INV_Y_CAP: float = float(os.environ.get("SAMPLE_WEIGHT_INV_Y_CAP", "5.0"))

# ---------------------------------------------------------------------------
# Feature flags para ABLATION (env-var driven, default = legacy)
# ---------------------------------------------------------------------------
# Cada cambio del plan FE 2026-05-09 se activa selectivamente. Default
# todos OFF -> comportamiento equivalente al modelo LGB v3 (MAPE_oof
# 14.86%, gap 0.138) baseline. Activar uno a uno (smoke train ~1min) y
# comparar MAPE_oof + gap para ver cual aporta.
#
# Ejemplo:
#   docker compose run --rm \
#     -e ENABLE_OUTLIER_CASCADE_FF=1 \
#     -e CV_OUTER_STRATEGY=temporal_year \
#     trainer --varieties POP --tuning smoke


def _env_bool(name: str, default: bool = False) -> bool:
    """Lee bool de env var: '1', 'true', 'yes' (case-insensitive) -> True."""
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


# A — OutlierCapper: bounds por (FUNDO, FORMATO) con cascade fallback.
#     Justificacion: 86% del data es FORMATO=GRANEL y 72% FUNDO=A9; bounds
#     globales o solo-por-FUNDO los reflejan a ellos y cortaban grupos
#     chicos (CLAMSHELL 11 OZ con target μ=2.54 vs 5.35 GRANEL).
#     False (default) = group_col="FUNDO" legacy.
ENABLE_OUTLIER_CASCADE_FF: bool = _env_bool("ENABLE_OUTLIER_CASCADE_FF", False)

# D — Lags simples shift(1)/shift(2) + diff(1) por (FUNDO, FORMATO).
#     Justificacion: PACF de POP muestra lag 1=0.50, lag 2=0.33 (los mas
#     fuertes). Las rolling medians 7/14/30/90 ya existentes pueden
#     suavizar esa senal puntual.
#     Riesgo: alta correlacion con KG_JR_H_lag_FF_7 -> ruido.
ENABLE_SIMPLE_LAGS: bool = _env_bool("ENABLE_SIMPLE_LAGS", False)

# F — FUNDO_FORMATO interaction como dummies (15-18 cols).
#     Validado en prod_xl POP (2026-05-09) vs LGB v3 baseline:
#       - gap promedio mejora -0.011 (-8%): 0.138 -> 0.127.
#       - MAE_test marginal +0.004 (dentro del std=0.016, ruido).
#       - biz_MAPE_oof marginal +0.07pp (ruido vs baseline 14.86%).
#       - std gap +0.021 (mas inestable, fold 4 outlier con gap=0.225).
#     Cambio default OFF -> ON: el gap MEJORA real (8%) justifica el cambio
#     a pesar de la mayor varianza. La interaccion FUNDO_FORMATO es senal
#     legitima (V=0.26 vs target). Se mantiene como flag por si alguna
#     variedad futura no se beneficie (V<0.10 -> override con env=0).
ENABLE_FUNDO_FORMATO_INTERACTION: bool = _env_bool(
    "ENABLE_FUNDO_FORMATO_INTERACTION", True,
)

# H — LOF ANTES del OutlierCapper (Fase B.5 del plan 2026-06-11).
#     Hoy el LOF puntua data YA capeada: los extremos que el capper recorto
#     no aparecen en el score (señal diluida). Con la flag, el orden pasa a
#     imputer -> outlier_score -> outliers: el LOF ve los extremos reales y
#     el capper sigue protegiendo al modelo despues. lof_score no esta en
#     NUMERIC_FEATURES, asi que el capper no lo toca en ningun orden.
#     Default OFF hasta validar via A/B.
ENABLE_LOF_BEFORE_CAPPER: bool = _env_bool("ENABLE_LOF_BEFORE_CAPPER", False)

# I — Eliminar la version RAW cuando existe la skew-mitigada (Fase B.5).
#     Hoy FeatureGenerator AGREGA log1p/sqrt y conserva la raw -> tres
#     versiones correlacionadas de la misma columna (raw+log1p o raw+sqrt),
#     ruido para los splits y para Optuna. Con la flag, la raw sale del
#     passthrough cuando su transformada existe (los arboles son invariantes
#     a transformaciones monotonas: no se pierde informacion de ranking).
#     Default OFF hasta validar via A/B.
SKEW_DROP_RAW: bool = _env_bool("SKEW_DROP_RAW", False)

# K — Paquete FE 2026-06-11 (analisis ACF/PACF + auditoria de colas).
#     Todos default OFF hasta validar via A/B. Evidencia que los motiva:
#       - ACF/PACF intra-FF del target: PACF lag1=+0.71, lag2=+0.37 (los
#         simple lags ENABLE_SIMPLE_LAGS capturan esto; encender en el A/B).
#       - Derivadas con colas brutales SIN proteccion: ratio_FF_30 kurt=383
#         max=55x, slope kurt=393, days_since kurt=836 (vs DPC raw kurt=162
#         que motivo el capper).
#       - P/BAYA: 39.2% NaN imputado con MEDIANA GLOBAL pese a ACF=0.74 y
#         drift por estrato.
#       - Estacionalidad: corr +0.48 a 1 anio (ya capturada) y +0.32 a 2
#         anios (no capturada).
#
# LAG_LOG_DERIVED: log1p en ratios (positivos por construccion), signed-log1p
#     en slope y log1p en days_since/std. Estateless y monotona: misma fila
#     -> mismo valor en train/inference; el sentinel -1 de cold-start queda
#     fuera del rango de log1p(x>=0) y sigue distinguible.
LAG_LOG_DERIVED: bool = _env_bool("LAG_LOG_DERIVED", False)
# IMPUTER_GROUP_MEDIAN: las columnas que caen al fallback de mediana (>30%
#     missing, hoy P/BAYA) usan mediana jerarquica (FUNDO,FORMATO) -> FUNDO
#     -> global en vez de mediana global unica (mismo patron cascade que
#     OutlierCapper).
IMPUTER_GROUP_MEDIAN: bool = _env_bool("IMPUTER_GROUP_MEDIAN", False)
# ENABLE_FEATURE_LAGS: medianas rolling por FF de P/BAYA y DPC (ventanas
#     7/30). Justificacion: ACF lag1 = 0.74 / 0.61; ademas el lag de P/BAYA
#     actua como mejor imputacion implicita de su 39% NaN.
ENABLE_FEATURE_LAGS: bool = _env_bool("ENABLE_FEATURE_LAGS", False)
# ENABLE_TARGET_VOLATILITY: KG_JR_H_std_FF_30 (std rolling shift(1) del
#     TARGET). Hoy solo existe la de KG/HA; "cuan predecible es el grupo"
#     es la senal que falta en el quintil bajo.
ENABLE_TARGET_VOLATILITY: bool = _env_bool("ENABLE_TARGET_VOLATILITY", False)
# ENABLE_SEASONAL_2Y: lag estacional a 730d +/-15d (alternancia bienal).
ENABLE_SEASONAL_2Y: bool = _env_bool("ENABLE_SEASONAL_2Y", False)
# ENABLE_CALENDAR_EXTRA: armonico 2 de SEMANA + frequency encoding de
#     FUNDO/FORMATO (n de obs del grupo en train, normalizado).
ENABLE_CALENDAR_EXTRA: bool = _env_bool("ENABLE_CALENDAR_EXTRA", False)

# L — Modo EX-ANTE (experimento #11, 2026-06-13): mide el MAPE de forecast
#     VERDADERO. El modelo actual es nowcasting: KG/HA, %INDUS y sus
#     derivadas comparten el evento de cosecha con el target (el 14.27% OOF
#     esta condicionado a conocer la cosecha del dia). Con la flag activa:
#       1. ConcurrentFeatureDropper elimina del pipeline las features que
#          usan el valor del DIA del evento (KG/HA, %INDUS, KG_TOTAL,
#          INDUS_KG_HA, KG_PER_BAYA, KG_HA_PER_DPC, ratios actual-vs-lag,
#          REL_GLOBAL y sus variantes skew) — quedan lags, calendario,
#          categoricas y derivadas lag-vs-lag.
#       2. Los rolling lags y std/slope se computan sobre la serie DIARIA
#          (1 punto por grupo+dia) en vez de por fila: el shift(1)
#          posicional incluia filas hermanas del MISMO dia (leakage del
#          evento en forecast; valido solo en nowcasting).
#     Default OFF = comportamiento nowcasting actual. Solo experimento:
#     no cambia la API ni el modelo registrado.
EXANTE_MODE: bool = _env_bool("EXANTE_MODE", False)

# J — Reporte dual de CV (Fase A.2 del plan 2026-06-11): cuando el outer CV
#     es stratified, al final del nested CV se corre un chequeo temporal
#     (TemporalYearSplit, fit con los best_params por fold expanding) y se
#     loggean `temporal_mape_oof` / `temporal_r2_oof` / `temporal_mae_test_mean`
#     junto a las metricas stratified. Costo: DUAL_CV_FOLDS fits extra SIN
#     tuning (~1-3 min). Asi cada run reporta interpolacion (stratified) y
#     forecast honesto (temporal) sin discusion de "cual es el numero real".
DUAL_CV_REPORT: bool = _env_bool("DUAL_CV_REPORT", True)
DUAL_CV_FOLDS: int = int(os.environ.get("DUAL_CV_FOLDS", "3"))

# G — CV outer strategy.
#   "stratified"     : StratifiedKFold por FUNDO_FORMATO (DEFAULT — decision
#     de producto 2026-06-11: metricas comparables con el historico del
#     proyecto). Mide interpolacion dentro de los anios ya vistos.
#   "temporal_year"  : TemporalYearSplit expanding-window por ANIO (override
#     con CV_OUTER_STRATEGY=temporal_year). Mide el error REAL de predecir un
#     anio futuro bajo drift (PSI hasta 2.09 entre anios): con datos 2022-2026,
#     MAPE OOF temporal ~29.6% y R2 ~0.27 (A/B 2026-06-11) vs ~16-17% / ~0.8
#     del stratified — la diferencia es year-leakage del stratified, no un
#     defecto del split temporal. Usarlo como chequeo de honestidad antes de
#     deploy. Nota: filas de anios warmup quedan con OOF NaN;
#     business_validation y residuals ya las enmascaran.
# Inner CV siempre stratified (Optuna trial scope: equilibrio por estrato).
CV_OUTER_STRATEGY: str = os.environ.get("CV_OUTER_STRATEGY", "stratified")
TEMPORAL_CV_MIN_TRAIN_YEARS: int = int(os.environ.get("TEMPORAL_CV_MIN_TRAIN_YEARS", "2"))

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
# El proyecto SIEMPRE corre contra un MLflow server (Postgres backend +
# S3 artifact-root). En local lo provee `docker compose up` (servicio
# `mlflow`). En produccion apuntas la misma env var a tu server real.
#
# Default = http://localhost:5000 = el servicio Docker expuesto al host
# (asi corren scripts que ejecutan en el host, ej. utilidades manuales).
# DENTRO del container del trainer se sobreescribe a http://mlflow:5000
# via docker-compose.yml.
#
# El `artifact_location` lo decide el server (modo proxy: --serve-artifacts
# + --artifacts-destination s3://<bucket>/artifacts; los runs quedan como
# mlflow-artifacts:/...). El client NO debe pasarlo: si lo hiciera rompe el
# modelo (apuntaria a un path local del cliente que el server no puede leer).
MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Prefijo del nombre de experimento. El experimento final por variedad sera
# `f"{MLFLOW_EXPERIMENT_PREFIX}{variety}"`.
# Default vacio: el nombre del experimento es la VARIEDAD (e.g. "POP").
# Esto crea UN experimento INDEPENDIENTE por variedad, y cada training
# es un run versionado (e.g. "xgb_v3") dentro de ese experimento.
MLFLOW_EXPERIMENT_PREFIX: str = os.environ.get("MLFLOW_EXPERIMENT_PREFIX", "")

# Prefijo del Model Registry (registered model = `f"{prefix}{variety}"`).
# Cada training de la misma variedad genera una nueva VERSION del mismo
# registered model. ADR-001 garantiza que SIEMPRE corremos contra un MLflow
# server con backend SQL (Postgres en local + AWS), por lo que el Registry
# esta disponible incondicionalmente.
MODEL_REGISTRY_PREFIX: str = os.environ.get("MODEL_REGISTRY_PREFIX", "rnd-forest-")

# Guard de registro (incidente 2026-06-13): un run dev con EXANTE_MODE=1
# paso el quality gate (20.8% < 25%) y registro su campeon experimental
# como rnd-forest-POP v2 — la API sirve siempre la ULTIMA version, asi que
# un experimento degradado habria reemplazado al campeon real en serving.
# REGISTER_ENABLED=0 bloquea todo registro sin tocar el resto del run
# (metricas/reportes/artifacts intactos). Ademas, variety_runner bloquea
# automaticamente el registro cuando hay flags EXPERIMENTALES activos
# (hoy: EXANTE_MODE) aunque este guard quede en 1.
REGISTER_ENABLED: bool = _env_bool("REGISTER_ENABLED", True)

# ---------------------------------------------------------------------------
# Branding del reporte gerencial
# ---------------------------------------------------------------------------
REPORT_PROJECT_NAME: str = "Pronostico de productividad de cosecha (POP)"
REPORT_BUSINESS_UNIT: str = "Operaciones Agricolas"
REPORT_TARGET_LABEL: str = "kg por jornal-hora"

# Umbrales semaforo del reporte (sobre R2 mean del nested CV)
REPORT_R2_GOOD: float = 0.85  # >= verde / "Excelente"
REPORT_R2_WARN: float = 0.70  # >= amarillo / "Aceptable", < rojo / "Insuficiente"

# Targets gerenciales que se renderizan como gauges en el HTML.
# Mover la aguja por encima/debajo de estos valores cambia el color del gauge.
REPORT_R2_TARGET: float = 0.90  # R2 gerencial (KG/JR OOF) que el negocio quiere superar
REPORT_MAE_TARGET: float = (
    0.20  # MAE del modelo (KG/JR_H Test CV) que el negocio quiere NO superar
)

# Descripcion en lenguaje natural del modelo. Aparece en el hero del
# dashboard ejecutivo para que un lector no-tecnico entienda en 1 frase
# que predice el modelo, en que unidad y para que sirve.
REPORT_MODEL_DESCRIPTION: str = (
    "Predice la productividad por jornal (kilogramos cosechados por "
    "jornada de trabajo) usando datos historicos de cosecha, formato del "
    "producto, fundo y fechas. Permite anticipar el rendimiento esperado "
    "para planificar logistica, equipos y compromisos comerciales."
)

# Veredicto ejecutivo: combina MAPE de negocio (sobre data total) y
# brecha train-test (overfitting) para clasificar el modelo en 4 niveles.
# Cada nivel mapea a un semaforo + recomendacion accionable.
#
# Lectura: el modelo cae en el nivel mas conservador donde AMBAS metricas
# entren. Ej.: MAPE=12% (nivel 1 OK) pero gap=0.30 (nivel 4 OUT) -> nivel 3.
REPORT_VERDICT_THRESHOLDS: dict = {
    "alta_confianza": {"max_mape_pct": 15.0, "max_abs_gap": 0.10},
    "confianza_aceptable": {"max_mape_pct": 22.0, "max_abs_gap": 0.18},
    "confianza_limitada": {"max_mape_pct": 35.0, "max_abs_gap": 0.30},
    # peor que confianza_limitada -> "no_recomendado"
}

# Subgroup MAPE multiplier sobre el global a partir del cual un FORMATO/FUNDO
# se marca como "problematico" en la seccion de Acciones Recomendadas.
REPORT_SUBGROUP_WARN_RATIO: float = 1.5  # >= 1.5x el MAPE global = warning

# Tamano minimo de un subgrupo (FORMATO o FUNDO) para que cuente como
# candidato a "problematico". Mas chico = ruido puro.
REPORT_SUBGROUP_MIN_N: int = 10

# Umbrales de tarjetas KPI ejecutivas (lenguaje natural). Cambiar aqui mueve
# tanto el HTML como el Excel sin tocar codigo.
KPI_PRECISION_HIGH_MAPE_PCT: float = 15.0  # MAPE <= 15 -> ALTO
KPI_PRECISION_MEDIUM_MAPE_PCT: float = 25.0  # MAPE <= 25 -> MEDIO, sino BAJO
KPI_R2_HIGH_PCT: float = 80.0  # R2*100 >= 80 -> ALTO
KPI_R2_MEDIUM_PCT: float = 60.0  # >= 60 -> MEDIO, sino BAJO
KPI_BASELINE_HIGH_IMPROVEMENT_PCT: float = 50.0
KPI_BASELINE_MEDIUM_IMPROVEMENT_PCT: float = 25.0

# Umbrales para acciones recomendadas auto-generadas.
ABS_GAP_WARN: float = 0.20  # |gap| > 0.20 = "memorizo entrenamiento"
FULL_MAPE_CRITICAL_PCT: float = 25.0  # MAPE > 25% = critico

# Group-rare en data_loader: categorias con n<RARE_MIN_COUNT se colapsan en
# 'OTROS'. Solo se aplica a las columnas listadas en RARE_GROUP_COLS.
RARE_MIN_COUNT: int = 50
RARE_GROUP_COLS: list[str] = ["FORMATO"]

# Estado semaforo (VERDE/AMARILLO/ROJO) en la hoja Resumen del Excel:
#   R2 OOF >= REPORT_R2_TARGET            -> VERDE
#   R2 OOF >= REPORT_R2_AMBER_THRESHOLD   -> AMARILLO, sino ROJO
#   MAE   <= REPORT_MAE_TARGET            -> VERDE
#   MAE   <= REPORT_MAE_TARGET * REPORT_MAE_AMBER_RATIO -> AMARILLO, sino ROJO
REPORT_R2_AMBER_THRESHOLD: float = 0.70
REPORT_MAE_AMBER_RATIO: float = 2.0

# Modo de carga de plotly.js en el HTML:
#   True  = embebido gzip+base64 (+~1.9MB al HTML, autocontenido: se puede
#           enviar por correo o abrir sin internet; navegadores sin
#           DecompressionStream caen solos al CDN).
#   False = CDN — HTML liviano pero los charts requieren internet.
# Override desde env: REPORT_PLOTLY_OFFLINE=1 (embebido) o =0 (CDN).
# Default vuelto a embebido (2026-06-12): el costo que motivo el CDN era el
# bundle plano de ~4.8MB; con gzip cuesta 1.9MB y el Winner queda portable
# para gerencia (el caso de uso real: compartir el archivo, no servirlo).
REPORT_PLOTLY_OFFLINE: bool = os.environ.get("REPORT_PLOTLY_OFFLINE", "1") != "0"

# ---------------------------------------------------------------------------
# === HISTORIAL ===
# ---------------------------------------------------------------------------
# Notas historicas / decisiones de diseño preservadas como contexto. NO
# afectan la ejecucion: ningun codigo lee de aqui.
#
# CatBoost (2026-05-05): evaluado y eliminado del BACKEND_REGISTRY. No
#     aportaba en POP frente a XGB/LGB; mismo patron que GAMM Phase 0.
#     Si se reincorpora a futuro hay que reagregarlo al registry y al
#     pipeline de tuning.
