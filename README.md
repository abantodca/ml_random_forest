# ml_training

Pipeline modular de regresión para pronosticar productividad de cosecha
(`KG/JR_H` — kg por jornal-hora) — soporta **múltiples variedades** (en serie
o en paralelo) y **múltiples backends** (XGB y LGB) con selección automática
del campeón por variedad.

Implementado con **scikit-learn** (Pipeline + transformers `BaseEstimator`),
**Optuna** (TPE bayesiano multivariado) sobre **Nested CV**, **LightGBM** y
**XGBoost** como backends entrenados en paralelo, **MLflow** (local o
servidor remoto) para tracking + Model Registry, y un **dashboard HTML
ejecutivo** autocontenido por variedad.

Es un **proyecto único** que cubre el ciclo completo de MLOps: **entrenamiento**
(este pipeline), **API** de inferencia (`api/` — FastAPI que sirve los modelos
`rnd-forest-*` y persiste pronósticos) y **UI** gerencial (`ui/` — Streamlit que
consume la API). Mismo `docker-compose` en local y misma infra Terraform en
producción (ECS Fargate + ALB).

**Entorno:** Docker (Windows / Linux / Mac). El stack local levanta MLflow
server (Postgres + S3 real) + nginx para reports HTML + la API + la UI. El mismo
compose se traslada a producción cambiando endpoints (RDS, ALB). `Taskfile.yml`
orquesta build, up, data split, training, logs y auditoría. El backend MLflow
SIEMPRE es Postgres + S3 — no se usa `file://mlruns` ni sqlite ni LocalStack
(ADR-001 / ADR-003).

> **Nota sobre el nombre del repo:** se llama `ml_random_forest` por razones
> históricas. El pipeline actual entrena **XGBoost + LightGBM** (no Random
> Forest) y selecciona el mejor por variedad.

---

## Quick start (local)

```bash
# 0. Una sola vez tras clonar: bajar la data raw desde S3 (no vive en git).
#    El bucket por default es el mismo que S3_MLFLOW_BUCKET de tu .env.
aws s3 cp s3://<tu-bucket>/data/BD_HISTORICO_ACUMULADO.xlsx data/

# 1. Día a día
task data:split                                       # genera DB-HISTORICA.xlsx (corre en container)
task build                                            # build (trainer+api+ui) + arranca TODO el stack
task train VARIETIES=POP TUNING=smoke                 # smoke (~1-2 min, XGB+LGB, elige campeón)
task train VARIETIES=POP,JUPITER TUNING=dev           # multi-variety
task train VARIETIES=all TUNING=prod PARALLEL=4       # todas en paralelo (~2h cada una)
task up                                               # levanta db+mlflow+reports+api+ui (sin rebuild)
task logs                                             # tail trainer + mlflow
task down                                             # detiene servicios (preserva data)
```

URLs locales (con servicios `up`):

- **UI (Streamlit)**: <http://localhost:8501>
- **API (Swagger)**: <http://localhost:8000/docs>
- MLflow UI: <http://localhost:5000>
- Reports HTML: <http://localhost:8080/reports/>
- Artifacts: <http://localhost:8080/artifacts/>

> Flujo típico: `task build` (levanta todo) → `task train VARIETIES=POP TUNING=smoke`
> (registra el modelo `rnd-forest-POP`) → abrir la UI en :8501 (la variedad
> aparece disponible y se pueden pedir pronósticos). La API carga modelos
> **lazy** (`MLFLOW_PRELOAD_MODELS=false`): el primero se carga al primer pedido.

## Stack full: API (FastAPI) + UI (Streamlit)

Un solo proyecto, una sola fuente de verdad de código ML: el `src/` raíz lo usan
tanto el **trainer** (`main.py`) como la imagen de la **API** (se copia en build
con contexto = raíz del repo). Sin copias vendidas que se desincronicen.

| Pieza | Carpeta | Rol | Puerto |
|---|---|---|---|
| Trainer | `src/`, `main.py` | entrena y registra modelos `rnd-forest-*` | — |
| **API** | `api/` | FastAPI: sirve modelos de MLflow + persiste pronósticos | 8000 |
| **UI** | `ui/` | Streamlit: dashboard gerencial que consume la API | 8501 |

**Base de datos de pronósticos.** La API usa una base `forecasts` separada de la
de MLflow. En local viven ambas en el mismo Postgres (initdb crea `forecasts`).
En producción **reusa el RDS de MLflow**: la API auto-crea la base `forecasts`
en su primer arranque (`api/app/models/database.ensure_database`, idempotente).

**Producción — ruteo en el mismo ALB** (path-based, sin colisiones):

| Path | Servicio |
|---|---|
| `/app/*` | UI (Streamlit, `STREAMLIT_SERVER_BASE_URL_PATH=app`) |
| `/api/health*`, `/api/forecasts*`, `/api/varieties*`, `/api/history*` | API |
| `/docs`, `/openapi.json`, `/redoc` | API (Swagger) |
| `/reports/*`, `/artifacts/*` | Reports (nginx) |
| `/` (default) | MLflow UI |

> La API se rutea por **prefijos específicos**, no por `/api/*` genérico: MLflow
> con `--serve-artifacts` expone `/api/2.0/mlflow-artifacts/*` y un comodín
> robaría esa ruta. La UI llama a la API por service discovery interno
> (`http://api.ml-training.local:8000`), no por el ALB.

### Capacidad y costo (análisis)

Las tasks Fargate están dimensionadas para el caso real y son **escalables por
tfvars** sin tocar código (`api_cpu`, `api_memory`, `ui_cpu`, `ui_memory`,
`api_preload_models` en el `envs/prod/terraform.tfvars` que se construye desde
`GUIA_MLOPS_AWS_V2.md` #3.2.4):

| Servicio | vCPU / RAM | Por qué |
|---|---|---|
| API | 1 vCPU / 2 GB | Carga modelos en RAM. Con lazy-load (`preload=false`) y ~6 variedades sobra. Subir `api_memory` a 4096 si se activa preload de muchas variedades; `api_cpu` a 2048 para más concurrencia. |
| UI | 0.5 vCPU / 1 GB | Streamlit es liviano (render server-side). |

Costo Fargate incremental de añadir API + UI (us-east-1, ~$0.0405/vCPU-h +
~$0.00445/GB-h):

| Escenario | API+UI / mes (aprox.) |
|---|---|
| **Con scheduler on/off** (L/Mi/V 8–12 PET ≈ 87 h/mes) | **~$6–9** |
| 24/7 (sin apagar) | ~$53 |

El **scheduler** (Lambda + EventBridge) enciende/apaga RDS + MLflow + Reports +
**API + UI** en bloque (`task wake` / `task sleep`), así que el costo real sigue
el patrón on/off existente. No hay instancia extra que gestionar: la API reusa
el RDS de MLflow.

## CLI directa (sin Taskfile, sin Docker — requiere venv + MLflow server)

```bash
# Siempre entrena XGB y LGB y `select_champion` elige el ganador
# (ADR-002: el sistema decide, no hay flag para forzar un modelo).
python main.py --tuning dev --varieties POP

# multi-variety (cache se limpia entre variedades)
python main.py --tuning prod --varieties POP,JUPITER,VENTURA

# todas las hojas presentes en data/training/DB-HISTORICA.xlsx
python main.py --tuning prod --varieties all

# multi-variety EN PARALELO (procesos independientes; auto-ajusta inner_cv_n_jobs)
python main.py --tuning prod --varieties all --parallel-varieties 4

# overrides finos del presupuesto Optuna
python main.py --tuning prod --n-trials 120 --final-trials 60
python main.py --tuning prod --skip-final-tuning           # ahorra ~1/(outer+1)

# Model Registry — control fino sobre la version registrada
python main.py --tuning prod --varieties POP --no-register                 # no toca el Registry
python main.py --tuning prod --varieties POP --registry-stage Staging      # registra y promueve
# Para Production, se recomienda usar el workflow `promote.yml` con gates
# de calidad (GUIA_MLOPS_AWS_V2.md #12) en vez del flag.
```

### Tuning profiles (`src/config.py`)

| `--tuning` | n_trials | final_trials | outer × inner | trials totales | tiempo estimado* |
|---|---|---|---|---|---|
| `smoke`   | 5   | 3  | 2 × 2 | 13   | ~1 min      |
| `dev`     | 20  | 10 | 3 × 3 | 70   | ~20 min     |
| `prod`    | 60  | 30 | 5 × 3 | 330  | ~1.5-2.5 h  |
| `prod_xl` | 100 | 50 | 6 × 3 | 650  | ~5-6 h      |

\* sobre 10 073 filas, **POR BACKEND**. El pipeline entrena XGB + LGB siempre,
así que el wallclock real es ~2× los valores arriba. Con `--parallel-varieties N`
el wallclock por job se acerca a `total / N` si hay cores suficientes.

### Selección de campeón (`src/step_05_evaluate/champion.py`)

Cuando se entrenan XGB y LGB para la misma variedad, `select_champion`
decide con un **lex-order estricto** (no un score combinado):

1. **Gate de overfitting** — `|gap|*100 <= CHAMPION_MAX_GAP` como **restricción**
   (descalifica modelos rotos). El gap NO se minimiza: minimizarlo premiaba al
   modelo más subajustado, no al que mejor predice.
2. **Generalización** — menor **MAPE OOF** de negocio (cada fila predicha por un
   modelo que no la vio en train; la métrica honesta de producción).
3. **Eficiencia** — menor `elapsed_seconds` ante empate técnico.

Empates "blandos" en MAPE OOF usan `OOF_MAPE_TIE_TOLERANCE=0.5pp` para que ruido
de CV no haga inestable la decisión. El `full_mape` (in-sample) se conserva como
métrica informativa en dashboards pero no participa en la decisión. La justificación
textual de por qué ganó el campeón se persiste en `champion_summary["justification"]`.

`composite_score` aún se computa y loguea como tag de MLflow para compatibilidad
con dashboards históricos, pero **no** participa en la decisión.

### Salidas

Versionadas por run (`xgb_v3`, `lgb_v7`, …) para conservar historial sin
sobrescribir entre re-entrenamientos:

- `artifacts/final_pipeline_<variety>_<run_name>.joblib` — pipeline (preprocesador + `OOFEnsembleRegressor`) listo para `joblib.load(...)` y `predict(...)`.
- `artifacts/run_summary_<variety>_<run_name>.json` — métricas, hiperparámetros, paths, duración.
- `artifacts/best_params_<variety>_<run_name>.json` — hiperparámetros sin truncado de MLflow.
- `artifacts/run_summary_AGGREGATE.json` — resumen multi-variety: campeones, fallos, tiempo total.
- `reports/Winner_<variety>.html` — dashboard ejecutivo autocontenido (solo para el campeón).
- `reports/Winner_<variety>.xlsx` — Excel multi-hoja con métricas y predicciones (solo campeón).
- MLflow server (Docker, `task up`) — runs versionados, métricas, params, Model Registry. UI en `http://localhost:5000`. Backend Postgres (`pg-data` volume) + S3 real (bucket `S3_MLFLOW_BUCKET` del `.env`).
- `logs/pipeline_run.log`, `logs/variety_<variety>.log`, `logs/business_audit.jsonl`.

---

## Mapa de la arquitectura

```
ml_training/
├── main.py                                 # Entrypoint thin (parse + delega en orchestration/)
├── Taskfile.yml                            # Tareas locales: build / up / train / logs / audit / clean
├── docker-compose.yml                       # postgres + mlflow custom + nginx + trainer
├── Dockerfile                               # imagen trainer (multi-stage + tini + non-root)
├── docker/mlflow/Dockerfile                 # mlflow v3.12 + psycopg2-binary + boto3
├── .env.example                             # Variables de entorno (S3 buckets + AWS profile + límites)
├── requirements.txt, requirements-dev.txt
├── scripts/                                 # módulos importables desde main.py + tasks Python
│   ├── prepare_data.py                     # split del acumulado por VARIEDAD (usado por main.py en AWS Batch + task data:split)
│   └── s3_sync.py                          # sync de artifacts/ + reports/ a S3 al final del run (usado por main.py)
└── src/
    ├── config.py                            # Esquema, rutas, seeds, MLflow URI, branding del reporte
    ├── utils/logger.py                      # setup_logging() archivo + consola
    ├── utils/sklearn_helpers.py             # helpers transversales para sklearn (dump_json_artifact, …)
    ├── diagnostics/                          # EDA standalone (eda.py) + dashboards / drift (ADR-004)
    ├── step_01_load/
    │   ├── data_loader.py                  # load_data() → (X_raw, y) con validación + leakage check
    │   └── validation.py                    # checks de schema y leakage
    ├── step_02_clean/
    │   ├── missing_flags.py                 # MissingFlagger (boolean per-feature)
    │   ├── imputers.py                      # CustomKNNImputer (KNN + median fallback)
    │   ├── outliers.py                      # OutlierCapper (iqr / percentile, tuneable)
    │   ├── outlier_score.py                 # score de outliers por (FUNDO, FORMATO) en cascada
    │   └── _helpers.py
    ├── step_03_features/
    │   ├── feature_engineering.py           # FeatureGenerator (cíclicas + ratios estructurales + one-hot)
    │   └── lag_features.py                  # add_lag_features (~35 features rolling/seasonal/ratios) — invocado desde data_loader.py
    ├── pipeline/build_pipeline.py           # missing_flags → imputer → outliers → features → variance_filter
    ├── step_04_train/
    │   ├── registry.py                      # BACKEND_REGISTRY: lista los backends entrenables (XGB, LGB)
    │   ├── model_xgb.py                     # get_xgb_model() — XGBRegressor envuelto en TTR (objective=reg:absoluteerror, MAE nativo)
    │   ├── model_lgb.py                     # get_lgb_model() — LGBMRegressor envuelto en TTR (objective=regression_l1, MAE nativo)
    │   ├── search_spaces.py                 # Espacios Optuna por backend (registry extensible)
    │   ├── target_transform.py              # log1p + cap p99.5 vía TransformedTargetRegressor (CV-safe)
    │   ├── oof_ensemble.py                  # K refits sobre folds; predict promedia las K
    │   ├── sample_weights.py                # weights ∝ 1/freq sobre bins del target (cap=5×, sqrt)
    │   ├── temporal_cv.py                   # TemporalYearSplit (chequeo de honestidad: CV_OUTER_STRATEGY=temporal_year; default=stratified)
    │   └── tuning.py                        # perform_nested_cv() + sample_weights + CV adaptativo
    ├── step_05_evaluate/
    │   ├── metrics.py                       # MAE, RMSE, R², MAPE
    │   ├── diagnostics.py                   # gráficos matplotlib → base64 PNG
    │   ├── statistical_tests.py             # tests sobre residuos (BP, DW, ADF, …)
    │   ├── champion.py                      # select_champion (gate de gap → MAPE_oof → tiempo)
    │   ├── explainability/                  # paquete: KPIs ejecutivos, glosario, WinnerKit, sesgos, acciones
    │   └── html/                            # dashboard ejecutivo modular
    │       ├── winner_dashboard.py          # entrypoint: Winner_<variety>.html
    │       ├── sections.py, technical.py    # KPIs, tarjetas, tablas, sección técnica
    │       ├── styles.py, helpers.py        # CSS embebido + utilidades
    │       └── __init__.py
    ├── step_06_track/
    │   ├── mlflow_registry.py               # init / log_metrics / log_params / log_pipeline / Model Registry
    │   ├── business_validation.py           # KG/JR (= KG/JR_H × H-EF) OOF + in-sample
    │   └── business_export/                  # paquete: builders + export + formatting → Winner_<variety>.xlsx
    └── orchestration/
        ├── cli.py                           # argparse + resolve_varieties / resolve_settings
        ├── single_run.py                    # entrena UN modelo para UNA variedad → ModelResult
        ├── variety_runner.py                # orquesta multi-modelo + select_champion + render dashboard
        ├── runners.py                       # run_sequential / run_parallel (procesos independientes)
        └── cleanup.py                       # gc.collect + plt.close + mlflow.end_run defensivo
```

---

## Flujo del pipeline

```
main.main()
└── parse_args + valid_backends() (siempre entrena todos los del registry)
└── runners.run_sequential | run_parallel
        └── variety_runner.run_variety(variety)
                ├── 1. data_loader.load_data(sheet=variety)
                │       ├── lee Excel, valida, drop leakage/inútiles, group-rare en FORMATO
                │       └── add_lag_features → ~31 features rolling/seasonal/ratios pre-CV
                ├── 2. build_pipeline.create_preprocessing_pipeline()
                │       ├── MissingFlagger
                │       ├── CustomKNNImputer       (KNN + median fallback si >30% missing)
                │       ├── OutlierCapper           (iqr | percentile, factor tuneable)
                │       ├── FeatureGenerator       (ratios estructurales + cíclicas sin/cos + one-hot, drop FECHA)
                │       └── VarianceThreshold      (descarta dummies constantes en la variedad)
                ├── 3. PARA CADA modelo en {xgb, lgb}:
                │       └── single_run.train_model(variety, model_type)
                │               └── tuning.perform_nested_cv()
                │                       ├── Outer CV: StratifiedKFold por FUNDO_FORMATO
                │                       │   (default; CV_OUTER_STRATEGY=temporal_year
                │                       │   activa TemporalYearSplit expanding-window
                │                       │   por ANIO — chequeo honesto pre-deploy)
                │                       ├── sample_weights ∝ 1/freq por bins de y
                │                       │   (igual ancho, cap=5×, normalizados a media=1)
                │                       ├── Outer CV (k=5)  → MAE_test, MAE_train, gap, R²
                │                       │   └── Inner CV (k=3) + Optuna TPE multivariado
                │                       │       sobre preprocessor params + model params
                │                       ├── Ronda final sobre dataset completo (final_trials)
                │                       └── refit final con OOFEnsembleRegressor
                │                           (K=5 pipelines sobre folds; predict promedia)
                ├── 4. champion.select_champion([result_xgb, result_lgb])
                │       └── lex-order: gate de gap → MAPE_oof → tiempo
                ├── 5. business_validation: KG/JR = KG/JR_H × H-EF (OOF + in-sample)
                ├── 6. mlflow_registry: log metrics + params + pipeline + signature
                │       └── registra nueva version del campeón en `rnd-forest-<variety>`
                └── 7. winner_dashboard + business_export → Winner_<variety>.html / .xlsx
```

---

## Decisiones técnicas (con respaldo estadístico)

### Esquema de features

Tras EDA sobre 10 073 filas:

| Columna | Decisión | Justificación |
|---|---|---|
| `KG/JR`, `H-EF` | **excluir** | leakage — `target = KG/JR ÷ H-EF` exactamente (max\_abs\_diff = 0). |
| `VARIEDAD` | excluir | un único valor por hoja (split por variedad). |
| `CALIBRADO`, `DIA_SEM`, `MES` (raw) | excluir | mutual_info ≈ 0 vs target o reemplazadas por sus cíclicas. |
| `KG/HA` | mantener | top MI = 0.48 (driver dominante). |
| `%INDUS` | mantener | MI = 0.29. |
| `DPC` | mantener | MI = 0.22. |
| `P/BAYA` | mantener | MI = 0.13 (KNN-impute, ~39% missing). |
| `HA` | mantener | MI = 0.08. |
| `DIA_COSECHA` | mantener | día desde el inicio de la cosecha (capta deriva intra-temporada). |
| `FORMATO` | one-hot + group-rare | colapsa categorías con n<50 a `OTROS` (`RARE_MIN_COUNT`). |
| `FUNDO` | one-hot | mean LN ≈ 2.02 vs A9 ≈ 5.36 (≈ 2.6×). |
| `FECHA` | derivar cíclicas + año | estacionalidad fuerte (abril 2.3 → septiembre 6.7) y deriva temporal entre años. |

### Features derivadas (lag + estructurales)

Además de las 6 numéricas raw, 2 categóricas one-hot y 13 cíclicas/temporales,
el pipeline genera dos bloques de features derivadas:

**Lag features** (`step_03_features/lag_features.py`, ~31 columnas, calculadas en `data_loader.py` antes del CV split):

- **Rolling por grupo × variable × ventana** (24 cols): mediana de las N obs anteriores con `shift(1)` para excluir la fila actual.
  - Grupos: `FUNDO+FORMATO` (FF), `FUNDO` (F), `FORMATO` (FMT) — cascada de densidad.
  - Variables: `KG_JR_H` (target) y `KG/HA`.
  - Ventanas: 7d, 14d, 30d, 90d.
- **Lag estacional anual** (2 cols): mediana en ventana `fecha − 365d ± 15d` por (FUNDO, FORMATO). Captura ciclo agronómico que `90d` no ve.
- **Ratios temporales** (3 cols):
  - `KG_HA_ratio_FF_30/90 = KG/HA actual / KG_HA_lag_FF_30(o 90)` — desempeño relativo al histórico.
  - `delta_KG_JR_H_30_90 = lag_30 / lag_90` — aceleración / deceleración del target.
- **Cold-start flags** (2 cols): `LAG_FF_COLD`, `LAG_FF_SEASONAL_COLD` con sentinel `-1` para grupos sin historia. Los árboles aíslan estas filas como hojas distintas.

> Los lags `KG_JR_H_lag_*` cumplen la función de un target encoding pero **temporal,
> multi-window y multi-grupo** (con fallback FF → F → FMT por densidad). Por eso un
> mean encoding plano sobre FUNDO/FORMATO sería redundante y conceptualmente inferior.

**Ratios estructurales intra-fila** (`step_03_features/feature_engineering.py`, 4 columnas, calculados dentro del Pipeline después del imputer y outlier capping):

| Feature | Fórmula | Captura |
|---|---|---|
| `KG_TOTAL` | `KG/HA × HA` | Kilos absolutos cosechados (escala/tamaño de la parcela). |
| `INDUS_KG_HA` | `%INDUS × KG/HA` | Kg de calidad industrial por hectárea. |
| `KG_PER_BAYA` | `KG/HA ÷ P/BAYA` | Eficiencia volumétrica (cuánto pesa el agregado). |
| `KG_HA_PER_DPC` | `KG/HA ÷ DPC` | Velocidad de cosecha por unidad DPC. |

Las divisiones usan `np.where(den > 0, num/den, NaN)`; los NaN resultantes son tratados nativamente por XGB y LGB (rama default elegida por loss en el split). No tocan target ni `H-EF`, así que no requieren CV-fold awareness.

### Tuning

- **Optuna TPE multivariado** (acelera convergencia explotando correlaciones entre params); sin pruner porque cada trial reporta UN solo score (CV ya hecho), no valores intermedios.
- **El preprocesador también se tunea**: `imputer__n_neighbors`, `outliers__method`, `outliers__factor`. El espacio de búsqueda es el pipeline completo, no solo el modelo.
- **Search spaces aislados por backend** (`search_spaces.py`): registry extensible — añadir un backend implica un solo archivo, sin tocar `tuning.py`.

### Nested CV adaptativo

- **Outer (k=5)** mide generalización del PROCEDIMIENTO completo (preproc + tuning + entrenamiento). **Inner (k=3)** selecciona hiperparámetros sin contaminar el outer test.
- **Estratificación adaptativa por variedad**: cascada `FUNDO_FORMATO` → `FUNDO` → `FORMATO` → `KFold` simple. En cada nivel se valida que tras colapsar clases con `n<min_count` a `RARE` queden ≥2 clases distintas con tamaño suficiente para el inner.
- `min_count = max(outer, ceil(inner × outer / (outer−1)))` — garantiza que el inner Stratified tenga ≥`inner_folds` por clase tras el split del outer.

### Anti-overfitting / estabilidad

- **`TransformedTargetRegressor`** (`target_transform.py`): aplica `log1p(min(y, p99.5))` en el espacio de y antes de fittear el modelo y `expm1` al predecir. CV-safe porque el cap se calcula DENTRO de cada fold sobre `y_train`. Estabiliza varianza y aplasta ~0.5% de outliers extremos sin tocar `y_test` del scoring.
- **Sample weights** (`tuning.compute_sample_weights`): pesos inversos a la densidad del target con bins de IGUAL ANCHO (no qcut), cap=5× + `sqrt` para saturar la cola larga (max post-norm ~2.8 vs ~8 sin sqrt), normalizados a media=1. Compensa el sesgo "regresión a la media" de los árboles dando más peso a deciles raros sin amplificar outliers.
- **`OOFEnsembleRegressor`** (`oof_ensemble.py`): refit final = K=5 pipelines clonados, cada uno entrenado en `(K−1)/K` del dataset según un `KFold`; `predict()` promedia las K predicciones. Reduce varianza ~5–10% del modelo de producción a costa de 5× el tiempo del refit final (despreciable vs nested CV). `K=1` degenera al modo legacy bit-for-bit.

### Validación en unidad de negocio

`KG/JR_H` (kg/jornal-hora) es la unidad del modelo, pero la unidad
gerencial es `KG/JR` (kg/jornal). `business_validation.py` recompone
`KG/JR = predicción × H-EF` y reporta MAPE/R² en esa unidad, tanto
**OOF** (honesto, para el dashboard) como **in-sample** (sanity check del
modelo de producción aplicado a toda la historia).

### Reporte gerencial (`Winner_<variety>.html`)

Una sola página HTML autocontenida (Plotly inline o por CDN según
`REPORT_PLOTLY_OFFLINE`):

- Header con marca, unidad de negocio, descripción del modelo, fecha.
- Resumen ejecutivo con badge semáforo (Excelente / Aceptable / Insuficiente / No recomendado) según R² OOF + brecha train-test.
- Gauges de R² y MAE contra targets gerenciales (`REPORT_R2_TARGET`, `REPORT_MAE_TARGET`).
- 4 KPIs con código de color (precisión, R², mejora vs baseline, estado).
- Gráficos embebidos: predicho vs real OOF, residuales.
- Desempeño por subgrupo (FORMATO, FUNDO) con flag automático de subgrupos problemáticos (`MAPE > 1.5 × MAPE_global` y `n ≥ REPORT_SUBGROUP_MIN_N`).
- Análisis de overfitting (gap relativo, verdict).
- Tabla de hiperparámetros óptimos + representación técnica del pipeline.
- Bloque de "Aplicación Total" (modelo refit aplicado a toda la historia, in-sample).

Las predicciones honestas que alimentan los gráficos son **out-of-fold**:
cada predicción proviene de un modelo que NO observó esa fila durante
el entrenamiento.

---

## Convención de nombres

Los módulos `step_XX_verbo/` codifican el orden del pipeline en el propio
nombre — el lector entiende la secuencia sin abrir un diagrama. Se mantienen
así por:

1. **Determinismo visual**: `01_load → 02_clean → 03_features → 04_train → 05_evaluate → 06_track` es legible sin contexto.
2. **Compatibilidad Python**: módulos no pueden empezar con dígitos puros (`01_load` falla); el prefijo `step_` lo resuelve.
3. **Estabilidad de imports**: renombrar implica tocar todos los `from src.step_X import ...` y los `.joblib` ya serializados (que recuerdan el path del pipeline). El costo supera al beneficio.

---

## Reproducibilidad

- `RANDOM_STATE = 42` propagado a `KFold`, `StratifiedKFold`, `TPESampler`, `OOFEnsembleRegressor`, `XGBRegressor`, `LGBMRegressor`.
- Cada run de MLflow guarda parámetros del pipeline completo (modelo + preprocesador), métricas Train/Test/Full, y el pipeline serializado con `infer_signature` (modelo autodescribible para `mlflow models serve`).
- `best_params_<variety>_<run_name>.json` se sube como artifact en `hyperparameters/` para evitar el truncado a 250 chars de `mlflow.log_params`.
- El log incluye `archivo:línea:función` para localizar fallos en segundos.
- `logs/business_audit.jsonl` registra cada run con métricas de negocio para auditoría histórica (lectura manual; el comparador automático se removió por desuso).

---

## MLflow: experimentos y Model Registry

- **Un experimento por variedad**: `MLFLOW_EXPERIMENT_PREFIX + variety`. Default prefix vacío → el experimento es el nombre de la variedad (`POP`, `JUPITER`, …).
- **Run versionado dentro del experimento**: `xgb_v1`, `xgb_v2`, …, `lgb_v1`, … (`next_run_version` autoincrementa por modelo). El campeón histórico vive en el mismo experimento que sus rivales y se distingue por sus tags.
- **Model Registry**: `MODEL_REGISTRY_PREFIX + variety` (default `rnd-forest-POP`, `rnd-forest-JUPITER`, …). Cada training del campeón crea una nueva versión del registered model. La promoción a `Staging` / `Production` NO es parte del training run — se hace fuera del pipeline (manual desde la UI, vía `mlflow models transition`, o vía el workflow CI/CD `promote.yml` descripto en `GUIA_MLOPS_AWS_V2.md` #12, que aplica gates de calidad antes de promover).

> El `MODEL_REGISTRY_PREFIX` por default está alineado con cualquier servicio
> downstream que cargue modelos como `f"rnd-forest-{variety}"`. Cambiarlo
> aquí requiere coordinarlo con esos consumers o no encontrarán los modelos.

---

## Auditoría / hardening aplicado

| Categoría | Mejora |
|---|---|
| Overfitting | Cada outer fold reporta `MAE_train`, `MAE_test`, `gap`. El reporte HTML agrega "Análisis de overfitting" con verdict (verde/amarillo/rojo) según gap. El selector de campeón usa `|gap|` como **primer** criterio. |
| Selección multi-modelo | `select_champion` con lex-order estricto + tolerancias por bucket. Justificación textual auto-generada. |
| Estabilidad varianza | `TransformedTargetRegressor` (log1p + cap p99.5 CV-safe) + `OOFEnsembleRegressor` (K=5 refits promediados). |
| Compensación cola | `compute_sample_weights` por bins de igual ancho del target (cap=5×). |
| MLflow run | Tags clave (`r2_mean`, `mae_test_mean`, `mae_train_mean`, `overfit_gap`, `composite_score`) → filtros directos en la UI. |
| MLflow artifacts | `best_params_*.json` versionado por run_name (sin overwrite entre re-entrenamientos). |
| MLflow signature | `infer_signature` con muestra de X/y; ints casteados a float64 para no romper schema enforcement con NaN en inferencia. |
| Multi-variety | `--parallel-varieties N`: procesos independientes con cache cleanup natural; auto-ajusta `inner_cv_n_jobs = cores // N` para evitar oversubscription. |
| Cache cleanup | Entre variedades: `gc.collect()`, `plt.close('all')`, `mlflow.end_run()` defensivo (`orchestration/cleanup.py`). |
| Skip-final-tuning | Cuando se omite la ronda final, usa el `best_params` del **mejor** outer fold (argmin MAE_test), no el último. |
| Stratified adaptativo | Cascada `FUNDO_FORMATO → FUNDO → FORMATO → KFold` con colapso de clases raras a `RARE`. |
| Group-rare | `RARE_MIN_COUNT=50` en FORMATO: categorías con `n<50` se colapsan a `OTROS` antes del one-hot. |
| Auditoría JSONL | `logs/business_audit.jsonl` (1 línea por run) — para inspección manual de métricas de negocio. |
