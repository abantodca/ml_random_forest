# Arquitectura — ml_training

Vista **visual** (C4 + secuencia + despliegue) del sistema end-to-end. Es la capa
de diagramas que complementa, sin duplicar, las fuentes autoritativas:

- **`README.md`** — diseño ML a fondo: features, champion, nested-CV, anti-overfitting,
  convenciones MLflow (`#197 Mapa de la arquitectura`, `#264 Flujo del pipeline`).
- **`GUIA_MLOPS_AWS_V2.md`** — runbook local + AWS y los **ADR-001..004**.
- **`CLAUDE.md`** — invariantes no-obvios (leer antes de tocar código).

> Los diagramas son Mermaid: GitHub los renderiza nativo. En local, cualquier
> previsualizador de Markdown con Mermaid (VS Code + extensión) los muestra.

---

## 1. Contexto del sistema (C4 · nivel 1)

Qué problema resuelve y con quién/qué habla. El sistema pronostica
**`KG/JR_H`** (kg por jornal-hora) por **variedad** de cultivo.

```mermaid
graph TB
    operador["👤 Operador ML<br/><i>entrena, tunea, promueve modelos</i>"]
    negocio["👤 Usuario de negocio<br/><i>consulta pronósticos por variedad</i>"]

    subgraph sistema["🟦 ml_training (sistema MLOps)"]
        direction LR
        trainer["Trainer<br/>XGB + LGB por variedad"]
        api["API<br/>sirve modelos + persiste forecasts"]
        ui["UI<br/>dashboard de gestión"]
    end

    s3[("☁️ S3<br/>artifacts + reports + data")]
    mlflow["MLflow<br/>tracking + Model Registry"]
    pg[("🐘 Postgres<br/>MLflow backend + forecasts")]

    operador -->|task train / CLI| trainer
    negocio -->|navegador| ui
    ui -->|REST interno| api
    trainer -->|registra rnd-forest-&lt;variety&gt;| mlflow
    api -->|carga modelos| mlflow
    trainer & api & mlflow --> s3
    mlflow & api --> pg
```

**Frontera clave:** el sistema **decide** el modelo campeón (ADR-002); no hay flag
para forzar un backend. El contrato entre piezas es el prefijo de registro
`rnd-forest-<variety>` (invariante #8 en `CLAUDE.md`).

---

## 2. Contenedores (C4 · nivel 2)

Tres deployables comparten **una sola** base de código ML (`src/`). Esto es
deliberado: el `api/Dockerfile` tiene como build-context la raíz del repo
para `COPY src/` — trainer y API cargan el **mismo** pipeline (invariante #1).

```mermaid
graph TB
    subgraph build["Código compartido"]
        src["📦 src/ — pipeline ML<br/><i>única fuente de verdad</i>"]
    end

    subgraph deployables["Deployables"]
        trainer["🐍 Trainer<br/>main.py + src/<br/>(Batch / docker run)"]
        api["⚡ API · FastAPI<br/>api/app + src/"]
        ui["📊 UI · Streamlit<br/>ui/app"]
    end

    subgraph estado["Estado / infra"]
        mlflowSrv["MLflow server<br/>--serve-artifacts"]
        pg[("Postgres<br/>mlflow + forecasts DB")]
        s3[("S3<br/>artifacts/ reports/ data/")]
        nginx["nginx<br/>reports estáticos :8080"]
    end

    src -.COPY build-time.-> trainer
    src -.COPY build-time.-> api
    trainer -->|log_model + sync| mlflowSrv
    trainer -->|HTML/Excel| nginx
    trainer --> s3
    api -->|REST 8000| mlflowSrv
    api -->|forecasts| pg
    ui -->|service discovery<br/>api.ml-training.local:8000| api
    mlflowSrv --> pg
    mlflowSrv -->|artifacts proxy| s3

    classDef shared fill:#dbeafe,stroke:#2563eb;
    class src shared;
```

| Contenedor | Puerto local | Rol | Doc |
|---|---|---|---|
| Trainer | — (job) | entrena XGB+LGB, elige champion, registra | README #264 |
| API (FastAPI) | `:8000/docs` | sirve modelos + persiste forecasts a Postgres | — |
| UI (Streamlit) | `:8501` | dashboard de gestión | — |
| MLflow | `:5000` | tracking + registry (backend **siempre** PG+S3, ADR-001/003) | GUIA ADR-001 |
| Postgres | interno | DB de MLflow **+** DB `forecasts` (separadas, invariante #7) | — |
| nginx reports | `:8080` | HTML/Excel estáticos del dashboard | — |

> **Ruteo en prod:** la API se enruta por prefijos específicos
> (`/api/health*`, `/api/forecasts*`, …), **nunca** `/api/*` genérico — MLflow
> con `--serve-artifacts` es dueño de `/api/2.0/mlflow-artifacts/*` (invariante #6).

---

## 3. Componentes (C4 · nivel 3)

### 3.1 Trainer — pipeline por pasos (`src/`)

Los paquetes `step_XX_verbo/` **codifican el orden del pipeline** y sus nombres
están horneados en los `.joblib` serializados → **no se renombran** (invariante #4).

```mermaid
graph LR
    load["step_01_load<br/>data_loader · validation"] -->
    clean["step_02_clean<br/>flags · imputer · outliers"] -->
    feat["step_03_features<br/>FeatureGenerator<br/>LagFeatureTransformer"] -->
    train["step_04_train<br/>registry · XGB · LGB<br/>tuning (nested-CV)"] -->
    eval["step_05_evaluate<br/>champion · conformal<br/>explainability · html"] -->
    track["step_06_track<br/>MLflow registry<br/>business_export"]

    orch["orchestration/<br/>variety_runner · single_run · runners"]
    orch -.orquesta.-> load
    orch -.orquesta.-> track
    pipe["pipeline/build_pipeline.py<br/>sklearn Pipeline"]
    feat -.dentro del Pipeline.-> pipe
```

**Invariante #9 (anti-leakage):** los lags se computan **dentro** del
`sklearn.Pipeline` (`LagFeatureTransformer`, paso 0), no en el loader — así cada
fold de CV calcula lags solo desde su propio slice de train.

### 3.2 API — capas (`api/app/`)

```mermaid
graph TB
    routers["routers/<br/>forecasts · varieties · history · health"]
    services["services/<br/>forecast · mlflow · drift · uncertainty<br/>excel · feature_pipeline · health"]
    models["models/ · crud/<br/>SQLAlchemy + Postgres"]
    schemas["schemas/<br/>Pydantic I/O"]
    routers --> services --> models
    routers -.valida.-> schemas
    services -->|carga rnd-forest-*| mlflowext["MLflow"]
    services -.usa.-> srcpkg["📦 src/ (mismo pipeline)"]
```

### 3.3 UI — capas (`ui/app/`)

`views/` son las páginas reales (registradas vía `st.navigation`); **no** hay
`pages/`. El `client/` espeja la superficie de la API (mantener en sync, inv. #10).

```mermaid
graph LR
    views["views/<br/>home · forecast · models<br/>model_report · tracking · system"]
    client["client/<br/>api_client · endpoints · mappers"]
    views --> client -->|REST| apiext["API"]
```

---

## 4. Secuencia — un entrenamiento end-to-end

```mermaid
sequenceDiagram
    actor Op as Operador
    participant T as Trainer
    participant CV as nested-CV (step_04)
    participant CH as select_champion (step_05)
    participant ML as MLflow (PG+S3)
    participant N as nginx reports

    Op->>T: task train VARIETIES=POP TUNING=prod
    T->>T: load → clean → features (Pipeline)
    loop por backend (XGB, LGB)
        T->>CV: nested-CV + Optuna
        CV-->>T: OOF MAPE + params + wall time
        T->>ML: log_model + métricas (run por backend)
    end
    T->>CH: gate gap → OOF MAPE → wall time
    CH-->>T: campeón de la variedad
    T->>ML: register rnd-forest-POP (si pasa gate)
    T->>N: Winner_POP_*.html + index.html
    T->>ML: sync artifacts/ + reports/ a S3
    T-->>Op: champion + reporte
```

> `--tuning smoke` **nunca** registra modelos (invariante #2). El gate de champion
> es lex-order estricto: `|gap|` (constraint) → OOF business MAPE → wall time.

## 5. Secuencia — servir un pronóstico

```mermaid
sequenceDiagram
    actor U as Usuario
    participant UI as Streamlit
    participant API as FastAPI
    participant ML as MLflow
    participant PG as Postgres (forecasts)

    U->>UI: elige variedad + inputs
    UI->>API: POST /api/forecasts (service discovery)
    API->>ML: carga rnd-forest-<variety> (cache)
    API->>API: pipeline.predict + bandas conformes
    API->>PG: persiste forecast
    API-->>UI: punto + intervalo + drift
    UI-->>U: render
```

---

## 6. Despliegue

### Local (docker compose)

```mermaid
graph TB
    subgraph compose["docker compose · una red"]
        pg[(postgres)] --- mlflow --- reports
        mlflow --- api --- ui
        trainer
    end
    dev["👤 dev"] -->|:8501 :8000 :5000 :8080| compose
```

`postgres + mlflow + reports + api + ui` levantan como bloque; el trainer corre
on-demand. URLs: UI `:8501`, API `:8000/docs`, MLflow `:5000`, reports `:8080`.
**No hay `:80` local** (eso es el ALB de prod).

### AWS (Terraform · `infra/modules/`)

```mermaid
graph TB
    alb["ALB<br/>ruteo por path-prefix"]
    subgraph ecs["ECS"]
        apis["api"] & uis["ui"] & mlflows["mlflow"] & reportss["reports (sync S3)"]
    end
    batch["AWS Batch<br/>trainer job"]
    rds[("RDS Postgres")]
    s3[("S3")]
    ecr["ECR ×5"]
    alb --> apis & uis & mlflows & reportss
    ecs --> rds & s3
    batch --> rds & s3
    ecr -.imágenes.-> ecs & batch
```

Módulos Terraform: `network · storage · mlflow · api · ui · reports · batch ·
scheduler · lambdas · monitoring · cicd · _shared`. CI/CD (GHA OIDC → ECR/ECS)
en `GUIA_MLOPS_AWS_V2.md #3.10`. `task wake/sleep` apaga/enciende el bloque
(RDS+MLflow+reports+api+ui) como modelo de costo.

---

## 7. Invariantes que la arquitectura DEBE preservar

Resumen accionable; el detalle vive en `CLAUDE.md` (#1–#10).

| # | Invariante | Por qué |
|---|---|---|
| 1 | `src/` única fuente de verdad; API la `COPY`a (no vendoring) | trainer y API comparten pipeline; vendoring → drift silencioso |
| 4 | No renombrar `step_XX_verbo/` | paths horneados en `.joblib` serializados |
| 8 | Prefijo `rnd-forest-<variety>` es contrato trainer↔API | la API carga por ese nombre |
| 9 | Lags **dentro** del Pipeline | evita leakage entre folds de CV |
| 1/3 | MLflow backend **siempre** PG+S3 | nunca `file://mlruns`, sqlite, LocalStack |

---

*Para profundidad de cada decisión: `README.md` (#305 Decisiones técnicas con
respaldo estadístico) y los ADR en `GUIA_MLOPS_AWS_V2.md`.*
