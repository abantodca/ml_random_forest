# Guía MLOps AWS — Despliegue a producción de `ml_training`

> Manual operativo para llevar el trainer (`src/`, `main.py`) desde una laptop
> hasta AWS Batch con MLflow Tracking + Registry. La guía se lee en dos
> tramos: **Tramo I (Local)**, Capítulos 1-4, donde se construye el entorno
> Docker desde cero a partir del código fuente; y **Tramo II (AWS)**, Partes
> 1-13, donde el mismo binario se promueve a producción. Cada sección
> termina con una verificación; si falla, no se avanza.

---

## Tabla de contenidos

### 🏠 Tramo I — Entorno local (Capítulos 1-4)

> Construir el binario y validarlo en tu máquina, con Docker. **No avances al Tramo II** hasta que el smoke local termine en verde.

| # | Capítulo | Qué se hace |
|---|---|---|
| 1 | [Visión general](#capítulo-1--visión-general) | Qué entrenamos, dos entornos una imagen, endpoints, costo objetivo |
| 2 | [Decisiones fijas](#capítulo-2--decisiones-fijas) | Qué no se discute (lenguaje, runtime, region, modelos) |
| 3 | [Prerrequisitos del host](#capítulo-3--prerrequisitos-del-host) | Herramientas, WSL, credenciales AWS, variables de sesión |
| 4 | [Entorno local desde cero](#capítulo-4--entorno-local-desde-cero) | `.dockerignore`, `Dockerfile`, compose, Taskfile, smoke + verificación |

### ☁️ Tramo II — Producción AWS (Partes 1-12)

> Mismo binario, promovido a AWS Batch + MLflow productivo via Terraform + Task + GitHub Actions.

| # | Parte | Qué se hace |
|---|---|---|
| 1 | [Lifecycle](#parte-1--overview-del-lifecycle-y-stand-up) | Modos: STAND-UP / TEAR-DOWN / REBUILD / DESTROY |
| 2 | [Bootstrap irreversible](#parte-2--bootstrap-irreversible) | Backend Terraform (S3 + lock nativo) + OIDC (UNA VEZ) |
| 3 | [Módulos Terraform](#parte-3--modulos-terraform) | network · storage · mlflow · reports · batch · monitoring · lambdas · scheduler · cicd · consumer-iam |
| 4 | [Apply incremental + smoke](#parte-4--apply-incremental--smoke-test) | 4 `tasks/*.yml` + helpers `lib/` + Olas A/B/C + smoke end-to-end |
| 5 | [Patch trainer → CloudWatch](#parte-5--patch-del-trainer-emitir-mape-a-cloudwatch) | Emitir métrica MAPE custom |
| 6 | [CI/CD con GitHub Actions](#parte-6--cicd-con-github-actions) | Workflows `deploy.yml` · `training.yml` · `destroy.yml` |
| 7 | [Promotion gate](#parte-7--promotion-gate-extendido) | Staging → Production con MAPE + A/B + approval |
| 8 | [Runbook operativo](#parte-8--runbook-operativo-extendido) | Diario · semanal · incidentes · TEAR-DOWN · REBUILD · DESTROY |
| 9 | [Costos detallados](#parte-9--costos-detallados) | ~$75/mes con scheduler L-V 08-12 PET (incluye API + UI) |
| 11 | [Troubleshooting (catálogo)](#parte-11--troubleshooting-catalogo) | Errores comunes y fixes |
| 12 | [Apéndices](#parte-12--apendices) | Glosario · conceptos · mapa de archivos |

---

## Cómo leer esta guía

- **Orden de lectura**: local primero, AWS después. No avances al Tramo II
  sin que el smoke local del Capítulo 4 termine en verde.
- **Punto de partida real**: la única asunción es que el repo tiene `src/`,
  `main.py`, `scripts/` y `requirements.txt`. Todo lo demás — Dockerfile,
  compose, Taskfile, `.env` — se construye en el Capítulo 4.
- **Convención de comandos**: todos los bloques `bash` se ejecutan desde la
  raíz del repo. En Windows, exclusivamente desde **WSL Ubuntu** (no Git
  Bash, no PowerShell).
- **Una sola imagen**: el `Dockerfile` que usa `task build` localmente es el
  mismo binario que `task ecr:build IMG=trainer` empuja a ECR en producción.
- **Convención de avisos**:
  - `> **Nota** — …` aclara el porqué de una decisión (inline, dentro de pasos).
  - `> **Warning** — …` señala riesgos reales (pérdida de datos, costo
    inesperado, operación irreversible) dentro de pasos.
  - Para transiciones críticas entre Partes/Capítulos se usan **callouts
    GitHub-flavored**, que en GitHub.com y la mayoría de previews renderean
    como cajas coloreadas:
    - `> [!NOTE]` — información complementaria.
    - `> [!TIP]` — atajo o mejor práctica.
    - `> [!IMPORTANT]` — paso que **no se puede saltear**.
    - `> [!WARNING]` — operación con riesgo de pérdida de tiempo o datos.
    - `> [!CAUTION]` — operación **irreversible** (destroy, nuke, force-push).
- **Convención de verificación**: cada bloque importante cierra con un
  comando o tabla que valida el estado. Si falla, parar y resolver antes
  de seguir.
- **Re-ejecutabilidad**: toda sección que produce archivos o recursos AWS
  es idempotente (correr N veces da el mismo estado). El commit sugerido
  aparece al final cuando aplica. Las secciones sin gotchas únicos NO
  llevan callout `🔄 Al re-ejecutar` — solo se incluye cuando hay un
  pitfall o checkpoint específico que no esté en el cuerpo de la sección.

---

# Tramo I — Entorno local

## Capítulo 1 · Visión general

### 1.1 Qué entrenamos

`ml_training` predice **kg/jornal-hora** (`KG/JR_H`) por variedad a partir
de un Excel histórico de cosechas (`data/BD_HISTORICO_ACUMULADO.xlsx`). El
sistema entrena **XGBoost** y **LightGBM** con Optuna, evalúa con
`TimeSeriesSplit`, y elige campeón por variedad según orden lexicográfico
(gap → MAPE → tiempo).

> **Nota** — Pese al nombre del repo (`ml_random_forest`), los backends
> activos son XGBoost + LightGBM. Random Forest fue reemplazado por
> estabilidad numérica del target con `log1p` + cap-p99.

### 1.2 Dos entornos, las mismas imágenes

| | Local | Producción AWS |
|---|---|---|
| Compute training | Docker compose (laptop) | AWS Batch + EC2 c6i.2xlarge |
| Compute serving (API+UI) | Docker compose (`api` + `ui`) | ECS Fargate detrás del ALB (Capa 4.5) |
| Tracking server | MLflow container, Postgres en volumen | MLflow en ECS Fargate, RDS Postgres |
| Artifacts store | S3 sandbox | S3 productivo + Model Registry |
| Trigger | `task train` (manual) | GitHub Actions / Lambda dispatcher |
| Imágenes | 5 (trainer, mlflow, reports, api, ui) build local | Las mismas 5, push a ECR (#4.4) |

### 1.3 Endpoints en producción

```
http://<ALB-DNS>/             MLflow UI (tracking + Model Registry)
http://<ALB-DNS>/app/         UI Streamlit (dashboard gerencial de pronosticos)
http://<ALB-DNS>/docs         API FastAPI — Swagger (POST /api/forecasts, GET /api/varieties/history)
http://<ALB-DNS>/reports/     Dashboards HTML por variedad
http://<ALB-DNS>/artifacts/   Artifacts crudos por run
```

> Todo cuelga del **:80 de un solo ALB** ruteado por path (ver #3.12 para el
> App stack API+UI, Capa 4.5). El DNS del ALB lo imprime `task infra:urls`.

### 1.4 Flujo end-to-end

```
Developer
  │ push a main
  ▼
GitHub Actions (deploy.yml)
  │ lint + build + push a ECR (via OIDC; sin job test — ver ADR-008)
  ▼
ECR ml-training:<sha>
  │ workflow_dispatch training.yml  (o `aws lambda invoke ml-training-dispatcher`)
  ▼
Lambda dispatcher → AWS Batch SubmitJob (Spot queue)
  │ autoscale 0 → 1 EC2 c6i.2xlarge
  ▼
Container del trainer
  │ 1. hydrate S3_DATA_BUCKET/S3_DATA_KEY → data/training/DB-HISTORICA.xlsx
  │ 2. main.py: por variedad entrena XGB + LGB con Optuna
  │ 3. champion.select_champion()
  │ 4. log a MLflow (Postgres backend + S3 artifacts)
  │ 5. sync_to_s3(artifacts/, reports/) a S3_ARTIFACTS_BUCKET
  ▼
MLflow Model Registry: nueva versión en stage "None"
  │ workflow_dispatch training.yml (action=promote)
  ▼
Quality gate (MAPE < umbral && A/B contra Production actual)
  │ approval humano en GitHub Environments
  ▼
MLflow Model Registry: versión transicionada a "Production"
```

> **Sobre el Model Registry en Tramo Local.** El diagrama es el flujo
> **productivo** (Tramo II). En local, `task train` loggea runs al MLflow del
> compose y sube el `.joblib` a S3 sandbox, **pero no registra el modelo en el
> Registry** — deliberado: versiones/stages/transitions no aportan mientras
> iterás código. El patch de Parte 5 inyecta `mlflow.register_model()` al
> promover a Tramo II, y de ahí cada run aparece como versión nueva en stage
> `None` para el quality gate de Parte 7. Para verlo en local sin esperar,
> apuntar el trainer al MLflow productivo vía `docker-compose.override.yml.example`.

### 1.5 Contrato del run MLflow

Cualquier run que el trainer escriba — local en tu laptop o productivo
en Batch — cumple un **contrato fijo**, alineado con Cap 2. Quien
revise el run después (vos, un colega, el quality gate de Parte 7) lo
puede auditar end-to-end sin inspeccionar el código del trainer:

**Tags de trazabilidad** (set en `mlflow.set_tags({...})` por `run_metadata.py` + `single_run.py`):

| Tag | Valor | Por qué importa |
|---|---|---|
| `git_commit` | SHA corto del `HEAD` al lanzar el run | Reproducibilidad del binario. |
| `git_dirty` | `true` / `false` (`git diff --quiet HEAD`) | Si `true`, el run **no es promovible**: hay código sin commit. |
| `dataset_sha256` | hash SHA-256 de `data/training/DB-HISTORICA.xlsx` | Distingue dos runs con mismo `git_commit` pero data distinta. |
| `dataset_n_rows` | filas totales agregadas sobre todas las hojas | Detecta truncamientos accidentales del Excel. |
| `tuning` | `smoke` / `dev` / `prod` / `prod_xl` | Contexto de búsqueda (presupuesto de trials). |
| `variety` | `POP` / `VENTURA` / … | Necesario porque el experimento es uno por variedad. |

**Artifacts obligatorios** (loggueados con `mlflow.sklearn.log_model(...)`):

| Artifact | Generador | Por qué |
|---|---|---|
| `model/MLmodel` + `model.pkl` | `mlflow.sklearn.log_model(model, "model", signature=..., input_example=...)` | El joblib del campeón persistido por MLflow (separado del `joblib` que sube `s3_sync`). |
| **`signature`** | `mlflow.models.infer_signature(X_train, model.predict(X_train))` | Schema de input/output. Sin esto, el Registry no valida payloads en serving futuro. |
| **`input_example`** | `X_train.head(5)` | Permite `mlflow models predict --input-path example.json` sin saber qué columnas espera el modelo. |
| `requirements.txt` | `mlflow.utils.environment._mlflow_conda_env(...)` o equivalente | Snapshot de deps del run. Útil para reconstruir el entorno 6 meses después. |

**Metrics esperadas** (mínimas, además de las que loguee Optuna):

- `mape_oof` (out-of-fold del CV, **emitida también a CloudWatch** por el patch de Parte 5).
- `mape_test` (sobre el holdout final).
- `gap_oof_test` (diferencia absoluta — detecta sobreajuste al CV).

> **Verificación**: el #4.10 check #7 confirma que estos 8 tags están
> presentes después de `task train`. Si alguno falta, el run no
> califica para `task ops:promote` (Parte 7) — el gate lo
> rechazará.

### 1.6 Costo objetivo

| Configuración | Costo mensual aproximado |
|---|---|
| **Tramo Local (laptop + 2 buckets sandbox)** | **~$0.05** (sólo storage S3) |
| Tramo II — Scheduler L-V 08-12 PET (default, incluye API + UI) | ~$75 |
| Tramo II — Sin scheduler (24/7) | ~$195 |
| Tramo II — Hibernado (tear-down: NAT GW permanente + storage) | ~$35 |

Detalle en Parte 9 (costos por servicio y por modo de lifecycle).

---

## Capítulo 2 · Decisiones fijas

Las siguientes decisiones no se discuten dentro de esta guía. Cambiar
alguna implica un ADR previo y reescritura de las secciones afectadas.

| Decisión | Elección | Por qué | Cambia a futuro |
|---|---|---|---|
| Región AWS | `us-east-1` | Latencia razonable desde Perú, todos los servicios disponibles, mejor precio Spot. | `us-east-2` o `sa-east-1` por compliance. |
| Compute training | Batch + EC2 c6i.2xlarge, queues Spot + On-Demand, `retry=2`. El dispatcher rutea `prod_xl` → On-Demand (sin interrupciones para jobs de ~4-6h) y el resto (`smoke`/`dev`/`prod`) → Spot. El default de `task batch:train` es `prod_xl` (converge con local) → On-Demand; usá `TUNING=prod` para Spot (−70% costo). | −70% costo con Spot; retry cubre interrupciones (~5-10% en c6i.2xlarge). | Fargate Spot, o `g5.xlarge` si pasás a DL. |
| Compute serving | ECS Fargate | Sin gestión de host, autoscale, integración nativa con ALB. | EC2 con AMI custom para aceleración. |
| Backend MLflow | Postgres + S3 (artifacts) | Estándar industria; soporta concurrencia. | Filesystem sólo en dev. |
| RDS | Postgres 15, `db.t4g.small`, single-AZ | ~$23/mes 24/7 (mucho menos con scheduler); hostea MLflow + la base `forecasts` de la API. | Multi-AZ (hardening, futuro). |
| Auto on/off de **servicios** | Scheduler EventBridge L-V 08-12 PET wake/sleep de RDS + Fargate (MLflow + Reports); chequeo de Batch RUNNING antes de apagar. | UI 4 h/día; training off-window despierta servicios on-demand. **Esto es scheduling de servicios, no de jobs de training** (ver fila "Trigger training"). | 24/7 si hay equipo distribuido. |
| TLS / WAF / Multi-AZ | NO (ALB :80 HTTP, sin WAF, RDS single-AZ) | Default barato. MLflow con basic auth + SG restrictivo. | TLS/WAF/Multi-AZ antes de exponer a Internet (hardening, futuro). |
| Egress privado | NAT GW single-AZ ($32/mes) | Setup simple si tráfico <10 GB/mes. | VPC endpoints (hardening, futuro). |
| Trigger training (de los **jobs**, no de los servicios) | (a) GHA `training.yml` workflow_dispatch (wake-train-sleep); (b) `aws lambda invoke ml-training-dispatcher`. **Sin cron de jobs**, sin S3 PutObject trigger — el scheduler de la fila anterior solo wake/sleep-ea servicios, nunca dispara entrenamientos. | Click desde GitHub UI eligiendo variedad. Training off-window wake-ea servicios y los apaga al terminar. | EventBridge cron diario de training / S3 trigger (Parte 7.5). |
| Modelos entrenables | **XGBoost + LightGBM** sobre `KG/JR_H`, con `TransformedTargetRegressor` (`log1p` + cap-p99). Champion automático. | Lo que vive en `src/step_04_train/registry.py`. | Stacking (eliminado, no existe). |
| Variedades válidas | **Dinámicas**: hojas del Excel `BD_HISTORICO_ACUMULADO.xlsx`. | Source of truth = el Excel. `list_varieties()` enumera `pd.ExcelFile(path).sheet_names`. La variable Terraform `varieties_allowed` es un allow-list defensivo del Lambda dispatcher, no la definición. | Agregar variedad = agregar hoja + `aws s3 cp` + opcional ampliar `varieties_allowed`. |
| Auth CI/CD | OIDC (sin access keys de larga duración) | Auditable en CloudTrail, sin rotación manual, blast-radius limitado al repo. | Keys sólo en CI legacy. |
| Promotion | Quality gate (MAPE < umbral) + A/B contra Production + approval en GitHub Environments | Defense in depth; un MAPE menor no garantiza modelo mejor sin baseline. | Auto-promote si MAPE absoluto <5% (no recomendado). |
| Reproducibilidad | `SEED=42` propagado a `np.random`, `random.seed`, `xgb.random_state`, `lgb.seed`, `TimeSeriesSplit(random_state=...)` y `optuna.create_study(sampler=TPESampler(seed=...))`. Fijado en config (no se loggea como tag). | Sin seed, dos runs idénticos producen campeones distintos por orden de trials. El "smoke OK" deja de ser auditoría. | Bayesian sin sampler determinístico (no recomendado). |
| Lineage del dataset | Tag MLflow `dataset_sha256=<sha>` + `dataset_n_rows=<N>` calculados sobre `data/training/DB-HISTORICA.xlsx` antes de cada run. | Permite responder "¿qué cambió entre el run que regresionó y el anterior?" cuando el código (`git_commit`) es idéntico. | DVC / lakeFS si el dataset crece a TB. |
| Contrato del run MLflow | Cada run loggea **tags obligatorios** (`git_commit`, `git_dirty`, `dataset_sha256`, `dataset_n_rows`, `tuning`, `variety`) **+ signature** (`infer_signature(X, y_pred)`) **+ input_example** (`X.head(5)`) **+ requirements snapshot** (`pip freeze`). Verificado en #4.10 check #7. | Sin signature, el Registry no valida payloads en serving; sin tags el quality gate de Parte 7 audita una caja negra. | Custom evaluators de MLflow (`mlflow.evaluate`) si pasás a regresión multi-target. |
| Code-quality gates | `ruff check src/ main.py scripts/` (local, manual + CI). El job `lint` de `deploy.yml` lo invoca + `task infra:validate`. **No hay tests todavía**: el pipeline se valida con smoke runs (`task train TUNING=smoke`, ~1 min) y los gates de modelo (`CHAMPION_MAX_MAPE`, `CHAMPION_MAX_GAP`). Ver [ADR-008](docs/adr/ADR-008-ci-sin-tests-todavia.md). | Un job `test` con `pytest tests/` que "saltea si no hay tests" es código aspiracional que oculta cobertura cero. Mejor reflejar el estado real y agregar el job cuando exista `tests/`. | Agregar `mypy --strict src/` + `bandit -r src/` + `pytest --cov-fail-under=N` cuando exista `tests/`. Tests mínimos candidatos: champion selector con empate, shape contract del `prepare_data`, joblib roundtrip. |
| CVE policy | `trivy image ml-training:local --severity HIGH,CRITICAL`. **Tramo Local**: warn-only. **Tramo II** (push a ECR): bloqueante. SBOM generado con `docker sbom` en cada `task ecr:build`. | Imagen base `python:3.13.1-slim-bookworm` tiene CVEs conocidos; para clientes regulados (HIPAA, SOC2, banca) el SBOM es requisito formal. | `cosign` para signing + admission controller en ECR. |
| Drift gate (PSI) | `task eda` calcula `psi_train_test` por feature numérica y escribe `artifacts/eda_<variety>.json`. `task train` lee ese JSON: `psi > 0.25` en cualquier feature **warn-loud** en stdout y se tagea el run con `psi_warn=true`. Bloqueante a futuro si pasa a ser política. | Drift severo entre train/test indica que el split de validación no representa training — un campeón sobre data drifteada es modelo sobre ruido. | `psi > 0.10` como warn temprano + `psi > 0.25` como bloqueante. |

---

## Capítulo 3 · Prerrequisitos del host

### 3.1 Herramientas

| Herramienta | Versión mínima | Verificación |
|---|---|---|
| Docker | 24+ con BuildKit | `docker version`, `docker info \| grep "Server Version"` |
| Git | 2.30+ | `git --version` |
| AWS CLI v2 | 2.0+ | `aws --version` |
| Task | 3.34+ | `task --version` |
| Terraform | 1.6+ (sólo Tramo II) | `terraform version` |
| jq | 1.6+ (post-apply checks) | `jq --version` |

Instalación de Task en Linux / WSL Ubuntu:

```bash
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/bin
export PATH="$HOME/bin:$PATH"   # persistir en ~/.bashrc
task --version
```

macOS:

```bash
brew install go-task
```

### 3.2 Windows: WSL Ubuntu obligatorio

El repo vive típicamente en disco Windows
(`C:\Users\<user>\Documents\Proyectos\ml_random_forest\ml_training`) y se
opera **desde WSL Ubuntu** vía el mount `/mnt/c/...`. Toda la guía asume
esa terminal.

```bash
# Desde PowerShell:
wsl -d Ubuntu

# Dentro de WSL:
cd /mnt/c/Users/<user>/Documents/Proyectos/ml_random_forest/ml_training
pwd
```

Tres ajustes una sola vez:

1. **Docker Desktop** → Settings → Resources → WSL integration → enable
   "Ubuntu". El comando `docker` desde WSL pega contra el mismo daemon
   que Docker Desktop.
2. **CRLF / LF**: en WSL sobre NTFS, git puede marcar todo como
   modificado. Normalizar una vez:
   ```bash
   git config --global core.autocrlf input
   git add --renormalize .
   ```
3. **Permisos POSIX**: NTFS no persiste el bit ejecutable. Invocá scripts
   con `bash infra/<script>.sh`, no `./<script>.sh`.

> **Warning** — En Windows, NO mezclar Git Bash ni PowerShell con WSL.
> Diferencias sutiles de line endings y rutas rompen Terraform, Task y
> Docker. Una sola terminal de principio a fin: WSL Ubuntu.

### 3.3 Credenciales AWS

Aunque el Tramo I es "local", el trainer sube artifacts a S3 y MLflow
escribe sus runs a S3, así que necesitás credenciales válidas desde el
primer `task build`.

```bash
aws configure --profile default
# Access Key ID, Secret Access Key, region us-east-1, output json

aws sts get-caller-identity
# { "UserId": "...", "Account": "<12-digitos>", "Arn": "arn:aws:iam::...:user/..." }
```

> **Warning** — Las credenciales viven **siempre** en `~/.aws/credentials`
> del host. `docker-compose.yml` monta `~/.aws:/aws:ro` en los containers
> y el SDK las lee de ahí. Nunca pongas `AWS_ACCESS_KEY_ID` ni
> `AWS_SECRET_ACCESS_KEY` en `.env` ni en el `Dockerfile`.

### 3.4 Service quotas (sólo para Tramo II)

> [!NOTE]
> **Salteable si solo vas a trabajar en Tramo I (local con Docker).** Los
> aumentos de quota de EC2 son únicamente necesarios para AWS Batch (Tramo
> II Parte 4). Si por ahora solo querés validar el binario en tu laptop,
> volvé a esta sección antes de la Parte 2 del Tramo II.

Los aumentos tardan 24-48 h: pedirlos **antes** del primer `terraform apply`.

| Servicio | Quota | Mínimo |
|---|---|---|
| EC2 Running On-Demand Standard (A/C/D/H/I/M/R/T/Z) | `L-1216C47A` | 32 vCPU |
| EC2 All Standard Spot Instance Requests | `L-34B43A08` | 32 vCPU |
| VPC NAT gateways per AZ | `L-FE5A380F` | 5 (default) |

```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-1216C47A \
  --desired-value 32
```

### 3.5 Variables de sesión

> [!TIP]
> **Tramo I** (local con Docker) sólo *usa* `AWS_PROFILE` y
> `AWS_DEFAULT_REGION` (los buckets sandbox los resuelve `tasks/local.yml`
> internamente via `aws sts get-caller-identity`); `PROJECT`, `ACCOUNT_ID` y
> `ACCOUNT_SUFFIX` recién hacen falta en **Tramo II** (los bloques `bash`
> manuales de Partes 1-2 los referencian de tu shell). Igual podés
> `source scripts/prod.env` desde el arranque: setea todo de una y las vars de
> más quedan inertes hasta que las necesites.

Setear **una vez por terminal nueva** (los `export` viven sólo en ese shell):

```bash
source scripts/prod.env
```

Esto exporta `AWS_PROFILE` + region (Tramo I) y `PROJECT` / `ACCOUNT_ID` /
`ACCOUNT_SUFFIX` (Tramo II), y **además deriva `DATA_BUCKET` / `ARTIFACTS_BUCKET`**
(vía `ensure-env.sh`), de modo que cualquier snippet del runbook usa
`"$DATA_BUCKET"` directo sin recomponer `"${PROJECT}-data-${ACCOUNT_SUFFIX}"`.
`ACCOUNT_SUFFIX` se deriva dinámicamente de `aws sts get-caller-identity`, así que
el archivo es portable entre cuentas; el único valor que quizá edites es
`AWS_PROFILE` (default `default`).

El repo ya trae los dos archivos (`prod.env` sourcea a `ensure-env.sh`); si
necesitás recrearlos:

> 📂 **Pegar este bloque en**: `scripts/prod.env`

```bash
# Variables de sesion para el runbook (GUIA_MLOPS_AWS_V2.md #3.5) en un solo
# archivo sourceable: setea AWS_PROFILE + region (Tramo I) y PROJECT/ACCOUNT_*
# (Tramo II), y deriva DATA_BUCKET / ARTIFACTS_BUCKET via ensure-env.sh.
# Uso: `source scripts/prod.env` (una vez por terminal nueva).

# === MINIMO para Tramo I (local con Docker) ===
export AWS_DEFAULT_REGION="us-east-1"
export AWS_REGION="$AWS_DEFAULT_REGION"   # alias requerido por el backend S3 de Terraform (#4.2)
export AWS_PROFILE="default"              # o el profile que uses (debe quedar antes del STS de abajo)

# === ADICIONALES para Tramo II (AWS) ===
export PROJECT="ml-training"
export ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export ACCOUNT_SUFFIX="${ACCOUNT_ID: -7}"

# Deriva y exporta los nombres de bucket canonicos (DATA_BUCKET / ARTIFACTS_BUCKET)
# desde una fuente unica de verdad. Tras esto, cualquier snippet del runbook puede
# usar "$DATA_BUCKET" directamente sin recomponer "${PROJECT}-data-${ACCOUNT_SUFFIX}".
source "$(dirname "${BASH_SOURCE[0]}")/ensure-env.sh"
```

> 📂 **Pegar este bloque en**: `scripts/ensure-env.sh`

```bash
#!/usr/bin/env bash
# Guard + derivacion de nombres compuestos para apuntar al stack prod.
# Uso: `source scripts/ensure-env.sh` antes de cualquier `aws ... | terraform ...`
# que componga nombres de bucket / role / ARN con $PROJECT y $ACCOUNT_SUFFIX.
#
# Aborta si las vars base estan vacias (tipico: terminal nueva sin
# `source scripts/prod.env`): sin esto los exports posteriores producen
# strings tipo "-data-" (sin prefijo ni suffix) y el comando aws falla con
# `Invalid bucket name ""` / `argument --bucket: expected one argument`.
#
# Ademas DERIVA los nombres de bucket canonicos para que ningun snippet del
# runbook tenga que recomponerlos a mano (origen del bug Parte 4.3: el snippet
# usaba $DATA_BUCKET, que nunca se exportaba).

: "${PROJECT:?ERROR: \$PROJECT vacia. Correr 'source scripts/prod.env' primero.}"
: "${ACCOUNT_SUFFIX:?ERROR: \$ACCOUNT_SUFFIX vacia. Correr 'source scripts/prod.env' primero.}"

# Nombres canonicos de bucket — fuente unica de verdad.
# `:=` respeta un override explicito en el shell y es idempotente al re-sourcear.
: "${DATA_BUCKET:=${PROJECT}-data-${ACCOUNT_SUFFIX}}"
: "${ARTIFACTS_BUCKET:=${PROJECT}-artifacts-${ACCOUNT_SUFFIX}}"
export DATA_BUCKET ARTIFACTS_BUCKET
```

**(Opcional)** En vez de sourcear `prod.env` podés exportar estas variables a
mano; si lo hacés, corré también `source scripts/ensure-env.sh` para derivar
`DATA_BUCKET` / `ARTIFACTS_BUCKET`.

> **Por qué exportar `AWS_REGION` además de `AWS_DEFAULT_REGION`** — el AWS
> CLI y el SDK Python aceptan ambas, pero el backend `s3` de Terraform sólo
> reconoce `AWS_REGION` o `AWS_DEFAULT_REGION` cuando no hay un `region` en
> `~/.aws/config`. La task `infra:_init` (única invocación canónica de
> `terraform init -reconfigure` en este proyecto — definida en
> `tasks/infra.yml`) inyecta `region=${AWS_REGION}` por `-backend-config`;
> si solo exportás `AWS_DEFAULT_REGION` el flag se expande a `region=` y
> el init falla con `Missing region value`. Exportar las dos elimina el
> footgun. **No correr `terraform init` a mano** — usar siempre `task
> infra:plan`/`apply`/`validate`, que disparan `_init` como dep.

| Variable | Valor | Tramo | Usada para |
|---|---|---|---|
| `$AWS_DEFAULT_REGION` | `us-east-1` | I + II | Scope (compose lo lee, Taskfile lo propaga). |
| `$AWS_REGION` | `us-east-1` | II | Alias para el backend S3 de Terraform (`-backend-config="region=..."` en #4.2). |
| `$AWS_PROFILE` | `default` | I + II | Credenciales AWS (compose monta `~/.aws:ro`). |
| `$PROJECT` | `ml-training` | II | Prefijo de todos los recursos AWS. |
| `$ACCOUNT_ID` | 12 dígitos | II | tfstate bucket, ECR URIs, role ARNs. |
| `$ACCOUNT_SUFFIX` | 7 dígitos | II | Sufijo de buckets, evita colisión cross-account. Coincide con `scripts/aws-suffix.sh` (fuente única). |

> **Nota** — En Tramo II, estas variables las usan los bloques `bash`
> manuales (Partes 1-2). Los Taskfiles (`tasks/*.yml`) las recalculan
> internamente con `aws sts get-caller-identity`; no las heredan del shell.

> **Gotchas Capítulo 3 (prereqs host)**:
> - **Tooling**: `docker --version && task --version && terraform version && aws --version` debe imprimir las 4 versiones sin error.
> - **Variables**: `echo "$AWS_REGION $AWS_DEFAULT_REGION $PROJECT $ACCOUNT_ID $ACCOUNT_SUFFIX"` debe imprimir `us-east-1 us-east-1 ml-training <12 digitos> <7 digitos>` (ninguno vacío). En terminal nueva, `source scripts/prod.env` — la guía asume "una vez por terminal".
> - **Host**: WSL sin systemd o Docker Desktop sin WSL integration → `docker` falla con `Cannot connect to the Docker daemon`.
> - **Vars**: si `aws sts get-caller-identity` falla, `AWS_PROFILE` no está seteado o las credenciales expiraron → re-correr `aws configure` o renovar token SSO.

---

## Capítulo 4 · Entorno local desde cero

> **Objetivo del capítulo** — Tomar un repo con sólo el código fuente
> (`src/`, `main.py`, `scripts/`, `requirements.txt`) y montar a su
> alrededor toda la maquinaria de Docker, Compose, Task y `.env` necesaria
> para correr un smoke training de ~1 minuto contra MLflow y S3 sandbox.
> Al final del capítulo tenés un loop de desarrollo cerrado en tu laptop,
> sin haber tocado un solo recurso de AWS más allá de dos buckets S3.

### 4.1 Punto de partida

Antes de empezar, confirmá que el repo tiene los cuatro artefactos
mínimos sobre los que vamos a construir el resto. Todo lo demás
(Dockerfile, compose, Taskfile, `.env`) se genera en este capítulo:

```bash
ls -1 src main.py requirements.txt scripts/prepare_data.py
# src
# main.py
# requirements.txt
# scripts/prepare_data.py
```

Si alguno falta, parate acá: ese material viene del repo y la guía no
intenta reconstruirlo.

### 4.2 Layout objetivo

Para anclar mentalmente lo que sigue, este es el árbol que vas a tener al
cerrar el capítulo. En **negrita conceptual** lo que se construye aquí
(todo lo que aparece con un `(#4.x)` al costado); el resto es código
preexistente del repo o material que llega recién en Tramo II.

```
ml_training/
├── src/                          (existente)  código del trainer
├── main.py                       (existente)  CLI entrypoint
├── requirements.txt              (existente)  deps Python (runtime trainer)
├── requirements-dev.txt          (existente)  deps Python dev (ruff, pytest)
├── pyproject.toml                (existente)  config ruff + project metadata
├── scripts/prepare_data.py       (existente)  data split
├── Dockerfile                    (#4.4)       imagen del trainer
├── .dockerignore                 (#4.3)       qué NO va al build
├── docker/
│   ├── mlflow/Dockerfile         (#4.5.1)     MLflow + psycopg2 + boto3
│   ├── nginx-reports.conf        (#4.5.2)     nginx static (Tramo I local)
│   └── reports/                  (Tramo II)   imagen ECS reports (S3-sync + nginx)
│       ├── Dockerfile                          nginx:1.27 (Debian) + AWS CLI v2 + dumb-init
│       ├── nginx.conf                          config para servir /reports y /artifacts
│       └── entrypoint.sh                       aws s3 sync inicial + nginx -g daemon off
├── docker-compose.yml            (#4.5.3)     postgres + mlflow + reports + trainer
├── docker-compose.override.yml.example  template para apuntar el trainer local a MLflow prod
├── Taskfile.yml                  (#4.6)       tasks locales (atajos AWS se agregan en #4.1.8)
├── tasks/local.yml               (#4.6.2)     helper para buckets sandbox
├── .env.example                  (#4.7)       plantilla de variables
└── .env                          (#4.7)       tu copia con buckets reales
```

### 4.3 `.dockerignore`

El Dockerfile usa `COPY` selectivo (sólo `src/`, `scripts/`, `main.py` y
`requirements.txt`), pero eso sirve de poco sin recortar antes el **build
context**: la carpeta entera que Docker empaqueta y envía al daemon. Sin
`.dockerignore` ese paquete arrastra `.git/` (cientos de MB), los Excel de
`data/`, los `artifacts/`/`reports/` de corridas previas, el legacy `mlruns/`
y los caches Python — y builds de segundos pasan a minutos.

Crear `.dockerignore` en la raíz:

```gitignore
# =============================================================================
# .dockerignore — qué NO entra al build context que se envia al daemon
# =============================================================================
# Filosofia: el Dockerfile usa COPY selectivo (solo src/, scripts/, main.py,
# requirements.txt). Este archivo acelera el envio al daemon ignorando todo
# lo que no se va a usar nunca durante el build.

# ── Git ──────────────────────────────────────────────────────────────────
.git/
.gitignore
.gitattributes

# ── Datos locales (se montan como volumen en runtime, no van en la imagen)
data/

# ── Salidas (se generan en runtime, montadas como volumen) ───────────────
artifacts/
logs/
reports/

# ── Legacy del modo file:// (ADR-001) ────────────────────────────────────
mlruns/

# ── Notebooks y experimentacion ──────────────────────────────────────────
notebooks/

# ── Subproyectos que NO entran al contexto raiz (trainer/api) ─────────────
# La UI buildea con su propio contexto (ui/), e infra/ es solo Terraform.
# La API SI necesita api/ + src/ (no excluir), pero no ui/ ni infra/.
ui/
infra/

# ── Cache Python ─────────────────────────────────────────────────────────
__pycache__/
**/__pycache__/
*.py[cod]
*.pyo

# ── Tests y caches de tooling (no van a la imagen de runtime) ────────────
tests/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# ── Entornos virtuales ───────────────────────────────────────────────────
.venv/
venv/
env/

# ── IDE ──────────────────────────────────────────────────────────────────
.vscode/
.idea/

# ── Documentacion (no necesaria en runtime; el build context vuela menos) ─
docs/
*.md
LICENSE

# ── Meta de build / orquestacion del HOST (el container no las necesita) ──
Taskfile.yml
Taskfile.*.yml
docker-compose.yml
docker-compose.*.yml
# Excluye el contenido de docker/ (mlflow/, nginx-reports.conf: solo host/compose)
# salvo docker/reports/, cuyo Dockerfile hace COPY de nginx.conf + entrypoint.sh
# desde el build context. Glob `docker/*` + re-inclusion: el padre docker/ queda
# "caminable", requisito para que BuildKit pueda re-incluir el subdirectorio.
docker/*
!docker/reports
Dockerfile
.dockerignore

# ── Dev-only deps (la imagen runtime usa solo requirements.txt) ──────────
requirements-dev.txt

# ── Variables de entorno locales (NUNCA en la imagen) ────────────────────
.env
.env.*
!.env.example

# ── Claude Code / agentes ────────────────────────────────────────────────
.claude/

# ── Sistema operativo ────────────────────────────────────────────────────
.DS_Store
Thumbs.db

# ── Archivos temporales ──────────────────────────────────────────────────
*.log
*.tmp
*.bak
```

> **Nota** — Las secciones críticas son **"Meta de build"**, **"Variables de
> entorno locales"** y **`.claude/`**: sin ellas horneás en la imagen el
> `Taskfile.yml`, el `docker-compose.yml`, tu `.env` con los buckets reales y
> la config de agentes de `.claude/` — inútil en runtime y, en el caso del
> `.env`, riesgo de filtración de secretos. `!.env.example` re-incluye la
> plantilla pública, que sí queremos que viaje con el repo.

**Verificación**

```bash
# Tras crearlo, una build dry-run sólo debería transferir KBs, no MBs
docker build --progress=plain --no-cache -t ml-training:dryrun . 2>&1 | head -5
# transferring context: ...kB     ← debe ser kB, no MB
```

> *Para verificar / re-ejecutar esta sección, ver #4.A.*

### 4.4 `Dockerfile`

La imagen del trainer es **multi-stage** para separar lo que compila
dependencias de lo que corre en producción. El **stage `builder`** trae
`build-essential` y compila las wheels de paquetes nativos (`lightgbm`,
`xgboost`, `psycopg2-binary`); el **stage `runtime`** arranca limpio, instala
esas wheels y descarta el toolchain. La imagen final pesa ~1.2 GB (vs ~2 GB
single-stage), sin compiladores que amplíen la superficie de ataque, y los
rebuilds hacen cache hit en el stage costoso mientras `requirements.txt` no
cambie.

#### 4.4.1 Stage 1 — builder

```Dockerfile
# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.13.1-slim-bookworm

FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./

# Cache mount de BuildKit: el pip cache persiste entre builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels -r requirements.txt
```

> **Nota** — La directiva `# syntax=docker/dockerfile:1.7` habilita la sintaxis
> extendida de BuildKit; sin ella el `RUN --mount=type=cache,target=/root/.cache/pip`
> se ignora en silencio y cada build vuelve a bajar las wheels desde PyPI
> (segundos → minutos).

#### 4.4.2 Stage 2 — runtime

```Dockerfile
FROM python:${PYTHON_VERSION} AS runtime

ARG GIT_SHA=unknown
ARG VERSION=dev
# created/timestamp NO va aqui: se inyecta como --label en `docker build` para
# no invalidar la cache ni cambiar el digest de capas en cada rebuild del commit.
LABEL org.opencontainers.image.title="ml-training" \
      org.opencontainers.image.description="Random Forest training pipeline" \
      org.opencontainers.image.source="https://github.com/abantodca/ml_training" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates tini git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 mluser \
    && useradd  --system --uid 1001 --gid mluser --home ${APP_HOME} mluser

WORKDIR ${APP_HOME}

COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Orden de COPY: de mejor cache (cambia poco) a peor cache (cambia más)
COPY --chown=mluser:mluser src/    ./src/
COPY --chown=mluser:mluser scripts/ ./scripts/
COPY --chown=mluser:mluser main.py  ./

# Directorios que init_dirs() asume (idempotente)
RUN mkdir -p data/training logs artifacts reports \
    && chown -R mluser:mluser ${APP_HOME}

# MLflow llama `getpass.getuser()` para taggear `mlflow.user` en cada run; sin
# USER/LOGNAME en el environment (el `USER mluser` de abajo NO setea la env var)
# emite "No username set in the environment" varias veces por modelo. Fijarlo
# explicito silencia el warning y deja un tag legible en la UI de MLflow. Vale
# tanto en compose como en AWS Batch (misma imagen).
ENV USER=mluser
USER mluser
STOPSIGNAL SIGTERM

# tini propaga SIGTERM correctamente cuando Batch mata el job
ENTRYPOINT ["/usr/bin/tini", "--", "python", "main.py"]
CMD ["--varieties", "POP", "--tuning", "smoke"]
```

> **Nota — por qué `git` viaja en el runtime.** `mlflow.utils.git_utils` y
> nuestro `collect_run_metadata` lo invocan en entrenamiento para taggear el
> run con el SHA del commit. Sin el binario todos los runs salen con
> `git_commit=unknown` y perdés la trazabilidad **modelo → commit**, la primera
> pregunta cuando un campeón regresione.

> **Nota — el uid 1001 no es casualidad.** `USER mluser` (uid 1001) está
> alineado con los bind-mount targets que crea `task _ensure_dirs` (#4.6.1) en
> el host. Si los dejás crear al vuelo, Docker los hace con uid `root` y el
> primer write del container falla con `Permission denied`.

**Verificación**

```bash
docker build -t ml-training:local .
docker images ml-training:local
# REPOSITORY    TAG     IMAGE ID       SIZE
# ml-training   local   <id>           ~1.2 GB
```

> *Para verificar / re-ejecutar esta sección, ver #4.A.*

> **Auditoría de CVEs** — la base `python:3.13.1-slim-bookworm` y las deps
> (`xgboost`, `lightgbm`, `psycopg2-binary`, `pandas`) heredan CVEs. Tramo
> Local: warn-only/opcional, `trivy image ml-training:local --severity HIGH,CRITICAL --ignore-unfixed`.
> Tramo II (push a ECR): scan-on-push del ECR + `trivy` en CI (`task ecr:scan`
> pendiente). No la corras aún: la imagen todavía no existe.

### 4.5 Servicios `docker/` y `docker-compose.yml`

El stack local se compone de **seis servicios**: `postgres` y `mlflow` corren
en background como tracking backend; `reports` es un nginx estático que expone
los HTML y joblibs; `api` (FastAPI) sirve los modelos y persiste pronósticos, y
`ui` (Streamlit) los consume; `trainer` es one-shot — vive lo que dura un
entrenamiento, invocado por `task train`. La tabla es el mapa rápido; las
subsecciones siguientes entran al detalle de cada componente `docker/`.

| Servicio | Imagen | Rol | Puerto host |
|---|---|---|---|
| `postgres` | `postgres:15-alpine` | Backend store de MLflow (metadata, Registry) + base `forecasts` | — (interno) |
| `mlflow` | Build de `docker/mlflow/Dockerfile` | Tracking server v3.12 + UI | `127.0.0.1:5000` |
| `reports` | `nginx:1.27-alpine` | Sirve `./reports/` y `./artifacts/` del host | `127.0.0.1:8080` |
| `api` | Build de `api/Dockerfile` (contexto raíz) | FastAPI: sirve modelos + persiste pronósticos | `127.0.0.1:8000` |
| `ui` | Build de `ui/Dockerfile` | Streamlit: dashboard que consume la API | `127.0.0.1:8501` |
| `trainer` | Build del `Dockerfile` raíz | One-shot `main.py` con args | — |

#### 4.5.1 `docker/mlflow/Dockerfile`

La imagen oficial de MLflow es **minimalista**: server + SQLite. Para un
backend real (Postgres) y artifact store en S3 faltan dos paquetes que
upstream no incluye: `psycopg2-binary` (sin él `--backend-store-uri
postgresql://...` falla con `ModuleNotFoundError`) y `boto3` (sin él MLflow no
firma requests a S3). De ahí esta imagen custom — el Dockerfile más corto del
repo:

```Dockerfile
FROM ghcr.io/mlflow/mlflow:v3.12.0

RUN pip install --no-cache-dir \
        psycopg2-binary==2.9.9 \
        boto3==1.38.0

LABEL org.opencontainers.image.title="mlflow-with-pg-s3" \
      org.opencontainers.image.description="MLflow 3.12.0 + psycopg2-binary + boto3" \
      org.opencontainers.image.source="https://github.com/abantodca/ml_training" \
      org.opencontainers.image.base.name="ghcr.io/mlflow/mlflow:v3.12.0"
```

#### 4.5.2 `docker/nginx-reports.conf`

Convierte un nginx vanilla en servidor de archivos estáticos para `reports/` y
`artifacts/` del host. Dos detalles: `autoindex on` deja navegables los
directorios (sin él, `/reports/` tira 403); y `Content-Disposition: attachment`
sobre `.joblib`/`.xlsx`/`.json` fuerza descarga en vez de render inline.

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;

    autoindex on;
    autoindex_exact_size off;
    autoindex_localtime on;

    location / {
        try_files $uri $uri/ =404;
    }

    location ~ \.(joblib|xlsx|json)$ {
        add_header Content-Disposition 'attachment';
    }
}
```

#### 4.5.3 `docker-compose.yml`

Seis servicios (`postgres`, `mlflow`, `reports`, `api`, `ui`, `trainer`),
con decisiones de robustez que viven como comentarios in-line en el YAML:
logging con rotación, healthchecks reales (no nominales), `--allowed-hosts`
más artifacts en modo proxy (`--artifacts-destination --serve-artifacts`)
de MLflow 3.x, credenciales AWS por bind-mount, y loopback bind para los
`ports:`. La **API** y la **UI** comparten el `src/` raíz (contexto de build
= raíz del repo) y la base `forecasts` del mismo Postgres (creada por
`docker/postgres/initdb`); la UI arranca sólo cuando la API está healthy.

```yaml
# Logging con rotación. Sin esto json-file crece sin bound.
x-logging: &default-logging
  driver: json-file
  options:
    max-size: "10m"
    max-file: "3"

services:
  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-mlflow}
    volumes:
      - pg-data:/var/lib/postgresql/data
      # initdb crea la base `forecasts` (API) junto a `mlflow`. Solo corre al
      # inicializar el volumen; si ya existe pg-data, no re-ejecuta (la API
      # igual la auto-crea si faltara — ensure_database).
      - ./docker/postgres/initdb:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow -d mlflow"]
      interval: 5s
      retries: 10
    logging: *default-logging

  mlflow:
    build:
      context: .
      dockerfile: docker/mlflow/Dockerfile
    restart: unless-stopped
    depends_on:
      postgres: { condition: service_healthy }
    environment:
      AWS_SHARED_CREDENTIALS_FILE: /aws/credentials
      AWS_CONFIG_FILE: /aws/config
      AWS_PROFILE: ${AWS_PROFILE:-default}
      AWS_DEFAULT_REGION: ${AWS_DEFAULT_REGION:-us-east-1}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-mlflow}
    volumes:
      - ~/.aws:/aws:ro
    # Artifacts en modo PROXY (recomendado): el server es el único que toca S3.
    # `--artifacts-destination` (en vez de `--default-artifact-root`) hace que
    # los runs nuevos tengan artifact_uri `mlflow-artifacts:/…` y que TODO el
    # IO de artifacts (subida del trainer, listado/descarga de la UI y del API)
    # pase por el server — sin que cada cliente necesite creds S3. Evita el
    # "Loading artifact failed / INTERNAL_ERROR" del modo mixto anterior.
    # Requiere que el server alcance S3: revisar AWS_DEFAULT_REGION (= región
    # real del bucket) y las creds montadas en ~/.aws.
    command: >
      sh -c "mlflow server
      --host 0.0.0.0 --port 5000
      --allowed-hosts mlflow,mlflow:*,localhost,localhost:*,127.0.0.1,127.0.0.1:*
      --backend-store-uri postgresql://mlflow:$${POSTGRES_PASSWORD}@postgres:5432/mlflow
      --artifacts-destination s3://${S3_MLFLOW_BUCKET:?Set S3_MLFLOW_BUCKET in .env}/artifacts
      --serve-artifacts"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health',timeout=3).status==200 else 1)"]
      interval: 10s
      timeout: 5s
      retries: 12
      start_period: 30s
    ports:
      - "127.0.0.1:5000:5000"
    logging: *default-logging

  reports:
    image: nginx:1.27-alpine
    restart: unless-stopped
    volumes:
      - ./reports:/usr/share/nginx/html/reports:ro
      - ./artifacts:/usr/share/nginx/html/artifacts:ro
      - ./docker/nginx-reports.conf:/etc/nginx/conf.d/default.conf:ro
    ports:
      - "127.0.0.1:8080:80"
    logging: *default-logging

  # ── API (FastAPI) ────────────────────────────────────────────────────────
  # Sirve los modelos rnd-forest-* registrados en MLflow y persiste pronósticos
  # en la base `forecasts`. Comparte el `src/` raíz vía contexto de build = raíz.
  api:
    build:
      context: .
      dockerfile: api/Dockerfile
    image: ml-training-api:local
    restart: unless-stopped
    depends_on:
      postgres: { condition: service_healthy }
      mlflow:   { condition: service_healthy }
    environment:
      DATABASE_URL: postgresql://mlflow:${POSTGRES_PASSWORD:-mlflow}@postgres:5432/forecasts
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MLFLOW_PRELOAD_MODELS: ${MLFLOW_PRELOAD_MODELS:-false}
      # Mismo prefijo que registra el trainer (src/config.py MODEL_REGISTRY_PREFIX).
      EXPERIMENT_PREFIX: ${MODEL_REGISTRY_PREFIX:-rnd-forest-}
      CORS_ORIGINS: ${CORS_ORIGINS:-http://localhost:8501}
      LOG_LEVEL: ${LOG_LEVEL:-info}
      # boto3 (vía cliente MLflow) descarga los artifacts desde S3 real.
      AWS_SHARED_CREDENTIALS_FILE: /aws/credentials
      AWS_CONFIG_FILE: /aws/config
      AWS_PROFILE: ${AWS_PROFILE:-default}
      AWS_DEFAULT_REGION: ${AWS_DEFAULT_REGION:-us-east-1}
    volumes:
      - ~/.aws:/aws:ro
    ports:
      - "127.0.0.1:8000:8000"
    # Healthcheck explicito: el gate `service_healthy` del que cuelga `ui`.
    # /api/health responde 200 aunque MLflow aun no tenga modelos (lazy-load),
    # asi la UI arranca apenas la API esta lista. start_period cubre init_db.
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/api/health"]
      interval: 10s
      timeout: 5s
      retries: 6
      start_period: 40s
    logging: *default-logging

  # ── UI (Streamlit) ───────────────────────────────────────────────────────
  # Arranca SOLO cuando la API esta healthy (depends_on -> service_healthy).
  ui:
    build:
      context: ui
      dockerfile: Dockerfile
    image: ml-training-ui:local
    restart: unless-stopped
    depends_on:
      api: { condition: service_healthy }
    environment:
      API_URL: ${UI_API_URL:-http://api:8000}
      LOG_LEVEL: ${LOG_LEVEL:-info}
    # Bind-mount de DEV (read-only): el código del host queda siempre vivo, así
    # editar una vista se refleja al instante (Streamlit runOnSave) sin esperar
    # un rebuild. Es seguro: las dependencias viven en /opt/venv (no en /app),
    # así que el mount NO las pisa. Para una imagen 100% limpia (sin host),
    # comentá esta línea y reconstruí con `task build`.
    volumes:
      - ./ui:/app:ro
    ports:
      - "127.0.0.1:8501:8501"
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8501/_stcore/health"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 25s
    logging: *default-logging

  trainer:
    build: .
    depends_on:
      mlflow: { condition: service_healthy }
    environment:
      MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI:-http://mlflow:5000}
      AWS_SHARED_CREDENTIALS_FILE: /aws/credentials
      AWS_CONFIG_FILE: /aws/config
      AWS_PROFILE: ${AWS_PROFILE:-default}
      AWS_DEFAULT_REGION: ${AWS_DEFAULT_REGION:-us-east-1}
      S3_ARTIFACTS_BUCKET: ${S3_ARTIFACTS_BUCKET:?Set S3_ARTIFACTS_BUCKET in .env}
      S3_ARTIFACTS_PREFIX: artifacts
      S3_REPORTS_PREFIX: reports
      # Hydrate paritario con AWS Batch (opcional en Tramo Local).
      # Si estas dos estan seteadas, el trainer baja el Excel ACUMULADO
      # desde S3 al boot y corre split_workbook() — replica el flujo
      # productivo end-to-end. Si quedan vacias, lee ./data del bind-mount.
      # El default de S3_DATA_KEY coincide con el del Lambda dispatcher
      # (infra/lambdas/dispatcher.py): el archivo acumulado, no el split.
      # Ver #4.13 (Proximo paso: Tramo II).
      S3_DATA_BUCKET: ${S3_DATA_BUCKET:-}
      S3_DATA_KEY: ${S3_DATA_KEY:-BD_HISTORICO_ACUMULADO.xlsx}
    volumes:
      - ~/.aws:/aws:ro
      - ./data:/app/data
      - ./logs:/app/logs
      - ./artifacts:/app/artifacts
      - ./reports:/app/reports
    mem_limit: ${TRAINER_MEM:-8g}
    cpus: ${TRAINER_CPUS:-4}
    command: ["--varieties", "${VARIETIES:-POP}", "--tuning", "${TUNING:-smoke}"]
    logging: *default-logging

volumes:
  pg-data:
```

> **Nota — la puerta al MLflow productivo.** Exponer `MLFLOW_TRACKING_URI` como
> shell var (default `http://mlflow:5000`) habilita entrenar desde tu laptop
> loggeando contra el MLflow productivo (`export MLFLOW_TRACKING_URI=http://<ALB-DNS>`):
> útil para validar features sin contaminar el Postgres local, o reproducir un
> run de producción que regresionó. Procedimiento completo en
> `docker-compose.override.yml.example`.

> *Para verificar / re-ejecutar esta sección, ver #4.A.*

#### 4.5.4 `docker/postgres/initdb/01-create-forecasts.sql`

El servicio `postgres` (4.5.3) monta `./docker/postgres/initdb` en
`/docker-entrypoint-initdb.d`. La imagen oficial de Postgres corre los scripts
de ese directorio **una sola vez**, al inicializar el volumen `pg-data`. Este
crea la base `forecasts` (de la API) junto a `mlflow`, idempotente. En
producción NO se usa: la API auto-crea `forecasts` en el RDS de MLflow en su
primer arranque (`app/models/database.ensure_database`).

> 📂 **Pegar este bloque en**: `docker/postgres/initdb/01-create-forecasts.sql`

```sql
-- Crea la base `forecasts` (de la API) junto a la base `mlflow` en el MISMO
-- Postgres local. Idempotente: solo crea si no existe. El entrypoint de la
-- imagen postgres ejecuta este script una vez, al inicializar el volumen.
--
-- En producción NO se usa este script: la API auto-crea la base `forecasts`
-- en el RDS de MLflow en su primer arranque (app/models/database.ensure_database).
SELECT 'CREATE DATABASE forecasts'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'forecasts')\gexec
```

### 4.6 `Taskfile.yml`

[Task](https://taskfile.dev) es el orquestador: Makefile moderno en
YAML, con `includes:` y variables tipadas. Tres responsabilidades:

1. **Lifecycle Docker local** — `task build` / `task down`.
2. **Trainer parametrizado** — `task train VARIETIES=... TUNING=... PARALLEL=...`, más `task eda` y `task data:split`.
3. **AWS namespaced** — `infra:`, `ecr:`, `batch:`, `ops:` (archivos separados bajo `tasks/`, cargados vía `includes:`) + atajos high-level a nivel root (`deploy`, `wake`, `sleep`, `teardown`, `destroy`, `nuke`, `status`).

En Tramo I construimos las dos primeras + el namespace `local:`
(buckets sandbox). Los namespaces AWS productivos se agregan en
**Tramo II Parte 4 #4.1.8**, cuando ya existen los `tasks/*.yml` que
los respaldan — si los declarás antes, `task --list` falla con
`open ./tasks/X.yml: no such file`.

#### 4.6.1 Raíz `Taskfile.yml`

```yaml
version: "3"

# `.env` es opcional. Si no existe, los defaults de config.py aplican.
dotenv: [ ".env" ]

# Tramo I: solo importamos el namespace `local:` (buckets sandbox S3).
# Los namespaces AWS se agregan en Tramo II Parte 4 #4.1.8.
includes:
  local:
    taskfile: ./tasks/local.yml
    vars:
      PROJECT: '{{.PROJECT}}'
      REGION: '{{.REGION}}'

vars:
  # TUNING se define a nivel-tarea (default "prod_xl"), NO como var global:
  # un default global pisaria el del include batch: (ver tasks/batch.yml).
  VARIETIES: '{{.VARIETIES | default "POP"}}'
  PARALLEL:  '{{.PARALLEL  | default "1"}}'
  PROJECT:   '{{.PROJECT   | default "ml-training"}}'
  REGION:    '{{.AWS_DEFAULT_REGION | default "us-east-1"}}'
  HOST_UID:  { sh: id -u }
  HOST_GID:  { sh: id -g }
  # Patron reutilizado: correr un modulo Python en el container trainer sin
  # levantar mlflow/postgres. Lo usan data:split y eda.
  # --user fija el uid/gid del host: lo que el contenedor escriba al bind-mount
  # (./data, ./reports) sale con tu ownership, no como mluser (uid 1001) — evita
  # el PermissionError al sobrescribir DB-HISTORICA.xlsx. MPLCONFIGDIR=/tmp porque
  # con --user el $HOME de la imagen no es escribible por el uid del host (matplotlib en eda).
  DC_PY: docker compose run --rm --no-deps --user "{{.HOST_UID}}:{{.HOST_GID}}" -e MPLCONFIGDIR=/tmp --entrypoint python trainer

tasks:

  default:
    desc: "Lista comandos del pipeline + ejemplos TUNING/VARIETIES"
    silent: true
    cmds:
      - |
        cat <<'EOF'

        ml_training — predice productividad de cosecha (KG/JR_H)
        ─────────────────────────────────────────────────────────

        Pipeline (correr en orden):
          task build              1ª vez o tras cambiar codigo/Dockerfile (stack completo)
          task up                 levanta db + mlflow + reports + api + ui (sin rebuild)
          task data:split         genera data/training/DB-HISTORICA.xlsx
          task data:upload        (opcional) sube el Excel acumulado a S3 (hydrate paritario con Batch)
          task eda VARIETIES=POP  (opcional) analisis exploratorio
          task train              entrena + genera HTML en reports/
          task down               apaga servicios al terminar

        Ejemplos de entrenamiento (TUNING × VARIETIES × PARALLEL):
          task train VARIETIES=POP                   overnight (DEFAULT) ~4-6 h prod_xl
          task train VARIETIES=POP TUNING=smoke      sanity check    ~1 min
          task train VARIETIES=POP TUNING=dev        baseline        ~20 min
          task train VARIETIES=POP TUNING=prod       produccion      ~2 h
          task train VARIETIES=POP,VENTURA           multiples variedades
          task train VARIETIES=all PARALLEL=3        todas, 3 en paralelo

        Variables (override por CLI, formato VAR=valor):
          VARIETIES   POP (default) | POP,VENTURA,... | all
          TUNING      smoke ~1m | dev ~20m | prod ~2h | prod_xl ~4-6h (default)
          PARALLEL    1 (default) | N variedades en paralelo
          SEED        42 (default) | reproducibilidad

        URLs (servicios up):
          http://localhost:8501             UI Streamlit (dashboard gerencial)
          http://localhost:8000/docs        API FastAPI (Swagger)
          http://localhost:5000             MLflow UI (tracking + runs)
          http://localhost:8080/reports/    dashboards HTML por variedad
          http://localhost:8080/artifacts/  joblib + best_params

        Mas info:  task --list   GUIA_MLOPS_AWS_V2.md
        EOF

  build:
    desc: "1ª vez o al cambiar codigo/Dockerfile: rebuild imagenes (trainer+api+ui) + levanta TODO el stack"
    cmds:
      - task: _ensure_dirs
      - docker compose build trainer api ui
      - docker compose up -d postgres mlflow reports api ui
      - task: _print_urls

  up:
    desc: "Levanta el stack local completo (db + mlflow + reports + api + ui) sin rebuild"
    cmds:
      - task: _ensure_dirs
      - docker compose up -d postgres mlflow reports api ui
      - task: _print_urls

  data:split:
    desc: "Paso 1: genera data/training/DB-HISTORICA.xlsx desde el Excel historico"
    cmds:
      - >-
        {{.DC_PY}} -m scripts.prepare_data
        --input  data/BD_HISTORICO_ACUMULADO.xlsx
        --output data/training/DB-HISTORICA.xlsx
        --min-rows 100

  data:upload:
    desc: "Paso 1b (opcional): sube el Excel acumulado a s3://$S3_DATA_BUCKET/$S3_DATA_KEY. Solo si .env activa el hydrate (replica el flujo prod/Batch); si no, `train` lee ./data y este paso sobra."
    cmds:
      # El trainer (main.py::_hydrate_data_from_s3) baja este Excel al boot SOLO
      # si S3_DATA_BUCKET+S3_DATA_KEY estan en .env. Sin el objeto en S3 -> 404 y
      # `train` aborta. Esta task lo sube para que el hydrate funcione local igual
      # que en AWS Batch. Ver #4.13.
      - |
        : "${S3_DATA_BUCKET:?S3_DATA_BUCKET no esta en .env -> hydrate S3 desactivado; nada que subir (train leeria ./data)}"
        KEY="${S3_DATA_KEY:-BD_HISTORICO_ACUMULADO.xlsx}"
        if [ ! -f data/BD_HISTORICO_ACUMULADO.xlsx ]; then
          echo "ERROR: data/BD_HISTORICO_ACUMULADO.xlsx no existe localmente."
          exit 1
        fi
        aws s3 cp data/BD_HISTORICO_ACUMULADO.xlsx "s3://${S3_DATA_BUCKET}/${KEY}" --region "{{.REGION}}"

  eda:
    desc: "Paso 2 (opcional): EDA estadistico standalone. Args: VARIETIES=POP"
    cmds:
      - "{{.DC_PY}} -m src.diagnostics.eda --variety {{.VARIETIES}}"

  train:
    desc: "Paso 3: entrena + genera HTML estatico en reports/. Vars: VARIETIES TUNING PARALLEL SEED"
    deps: [_up]
    vars:
      GIT_SHA:   { sh: bash scripts/metadata.sh git-sha }
      GIT_DIRTY: { sh: bash scripts/metadata.sh git-dirty }
      DATA_SHA:  { sh: bash scripts/metadata.sh data-sha }
      SEED: '{{.SEED | default "42"}}'
    cmds:
      - |
        if [ "{{.DATA_SHA}}" = "missing" ]; then
          echo "ERROR: data/training/DB-HISTORICA.xlsx no existe. Correr 'task data:split' primero."
          exit 1
        fi
      # --user/MPLCONFIGDIR: misma razon que DC_PY (ver vars). Sin --user el
      # trainer corre como mluser (uid 1001) y no puede escribir los bind-mounts
      # (./logs ./artifacts ./reports) que data:split dejo con tu ownership ->
      # PermissionError en logs/pipeline_run.log. MPLCONFIGDIR=/tmp porque con
      # --user el $HOME de la imagen no es escribible (matplotlib en los HTML).
      - >-
        docker compose run --rm
        --user "{{.HOST_UID}}:{{.HOST_GID}}"
        -e MPLCONFIGDIR=/tmp
        -e GIT_SHA={{.GIT_SHA}}
        -e GIT_DIRTY={{.GIT_DIRTY}}
        -e DATA_SHA={{.DATA_SHA}}
        -e SEED={{.SEED}}
        trainer
        --varieties {{.VARIETIES}}
        --tuning {{.TUNING}}
        --parallel-varieties {{.PARALLEL}}
      - task: _print_urls

  down:
    desc: "Detiene servicios. Preserva volumen Postgres"
    cmds:
      - docker compose down

  _up:
    internal: true
    cmds:
      - task: _ensure_dirs
      - docker compose up -d postgres mlflow reports
      - task: _print_urls

  _ensure_dirs:
    internal: true
    silent: true
    cmds:
      - mkdir -p artifacts reports logs data/training

  _print_urls:
    internal: true
    silent: true
    cmds:
      - |
        cat <<EOF

        ════════════════════════════════════════════════════════════════════
         Servicios LOCAL listos:
           UI (dashboard gerencial)   http://localhost:8501
           API (Swagger)              http://localhost:8000/docs
           MLflow runs                http://localhost:5000
           MLflow Model Registry      http://localhost:5000/#/models
           Reports (campeon HTML)     http://localhost:8080/reports/
           Artifacts                  http://localhost:8080/artifacts/
           S3 backend                 s3://${S3_MLFLOW_BUCKET}/
        ────────────────────────────────────────────────────────────────────
         Nota: en local NO hay :80 (eso es el ALB de produccion). Cada
         servicio expone su propio puerto. Entrena con: task train VARIETIES=POP TUNING=smoke
        ════════════════════════════════════════════════════════════════════
        EOF
```

> **Decisiones del Taskfile mínimo**:
> - **6 tasks públicas** (`build`, `up`, `data:split`, `eda`, `train`, `down`)
>   reflejan el flujo end-to-end. Lint, scans, logs o cleanup van por comando
>   directo (`ruff check …`, `docker compose logs -f …`, `docker compose down -v`):
>   una task por cosa de un solo uso es ruido.
> - **`default` justifica su lugar**: `task` sin args lista los cruces útiles de
>   TUNING × VARIETIES que `task --list` no comunica (~35 líneas, valor real),
>   a diferencia del `help`/`doctor`/`scan` que borramos.
> - **`_up` es interno**: `train` lo invoca via `deps: [_up]` para garantizar
>   mlflow/postgres arriba aunque hayas hecho `task down`. No sale en `--list`.
> - **`data:split` y `eda` usan `--no-deps`** (`{{.DC_PY}}`): corren el trainer
>   aislado, sin mlflow/postgres — son scripts standalone.
> - **`train` genera el HTML**: `variety_runner.py` regenera `reports/index.html`
>   al final de cada variedad; no hace falta una task `reports:dashboard`.
> - **No hay job `test` en CI** porque no hay `tests/` (ver
>   [ADR-008](docs/adr/ADR-008-ci-sin-tests-todavia.md)): un task que pasa
>   trivialmente es deuda encubierta.

| Variable | Default | Override por CLI |
|---|---|---|
| `VARIETIES` | `POP` | `task train VARIETIES=POP,VENTURA` |
| `TUNING` | `prod_xl` | `task train TUNING=smoke` (`smoke` / `dev` / `prod` / `prod_xl`) |
| `PARALLEL` | `1` | `task train VARIETIES=all PARALLEL=3` |
| `SEED` | `42` (decisión Cap 2) | `task train SEED=1337` (reproducibilidad — el código del trainer la lee de `os.environ["SEED"]`) |

> **Vars auto-calculadas** (no se override, salen del entorno):
> `GIT_SHA` (`git rev-parse HEAD`), `GIT_DIRTY` (`git diff --quiet`),
> `DATA_SHA` (primeros 12 chars del `sha256sum` de
> `data/training/DB-HISTORICA.xlsx`). Se inyectan como env-vars al
> container y el trainer las loguea como tags MLflow — son el núcleo
> del **contrato del run** (Cap 1.5). Si `data/training/DB-HISTORICA.xlsx`
> no existe, el `task train` aborta antes de levantar el container con
> un mensaje pidiendo correr `task data:split`.

(Los perfiles `smoke` / `dev` / `prod` / `prod_xl` ya aparecen con tiempos en el output de `task` sin argumentos; los detalles de folds y CV viven en `src/config.py`.)

#### 4.6.2 `tasks/local.yml` (canónico, único)

**Qué hace.** Crea (o reusa, si ya existen) los dos buckets S3 sandbox
— `{project}-data-<suffix6>` y `{project}-artifacts-<suffix6>` — con
el mismo naming + hardening que usaría Terraform en producción
(versioning + AES256 + Public Access Block). Es idempotente: correrlo
dos veces no falla ni duplica recursos.

**Cómo se usa** (después de pegar el archivo abajo, lo invocamos por
primera vez en #4.8):

```bash
task local:ensure-buckets               # crea o reusa los 2 buckets
task local:bucket-name KIND=data        # imprime el nombre completo
task local:bucket-name KIND=artifacts
```

> **Dónde se invoca.** El bloque `includes: local:` ya está pegado en
> #4.6.1 (entre `dotenv:` y `vars:` del Taskfile raíz) — no hay que
> agregar nada más. Este archivo se define **una sola vez acá**; en
> Tramo II Parte 4 #4.1.8 solo se lo referencia para evitar drift.

```yaml
# =============================================================================
# tasks/local.yml  -  Helpers para desarrollo local que toca AWS
# =============================================================================
# Incluido por Taskfile.yml raiz con namespace "local:".
#
# USO TIPICO:
#   task local:ensure-buckets        crea data + artifacts S3 si no existen (idempotente)
#                                    Reusa los nombres de prod ({project}-data-<suffix>),
#                                    asi un sync local puede compartir bucket con AWS Batch.
#   task local:bucket-name KIND=X    imprime nombre completo (KIND=data|artifacts)
# =============================================================================

version: "3"

# SUFFIX (ultimos 7 digitos del Account ID) se calcula via scripts/aws-suffix.sh.
# NO se define a nivel archivo a proposito: si estuviera aqui, Task lo evaluaria
# al cargar el namespace y dispararia `aws sts get-caller-identity` incluso para
# `task` (sin args) o `task --list`. Por-tarea => solo corre cuando se invoca.

tasks:

  ensure-buckets:
    desc: "Crea S3 buckets data + artifacts si no existen (idempotente). Misma cuenta+region que prod."
    silent: true
    vars:
      SUFFIX: { sh: bash scripts/aws-suffix.sh }
    cmds:
      - for: [data, artifacts]
        task: _ensure-bucket
        vars: { NAME: '{{.PROJECT}}-{{.ITEM}}-{{.SUFFIX}}' }
      - |
        cat <<EOF

        Listo. Para que el trainer local sincronice a estos buckets, exporta:
          export S3_DATA_BUCKET={{.PROJECT}}-data-{{.SUFFIX}}
          export S3_ARTIFACTS_BUCKET={{.PROJECT}}-artifacts-{{.SUFFIX}}
        EOF

  bucket-name:
    desc: "Imprime el nombre del bucket. Var: KIND=data|artifacts (REQ)"
    silent: true
    requires: { vars: [KIND] }
    vars:
      SUFFIX: { sh: bash scripts/aws-suffix.sh }
    cmds:
      - echo "{{.PROJECT}}-{{.KIND}}-{{.SUFFIX}}"

  _ensure-bucket:
    internal: true
    silent: true
    requires: { vars: [NAME] }
    cmds:
      - bash scripts/ensure-s3-bucket.sh "{{.NAME}}" "{{.REGION}}"
```

> **Detalles de diseño** (informativo):
>
> - **Reusa naming de prod**: si Tramo II ya aplicó `module.storage`,
>   `head-bucket` devuelve `EXISTE (reuso)` — mismo bucket que el productivo,
>   sin migración entre tramos.
> - **`if us-east-1`**: S3 rechaza `--create-bucket-configuration` en su región
>   default; el `if` abstrae esa aspereza.
> - **`local:` vs `aws:`**: `aws:` orquesta el stack productivo; `local:` agrupa
>   helpers que corren en tu máquina pero tocan AWS (`ensure-buckets`, futuros
>   `download-latest-model`…). La separación evita que `aws:` se llene de dev-utils.

**Verificación**

```bash
task --list
# Debe mostrar las 6 publicas: build, up, data:split, eda, train, down,
# mas las 2 del namespace local: ensure-buckets, bucket-name.
# Las internas (_up, _ensure_dirs, _print_urls) NO aparecen en --list.
```

Si `task --list` falla con `failed to read taskfile` o un error YAML
genérico, la causa casi siempre es **indentación con tabs**. Task es
estricto: el YAML spec exige espacios, y un solo tab en cualquier nivel
rompe el parser con un mensaje que no apunta a la línea ofensora.
Solución rápida: `expand -t 2 Taskfile.yml > /tmp/t && mv /tmp/t
Taskfile.yml` (y lo mismo con `tasks/local.yml`).

> *Para verificar / re-ejecutar esta sección, ver #4.A.*

#### 4.6.3 `scripts/` — helpers de los Taskfiles locales

Tres scripts cortos que los Taskfiles invocan via `sh:` / `cmds:`. Viven en
`scripts/` (no en `tasks/lib/`) porque también los reusa el Tramo II:
`metadata.sh` lo llama `task train` (tags del run MLflow); `aws-suffix.sh` y
`ensure-s3-bucket.sh` los usan `tasks/local.yml` y #4.8 — y más tarde el
bootstrap de tfstate (Parte 2) reusa los mismos dos.

> 📂 **Pegar este bloque en**: `scripts/metadata.sh`

```bash
#!/usr/bin/env bash
# Metadata del run de training. Cada subcomando imprime un valor en stdout
# y lo consume Taskfile.yml `train.vars` via `sh:`.
#
# Sin `set -e`: los comandos usan `||` para fallback ("unknown", "missing").

case "${1:-}" in
  git-sha)
    git rev-parse HEAD 2>/dev/null || echo unknown
    ;;
  git-dirty)
    git diff --quiet HEAD 2>/dev/null && echo false || echo true
    ;;
  data-sha)
    f=data/training/DB-HISTORICA.xlsx
    [ -f "$f" ] && sha256sum "$f" | cut -d' ' -f1 | cut -c1-12 || echo missing
    ;;
  *)
    echo "Usage: $0 {git-sha|git-dirty|data-sha}" >&2
    exit 1
    ;;
esac
```

> 📂 **Pegar este bloque en**: `scripts/aws-suffix.sh`

```bash
#!/usr/bin/env bash
# Imprime los ultimos 7 digitos del AWS Account ID (sufijo de bucket).
# Consumido por tasks/local.yml `vars.SUFFIX.sh`.
#
# `${acct#?????}` (POSIX) quita los primeros 5 chars del Account ID de 12.
# Evita `tail -c 7` que en Windows-coreutils (scoop) parsea el `7` como filename.
set -euo pipefail

acct=$(aws sts get-caller-identity --query Account --output text)
echo "${acct#?????}"
```

> 📂 **Pegar este bloque en**: `scripts/ensure-s3-bucket.sh`

```bash
#!/usr/bin/env bash
# Crea bucket S3 si no existe + aplica hardening (versioning, AES256, no public).
# Idempotente en dos niveles:
#   1) Si el bucket no existe, lo crea.
#   2) Aplica versioning + encryption + public-access-block SIEMPRE (no solo al
#      crear). Los tres son PUT idempotentes; re-aplicarlos auto-corrige drift
#      si alguien tocó el bucket a mano sin esos settings.
#
# Uso: ensure-s3-bucket.sh <name> <region>
# Consumido por: tasks/local.yml `_ensure-bucket` y infra/bootstrap.sh (tfstate).
set -euo pipefail

name="${1:?falta <name>}"
region="${2:?falta <region>}"

if aws s3api head-bucket --bucket "$name" 2>/dev/null; then
  echo "  $name  EXISTE (reaplicando hardening)"
else
  echo "  $name  no existe -> creando..."
  # us-east-1 NO acepta --create-bucket-configuration (es la default; AWS lo rechaza)
  if [ "$region" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$name" --region "$region"
  else
    aws s3api create-bucket --bucket "$name" --region "$region" \
      --create-bucket-configuration "LocationConstraint=$region"
  fi
fi

# Hardening idempotente (mismas defaults que el modulo storage de prod).
aws s3api put-bucket-versioning --bucket "$name" \
  --versioning-configuration Status=Enabled

aws s3api put-bucket-encryption --bucket "$name" \
  --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

aws s3api put-public-access-block --bucket "$name" \
  --public-access-block-configuration \
  'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'

echo "  $name  OK (versioning + AES256 + no public)"
```

> Los tres son ejecutables: tras pegarlos, `chmod +x scripts/*.sh`.

### 4.7 `.env.example` y `.env`

> [!NOTE]
> Este `.env.example` es **self-contained para Tramo I**. Las **únicas
> variables obligatorias** son `S3_MLFLOW_BUCKET` y `S3_ARTIFACTS_BUCKET` (los
> buckets sandbox de #4.8 vía `task local:ensure-buckets`); el resto tiene
> defaults sanos en `config.py` / `docker-compose.yml`, y las líneas comentadas
> son overrides opcionales.
>
> En Tramo II se reusa **tal cual**: misma imagen y código, sólo cambia el
> contenido — buckets sandbox → productivos (Terraform) y `MLFLOW_TRACKING_URI`
> → ALB en vez de `mlflow:5000`.

El `docker-compose.yml` valida en parse-time que `S3_MLFLOW_BUCKET` y
`S3_ARTIFACTS_BUCKET` estén seteadas (la sintaxis `${VAR:?mensaje}`
aborta con un error explícito si la variable está vacía o ausente).
Todas las demás variables del compose tienen defaults sensatos via
`${VAR:-default}` y no hace falta declararlas.

#### 4.7.1 Crear `.env.example`

```bash
# AWS (sin secrets)
# Las credenciales VIVEN EN ~/.aws/credentials del host (creado con
# `aws configure`). docker-compose monta ~/.aws:ro en los containers
# y el SDK las lee via AWS_PROFILE + AWS_SHARED_CREDENTIALS_FILE.
AWS_PROFILE=default
AWS_DEFAULT_REGION=us-east-1

# Postgres (MLflow backend store local)
# Default sano para localhost-only. Override en cualquier entorno
# compartido o VM en red.
# POSTGRES_PASSWORD=mlflow

# Trainer resource limits (opcional)
# TRAINER_MEM=8g
# TRAINER_CPUS=4

# Buckets S3 SANDBOX (REQUERIDOS para Tramo I)
# Crear con `task local:ensure-buckets` (#4.8).
#
# Estas DOS variables apuntan habitualmente al MISMO bucket:
#   - S3_MLFLOW_BUCKET     lo consume el server MLflow como --artifacts-destination
#                          (modo proxy: el server es el único que toca S3; los
#                          runs nuevos quedan como mlflow-artifacts:/...).
#                          IMPORTANTE: el server debe poder alcanzar S3 →
#                          AWS_DEFAULT_REGION debe ser la región REAL del bucket.
#   - S3_ARTIFACTS_BUCKET  lo consume el trainer en scripts/s3_sync.py
#                          (sube reports/<variety>/ y artifacts/*.joblib)
#
# Conviven en un solo bucket porque van a prefijos distintos y no colisionan.
# Separarlas solo si necesitas politicas IAM o lifecycle diferenciadas (Tramo II).
# En Terraform productivo existe solo `artifacts_bucket` — ambas resuelven a el.
S3_MLFLOW_BUCKET=ml-training-artifacts-XXXXXX
S3_ARTIFACTS_BUCKET=ml-training-artifacts-XXXXXX

# MLflow (opcional — útil si querés apuntar a un MLflow remoto)
# MLFLOW_TRACKING_URI=http://localhost:5000
# MLFLOW_EXPERIMENT_PREFIX=
# MODEL_REGISTRY_PREFIX=rnd-forest-       # (consumida solo en Tramo II; Tramo Local no registra modelos)

# Hydrate de data desde S3 (paridad con AWS Batch — OPCIONAL en Tramo Local)
# Si estas dos estan seteadas, el trainer baja el Excel ACUMULADO desde S3
# al boot y corre split_workbook() (replicando el flujo de Batch).
# Si quedan vacias, lee ./data/BD_HISTORICO_ACUMULADO.xlsx del bind-mount.
#
# OJO con S3_DATA_KEY: apunta al Excel ACUMULADO (input de
# scripts/prepare_data.split_workbook), NO al split DB-HISTORICA.xlsx.
# `main.py::_hydrate_data_from_s3` baja S3_DATA_KEY -> ACCUMULATED_FILE
# y luego split_workbook genera TRAINING_FILE. El default coincide con
# el del Lambda dispatcher (infra/lambdas/dispatcher.py L5112).
#
# Util para validar el contrato `s3_hydrate` antes de Tramo II (ver #4.5.3).
# S3_DATA_BUCKET=ml-training-data-XXXXXX
# S3_DATA_KEY=BD_HISTORICO_ACUMULADO.xlsx

# Reporte gerencial (opcional)
# REPORT_PLOTLY_OFFLINE=1

# ── App stack: API (FastAPI) + UI (Streamlit) ─────────────────────────────
# La API sirve los modelos rnd-forest-* desde MLflow y persiste pronosticos en
# la base `forecasts` (mismo Postgres local, otra DB). La UI consume la API.
#
# Prefijo del registered model — DEBE coincidir entre trainer y API.
# MODEL_REGISTRY_PREFIX=rnd-forest-
# Precarga de modelos en la API: false = lazy (arranque rapido, menos RAM).
# MLFLOW_PRELOAD_MODELS=false
# Origenes CORS permitidos por la API (coma-separados).
# CORS_ORIGINS=http://localhost:8501
# URL con la que la UI llama a la API (en compose: http://api:8000).
# UI_API_URL=http://api:8000
# Nivel de log compartido por api + ui.
# LOG_LEVEL=info
```

#### 4.7.2 Copia para uso real

```bash
cp .env.example .env
# Editar .env: completar los dos buckets con valores reales
```

Completá los dos buckets sandbox; el resto tiene defaults sensatos y los
comentarios in-line del `.env.example` explican cuándo activar cada override y
por qué las dos vars S3 suelen apuntar al mismo bucket.

> *Para verificar / re-ejecutar esta sección, ver #4.A.*

### 4.8 Buckets S3 sandbox

Es la **única dependencia AWS** del smoke local: el resto del stack vive en
containers en tu laptop. Los buckets son el backend remoto de artifacts y un
avance del Tramo II (mismos artefactos, luego apuntando a la cuenta productiva).

`task local:ensure-buckets` los crea idempotentemente con el mismo hardening que
el módulo `storage` de prod: versioning (rollback de artifacts pisados), SSE-S3
AES256 (encryption-at-rest sin KMS) y Public Access Block en sus cuatro flags.

```bash
export PROJECT="ml-training"
export AWS_DEFAULT_REGION="us-east-1"

task local:ensure-buckets
#   ml-training-data-<suffix>       CREADO (o EXISTE)
#   ml-training-artifacts-<suffix>  CREADO (o EXISTE)
```

Completar `.env` con los nombres reales:

```bash
SUFFIX=$(bash scripts/aws-suffix.sh)
sed -i "s|S3_MLFLOW_BUCKET=.*|S3_MLFLOW_BUCKET=ml-training-artifacts-${SUFFIX}|"     .env
sed -i "s|S3_ARTIFACTS_BUCKET=.*|S3_ARTIFACTS_BUCKET=ml-training-artifacts-${SUFFIX}|" .env
```

> **Por qué `scripts/aws-suffix.sh` y no `tail -c 7`** — `tail -c 7`
> incluye el `\n` final del `aws` CLI (devuelve 6 dígitos en Linux) y
> peor en Windows con CRLF (devuelve 5 dígitos, suffix inválido). El
> script POSIX `${acct#?????}` extrae los 7 dígitos correctos de forma
> portable y es la fuente única usada también por `tasks/local.yml`.

**Verificación**

```bash
SUFFIX=$(bash scripts/aws-suffix.sh)
aws s3api get-bucket-versioning --bucket "ml-training-artifacts-${SUFFIX}"
# { "Status": "Enabled" }
```

> **Los dos buckets quedan vacíos — es lo esperado.** `ensure-buckets` solo los
> **crea**, no sube datos:
>
> - `{project}-artifacts-<suffix>` lo puebla el trainer al final de `task train`
>   ([scripts/s3_sync.py](scripts/s3_sync.py)) y el server MLflow durante el run.
> - `{project}-data-<suffix>` queda **vacío** todo Tramo I (la data se lee del
>   bind-mount `./data → /app/data`); solo se puebla si validás el contrato
>   `s3_hydrate` antes de Batch (procedimiento con `aws s3 cp`, ver #4.13). En
>   Tramo II #4.2 se sube el Excel al bucket productivo (mismo nombre por SUFFIX).

> *Para verificar / re-ejecutar esta sección, ver #4.A.*

### 4.9 Primera ejecución

> [!TIP]
> Este es el **smoke test del Tramo I**. Asume que ya hiciste el setup de
> una sola vez: buckets S3 creados (#4.8) y `.env` completado (#4.7) — sin
> eso el paso 1 (`task build`) aborta **antes** de arrancar MLflow con
> `Set S3_MLFLOW_BUCKET in .env`. Si los cuatro comandos del pipeline de
> abajo terminan sin error, tenés infra local sana y un loop de
> desarrollo cerrado en tu laptop. La secuencia es estricta — cada
> paso asume el éxito del anterior. **El EDA (paso 3) va antes del
> `train` (paso 4) intencionalmente**: es el gate de calidad de datos
> — si muestra PSI alto, VIF explotado o BP/DW fuera de rango,
> entrenar es regalar CPU a un modelo que el quality gate de Parte 7
> va a rechazar igual. Si algo rompe: #4.10 valida el resultado (5
> checks) y #4.12 enumera fallos comunes con fixes.

```bash
# 0) PREREQUISITO (solo 1ª vez) — buckets S3 + .env. Ver #4.8 y #4.7.
#    task local:ensure-buckets  y luego completar S3_MLFLOW_BUCKET /
#    S3_ARTIFACTS_BUCKET en .env con los nombres reales. Sin esto, el
#    paso 1 aborta con "Set S3_MLFLOW_BUCKET in .env" (lo exige
#    docker-compose.yml en --artifacts-destination del server MLflow).

# 1) Build de la imagen + arranque de servicios
task build
# Tarda 5-10 min la primera vez (compila wheels en stage 1).
# Al final imprime las URLs:
#   UI (dashboard)  http://localhost:8501
#   API (Swagger)   http://localhost:8000/docs
#   MLflow UI       http://localhost:5000
#   Reports HTML    http://localhost:8080/reports/
#   Artifacts       http://localhost:8080/artifacts/

# 2) Generar el dataset de training
task data:split
# Lee  data/BD_HISTORICO_ACUMULADO.xlsx
# Escribe data/training/DB-HISTORICA.xlsx (1 hoja por variedad)

# 3) EDA estadistico - gate de calidad de datos (~30 s)
task eda VARIETIES=POP
# Carga data/training/DB-HISTORICA.xlsx hoja POP y genera:
#   reports/EDA_POP_<ts>.html   dashboard renderizado (BP/DW/ADF/VIF/MI/PSI)
#   reports/EDA_POP_<ts>.json   mismos hallazgos en JSON parseable
# Abrir el HTML en http://localhost:8080/reports/ y revisar la
# seccion "Hallazgos" (top 5 con severity). Si hay severity=high
# por PSI > 0.25 o VIF > 10, pausar y entender ANTES de entrenar.

# 4) Smoke test (~1 min)
task train VARIETIES=POP TUNING=smoke
# Al final del log:
#   FIN | variedades=1 | falladas=0 | tiempo_total=...s
#   Campeones por variedad:
#     POP                       -> xgb composite=...
```

> *Para verificar / re-ejecutar esta sección, ver #4.A.*

### 4.10 Verificación post-smoke (7 checks)

Exit code 0 es **necesario pero no suficiente**: hay caminos donde el proceso
termina limpio pero los artifacts no llegan a S3, MLflow registra el run sin
params, o nginx no expone los reports. Los cinco primeros checks cubren cada
componente del stack (MLflow, Postgres, S3, nginx, agregado); los dos últimos
validan el **contrato del run MLflow** (Cap 1.5: joblib usable + tags
obligatorios):

```bash
# 1) MLflow server responde
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5000/health
# 200

# 2) Postgres tiene el experimento POP con 2 runs (xgb + lgb)
docker compose exec postgres psql -U mlflow -d mlflow -c \
  "SELECT name, (SELECT COUNT(*) FROM runs WHERE experiment_id = e.experiment_id) AS n_runs
   FROM experiments e WHERE name = 'POP';"
# POP | 2

# 3) joblib del campeón en S3
SUFFIX=$(bash scripts/aws-suffix.sh)
aws s3 ls "s3://ml-training-artifacts-${SUFFIX}/artifacts/" --recursive \
  | grep -E "final_pipeline_POP_.*\.joblib$"
# Al menos un match con timestamp reciente

# 4) nginx sirve los reports
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/reports/
# 200 (o 301 si autoindex redirige)

# 5) run_summary del agregado existe en host
cat artifacts/run_summary_AGGREGATE.json | jq '.champions'
# { "POP": "xgb" }   (o "lgb", depende del run)

# 6) joblib roundtrip: el campeón se puede recargar y predecir
#    Detecta pickles corruptos / sklearn version mismatch / signature ausente
docker compose run --rm --no-deps --entrypoint python trainer -c '
import glob, joblib, pandas as pd
path = sorted(glob.glob("artifacts/final_pipeline_POP_*.joblib"))[-1]
m = joblib.load(path)
sample = pd.read_excel("data/training/DB-HISTORICA.xlsx", sheet_name="POP").head(1)
y = m.predict(sample.drop(columns=["KG/JR_H"], errors="ignore"))
assert y.shape == (1,), f"prediccion mal: {y.shape}"
print(f"OK  joblib={path}  y[0]={y[0]:.3f}")
'
# OK  joblib=artifacts/final_pipeline_POP_<ts>.joblib  y[0]=<float>

# 7) Contrato del run MLflow (Cap 1.5): los 8 tags obligatorios presentes
#    Si alguno falla, el run no es promovible al gate de Parte 7.
docker compose exec postgres psql -U mlflow -d mlflow -tA -c "
WITH last_run AS (
  SELECT r.run_uuid
  FROM runs r JOIN experiments e ON r.experiment_id = e.experiment_id
  WHERE e.name = 'POP' ORDER BY r.start_time DESC LIMIT 1
),
need AS (
  SELECT unnest(ARRAY['git_commit','git_dirty','dataset_sha256','dataset_n_rows',
                      'tuning','variety']) AS k
),
have AS (SELECT key AS k FROM tags WHERE run_uuid IN (SELECT run_uuid FROM last_run))
SELECT n.k AS missing FROM need n LEFT JOIN have h USING(k) WHERE h.k IS NULL;
"
# (esperado: cero filas)  -> si lista alguna, falta tag en el codigo del trainer
```

> **Si el check #6 falla** con `ModuleNotFoundError` al cargar el
> joblib, es señal de sklearn version mismatch entre el run y la
> imagen actual (cambiaron `requirements.txt`). Re-correr `task build`
> y reintentar.
>
> **Si el check #7 lista filas**, falta tagear en el código. El
> trainer (`src/orchestration/single_run.py` o similar) debe llamar
> `mlflow.set_tags({...})` al inicio del run con los 8 keys
> enumerados. Sin estos tags, `task ops:promote` aborta —
> documentado en Parte 7.

Si los siete checks dan verde, **el setup local está validado**. El Tramo II no
tiene urgencia: el stand-up de AWS toma 2–3 horas y no aporta nada hasta que el
código del trainer esté donde lo querés. Iterá local lo que necesites antes de subir.

### 4.11 Workflow día a día

Con la imagen ya cacheada, el ciclo cotidiano son unos pocos comandos: disparás
entrenamientos con perfiles cada vez más exigentes a medida que confiás en los
cambios, y apagás todo al final del día (los datos persisten en el volumen de
Postgres y en S3, así que el `down` es seguro).

```bash
# Iterar (task train levanta los servicios automaticamente via deps:[_up])
task train VARIETIES=POP TUNING=dev          # ~20 min, baseline
task train VARIETIES=POP TUNING=prod         # ~2 h, producción
task train VARIETIES=all PARALLEL=3          # todas en paralelo

# Seguir progreso en vivo (comando directo a docker compose)
docker compose logs -f --tail=200 trainer mlflow

# Tras tocar src/ o requirements.txt: rebuild + servicios up
task build

# Opcional tras un rebuild que toco deps: auditar CVEs HIGH/CRITICAL
# (requiere `trivy` instalado en el host)
trivy image ml-training:local --severity HIGH,CRITICAL --ignore-unfixed

# Noche: apagar (preserva volumen Postgres + S3)
task down
```

> **Nota — dashboard `reports/index.html`**: se regenera
> automáticamente al final de cada `task train` (vía
> `variety_runner.py:_write_global_dashboard_index`). No hace falta una
> task aparte. Si por algún motivo el archivo quedó desactualizado (ej.
> borraste manualmente algún HTML de `reports/`):
> `docker compose run --rm --no-deps --user "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp --entrypoint python trainer -m src.diagnostics.dashboard_index`
> (el `--user` evita el `PermissionError` al reescribir `reports/index.html`, igual que `{{.DC_PY}}`).

### 4.12 Troubleshooting local

La tabla cataloga los quince fallos más frecuentes al construir el entorno por
primera vez: **síntoma observable** (el mensaje exacto), causa raíz y fix mínimo.
Si tu error no encaja literal, leelo igual — la mayoría son variantes de estos.

| Síntoma | Causa probable | Fix |
|---|---|---|
| `ERROR: Set S3_MLFLOW_BUCKET in .env` | `.env` ausente o variable vacía | `cp .env.example .env` + completar #4.7 |
| `Unable to locate credentials` en logs MLflow / trainer | `~/.aws/credentials` no existe, o `AWS_PROFILE` apunta a un profile inexistente | `aws configure` + `cat ~/.aws/credentials` |
| `NoSuchBucket` al arrancar mlflow | Buckets en `.env` no existen | `task local:ensure-buckets` |
| `Host header ... not allowed` | Cliente pega contra un host fuera del `--allowed-hosts` | Usar `mlflow:5000` o `localhost:5000`; o editar el `command:` del compose |
| Trainer muere con `OOMKilled` (exit 137) | `mem_limit: 8g` insuficiente | `TRAINER_MEM=16g` en `.env`, después `task down && task build` |
| `Port 5000/8080 already allocated` | Otro proceso usa esos puertos | `lsof -i :5000` (o `:8080`), matar o cambiar `ports:` |
| `task train` no encuentra `DB-HISTORICA.xlsx` | Saltaste `task data:split` | Correr `task data:split` primero |
| `git_commit=unknown` en MLflow tag | El container no monta `.git/` (excluido por `.dockerignore`) | OK en dev local; `task train` ya inyecta `GIT_SHA` via `-e` |
| Postgres healthcheck en `starting` para siempre | Imagen corrupta o disco lleno | `docker compose down -v` (DESTRUCTIVO: borra volumen Postgres), `docker system prune`, reintentar |
| `Connection refused: mlflow:5000` desde trainer | Red Docker rota | `task down && task build`; si persiste, `docker network prune` |
| `nginx 403 Forbidden` en `/reports/` | `reports/` vacío o sin permisos | Correr al menos un `task train`; revisar `ls -la reports/` |
| `task train TUNING=prod` se cuelga | `PARALLEL` alto + `TRAINER_CPUS` bajo → oversubscription | Bajar `PARALLEL` o subir `TRAINER_CPUS` |
| `Stage 'builder' failed: target stage not found` o build mucho más lento que lo esperado | BuildKit no habilitado: la directiva `# syntax=docker/dockerfile:1.7` se ignora | `export DOCKER_BUILDKIT=1` en la shell, o instalar Docker Buildx (`docker buildx version`) |
| MLflow UI muestra "No runs" después de un `task train` exitoso | `MLFLOW_EXPERIMENT_PREFIX` seteado y la UI filtra por el prefix antiguo | Limpiar la barra de búsqueda en la UI o quitar `MLFLOW_EXPERIMENT_PREFIX` del `.env` |
| Check #6 de #4.10 falla con `ModuleNotFoundError` o `_RemainderArgs` | sklearn/xgb/lgb version mismatch entre el `.joblib` y la imagen actual (cambió `requirements.txt`) | `task build` (rebuild de la imagen) y volver a correr `task train` |

### 4.A Re-ejecución por sección (referencia)

Tabla rápida para re-correr cualquier sub-sección de #4 sin re-leer la
prosa. Cada fila tiene el chequeo mínimo de "esto quedó bien" y el
pitfall que se ve más seguido. Las sub-secciones del capítulo apuntan
acá con una línea de cross-ref.

| # | Verificar | Pitfall típico | Commit |
|---|---|---|---|
| **4.3** `.dockerignore` | `docker build --progress=plain --no-cache -t ml-training:dryrun . 2>&1 \| head -5` → `transferring context: ...kB` | Transfer >10 MB = `.dockerignore` no se aplicó (path o sintaxis); archivo debe estar en raíz junto al `Dockerfile` | `chore(docker): add .dockerignore` |
| **4.4** `Dockerfile` | `docker build -t ml-training:local . && docker images ml-training:local` → SIZE ~1.2 GB | Stage 1 falla compilando wheels: falta `build-essential` o `# syntax=docker/dockerfile:1.7` (sin eso cache mount se ignora) | `feat(docker): Dockerfile multi-stage del trainer` |
| **4.5.3** `docker-compose.yml` | `docker compose config` parsea sin errores y lista 6 servicios | CRLF en WSL rompe el YAML (`mapping values are not allowed here`); `dos2unix docker-compose.yml docker/*.conf docker/mlflow/Dockerfile` | `feat(docker): compose con mlflow + postgres + reports + trainer` |
| **4.6.2** `tasks/local.yml` | `task --list` muestra 6 públicas + `local:ensure-buckets` + `local:bucket-name` | Indentación con TABS rompe Task (`failed to read taskfile`); `expand -t 2` para reemplazar tabs | `feat(taskfile): tasks locales + namespace local:` |
| **4.7** `.env.example` y `.env` | `cp .env.example .env && grep -c "^[A-Z]" .env` ≥ 5 (vars activas, no comentadas) | Placeholder `XXXXXX` en `S3_*_BUCKET` rompe `task build` con `Set S3_MLFLOW_BUCKET in .env`; completar tras #4.8 | `chore(env): .env.example template` |
| **4.8** Buckets S3 sandbox | `aws s3 ls \| grep ml-training` lista 2 buckets con el `ACCOUNT_SUFFIX` de #3.5 | `aws sts get-caller-identity` sin credenciales → `SUFFIX` vacío → nombre inválido; re-correr #3.3 antes | N/A (op AWS, no genera archivos) |
| **4.9** Primera ejecución | Los 5 checks de #4.10 (`/health` 200, runs en Postgres, joblib en S3, nginx 200, `run_summary_AGGREGATE.json`) | `task build` colgado en `waiting for mlflow healthcheck` >2 min → `docker logs ml_training-mlflow-1`: `NoSuchBucket` o `Unable to locate credentials` | `chore(smoke): primer entrenamiento local OK` |

**Tiempos esperados**: #4.3 instantáneo · #4.4 ~5 min (1ª vez) / ~20s con
cache · #4.8 ~30s · #4.9 5-10 min (1ª vez) / 1-2 min con cache.

### 4.13 Próximo paso: Tramo II

Con el smoke local en verde, el **código del trainer está validado**:
toma datos reales, entrena ambos algoritmos, elige campeón por la
métrica compuesta, persiste artifacts en S3 y registra el run en
MLflow. El Tramo II no reescribe nada de eso — promueve el **mismo
binario** a AWS Batch, sustituyendo sólo la infraestructura que lo
rodea. La promoción se hace en cuatro pasos:

1. **Bootstrap del backend Terraform** (Parte 2) — operación irreversible
   que se ejecuta UNA sola vez por cuenta AWS.
2. **Aplicar módulos Terraform** (Partes 3–4) — levanta VPC, S3, ECR,
   MLflow sobre ECS Fargate, y la cola de AWS Batch.
3. **Build + push de la imagen del trainer a ECR** — el mismo
   `Dockerfile` de #4.4 se tagea y empuja con `task ecr:build IMG=trainer`.
4. **Smoke test en Batch** — equivalente productivo de `task train
   TUNING=smoke`, invocado con `task batch:smoke`.

> **Antes de empujar a ECR, audita CVEs HIGH/CRITICAL de la imagen
> local con `trivy image ml-training:local --severity HIGH,CRITICAL
> --ignore-unfixed`** (en local es warn-only — solo te muestra la
> superficie). El mismo binario se promueve a AWS sin modificación,
> así que las vulnerabilidades de hoy en tu laptop son las de mañana
> en Batch. Conocerlas acá te evita el push fallido en Tramo II Parte
> 4, donde el scan-on-push del ECR puede bloquear el deploy.

Lo que **no cambia** entre tu laptop y AWS — la garantía central del
diseño "una sola imagen":

- El `Dockerfile` es bit-perfect idéntico. La imagen que corrió tu
  smoke local es la misma que corre en Batch.
- El código (`main.py`, `src/`) se configura enteramente por variables
  de entorno; sólo cambia el origen de esas variables (en local salen
  del `.env`, en Batch de la job definition).
- El modelo final es reproducible: dada la misma data y los mismos
  params, el joblib y el MAPE son los mismos en local y en producción.

Lo que **sí cambia**, y es el alcance entero del Tramo II:

| Componente | Local | AWS |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` (DNS de Docker) | `http://<ALB-DNS>` (ALB + Service Discovery) |
| Postgres | Container con volumen Docker | RDS managed con backups automáticos |
| Credenciales AWS | `~/.aws:/aws:ro` bind-mount | IAM Task Role (metadata endpoint, sin secretos en disco) |
| Origen del dataset | Bind-mount `./data → /app/data` | `aws s3 cp s3://$S3_DATA_BUCKET/$S3_DATA_KEY ./data/...` al boot del container |
| Model Registry | No se popula (Tramo Local no registra) | `mlflow.register_model()` se invoca al final del run |
| Trigger del entrenamiento | `task train` manual | Lambda dispatcher o `workflow_dispatch` de GitHub Actions |

> **Validar el contrato del hydrate antes de subir a Batch.** El
> origen del dataset es la única ruta donde "mismo binario" se
> rompe sutilmente: en local lee un bind-mount, en Batch baja desde
> S3. Si nunca probaste el path S3 en local, lo descubrís recién en
> el primer `task batch:smoke`. Para validarlo sin AWS Batch son
> 3 pasos (asumiendo que ya corriste `task local:ensure-buckets` en
> #4.8 — el bucket `data` existe pero está vacío):
>
> ```bash
> # 1) Subir el Excel ACUMULADO al bucket sandbox de data.
> #    `task local:ensure-buckets` solo crea el bucket; este `cp` lo puebla.
> SUFFIX=$(bash scripts/aws-suffix.sh)
> aws s3 cp data/BD_HISTORICO_ACUMULADO.xlsx \
>     "s3://ml-training-data-${SUFFIX}/BD_HISTORICO_ACUMULADO.xlsx"
>
> # 2) Activar el hydrate en .env (descomentar las dos lineas de #4.7.1).
> #    OJO: S3_DATA_KEY apunta al ACUMULADO, no al split DB-HISTORICA.xlsx.
> cat >> .env <<EOF
> S3_DATA_BUCKET=ml-training-data-${SUFFIX}
> S3_DATA_KEY=BD_HISTORICO_ACUMULADO.xlsx
> EOF
>
> # 3) Re-correr el training — main.py::_hydrate_data_from_s3 detecta
> #    las vars y baja el Excel al boot, idéntico al flujo de Batch.
> task train VARIETIES=POP TUNING=smoke
> ```
>
> Si el run termina OK, el contrato `s3_hydrate` está validado:
> en Tramo II #4.2 sólo cambian los nombres de bucket (sandbox →
> productivos por SUFFIX), no la lógica del trainer. Para volver al
> modo bind-mount comentá las dos líneas de `.env`. Ver #4.5.3
> (docker-compose) y #4.7.1 (.env).

> **Nota — el modo híbrido.** Entrenar **desde tu laptop** loggeando contra el
> **MLflow productivo** (reproducir un run que regresionó, o validar features con
> trazabilidad de prod sin gastar Batch). Configuración en
> `docker-compose.override.yml.example` — ver también el callout de #4.5.3.

---

# ☁️ Tramo II — Producción AWS

> El mismo binario que validaste en Tramo I se promueve aquí a AWS Batch +
> MLflow productivo. La infraestructura se levanta con Terraform por capas
> (módulos), Task orquesta builds y lifecycle, y GitHub Actions automatiza
> deploy/training/destroy via OIDC.

> [!IMPORTANT]
> **No empieces el Tramo II sin haber pasado el smoke local del Capítulo 4**
> (`task train VARIETIES=POP TUNING=smoke` exitoso, MLflow UI mostrando el
> run, joblib del champion en `artifacts/`). Si el smoke local rompe, AWS
> sólo va a amplificar el problema y vas a pagar minutos de Batch para
> debug que es más rápido en Docker.

**Orden de lectura en el Tramo II:**
1. **Parte 1** — entender los 4 modos de lifecycle (lo que vas a hacer / no hacer).
2. **Parte 2** — bootstrap UNA VEZ por cuenta (irreversible si se hace mal).
3. **Parte 3** — construir los 12 módulos Terraform (incluye `api` + `ui`, Capa 4.5).
4. **Parte 4** — orquestador Task + apply incremental + smoke en AWS.
5. **Partes 5-7** — emitir métricas, CI/CD, promotion gate.
6. **Partes 8-13** — runbook, costos, hardening, troubleshooting, customizaciones.

---

# Parte 1 — Overview del lifecycle y stand-up

La V1 mezcla los modos a lo largo del runbook. En la V2 son explicitos.
Cada uno responde a una pregunta concreta:

| Modo | Pregunta que responde | Tiempo | Costo despues |
|---|---|---|---|
| **STAND-UP** | "Es la primera vez, parto de cero" | 2-3 horas | ~$75/mes (operando) |
| **TEAR-DOWN** | "No voy a usar la infra por 1+ semana, quiero ahorrar" | 15 min | ~$8/mes (solo storage) |
| **REBUILD** | "Volvi y quiero levantar otra vez sin perder modelos/data" | 20-30 min | ~$75/mes |
| **DESTROY** | "Termine el proyecto / migro a otra cuenta, borra TODO" | 30-45 min | $0/mes |

Diagrama de transiciones:

```
                          stand-up
            (vacio) ─────────────────► OPERATING (64$/mes)
                                          │  ▲
                                          │  │
                                  tear-down  rebuild
                                          │  │
                                          ▼  │
                                       HIBERNATED (8$/mes)
                                          │
                                       destroy
                                          │
                                          ▼
                                       (vacio)
```

> **Que cubre esta Parte 1**: solo el **STAND-UP** (#1.1, abajo) — el unico
> modo que necesitas en una primera lectura, porque todavia no tenes nada
> construido. Los otros 3 modos (TEAR-DOWN / REBUILD / DESTROY) son
> operaciones del runbook y viven en **#8.5-#8.7**: aplican cuando ya
> estuviste operando el sistema.
>
> **Regla de oro**: solo se DESTRUYE cuando estas seguro. Tear-down +
> rebuild es seguro y reversible; destroy NO lo es (perdes state, models
> en Registry, RDS snapshots si no los exportaste).

## 1.1 STAND-UP — primera vez, de cero a produccion

Cuando lo uso: la primera vez que despliego, o tras un **DESTROY**.

### Camino completo

```
Capítulo 3 (prereqs validados)
       │
       ▼
Parte 2 (bootstrap: S3 backend + OIDC; lock nativo S3) — 15 min, IRREVERSIBLE
       │
       ▼
Parte 3 (escribir modulos Terraform) — 30-60 min (copy-paste)
       │
       ▼
Parte 4.2 (verificación sintáctica `task infra:validate`) — 10 s
       │
       ▼
Parte 4.3 (apply storage solo: ECR + buckets) — 5 min
       │
       ▼
Parte 4.4 (build + push 5 imagenes a ECR) — 20-30 min (primera vez)
       │
       ▼
Parte 4.5 (apply full: network + RDS + Fargate [MLflow/Reports/API/UI] + Batch + Lambdas + ...) — 15-25 min
       │
       ▼
Parte 4.6 (smoke test: 1 job de Batch end-to-end) — 15-20 min
       │
       ▼
Parte 5 (patch trainer + re-push) — 10 min
       │
       ▼
Parte 6 (CI/CD GitHub Actions) — 30 min
       │
       ▼
Parte 7 (promotion gate) — 20 min
       │
       ▼
OPERATING (~$74/mes)
```

**Tiempo total realista**: 2-3 horas la primera vez, asumiendo que los
prereqs (0.3) estan OK y la imagen Docker del trainer ya esta probada
local (0.3.5).

### Lo que NO se hace en stand-up

- Hardening (TLS, WAF, Multi-AZ, KMS-CMK, VPC endpoints, DR cross-region):
  hardening, día 90+ (fuera del alcance de esta guía).
- Workflows extras (cleanup, drift detection): hardening, futuro.
- Promotion gate: la primera vez podes saltarte la Parte 7; los primeros
  models entran a `Staging` y los promotes a mano via `mlflow ui`.

## 1.2 Otros modos (TEAR-DOWN / REBUILD / DESTROY)

Estos modos son operaciones del runbook (ya tenes el sistema construido),
no del stand-up inicial. En tu primera lectura no los necesitas — saltalos
y volve cuando ya estes operando. Estan documentados en Parte 8:

- **#8.5 — TEAR-DOWN**: apagar todo preservando state + datos (~$8/mes
  hibernado, reversible con rebuild).
- **#8.6 — REBUILD**: volver despues de un tear-down (cambia solo el ALB
  DNS).
- **#8.7 — DESTROY**: eliminar TODO de la cuenta AWS (requiere 3 backups
  manuales previos — solo aplica si ya operaste el sistema y tenes
  modelos en el Registry, datos en RDS y Terraform state poblado).

La matriz cruzada de costos entre modos (stand-up vs tear-down vs destroy)
esta en #9.3.

---

# Parte 2 — Bootstrap irreversible

## 2.1 Por que el bootstrap es a mano

Terraform necesita un backend remoto (S3 con locking) para que el
state este compartido y safe contra concurrent applies. Pero el backend
no se puede crear con el mismo Terraform que lo usa (chicken-and-egg).

> **Nota — locking**: desde Terraform 1.10 el backend `s3` soporta
> `use_lockfile=true`, que guarda el lock como un objeto `<key>.tflock`
> **dentro del mismo bucket** del state. Antes hacia falta una tabla
> DynamoDB aparte (`${PROJECT}-tflock`) y el parametro `dynamodb_table`
> — ambos hoy deprecados. Resultado: **un solo recurso AWS** para
> backend + lock, sin DynamoDB de por medio.

Soluciones posibles:

- **Bootstrap a mano** (lo que hace esta guia): script bash que llama
  AWS CLI directo para crear el bucket S3. Una vez. **No versionado
  en Terraform**. Si lo destruis, lo recreas a mano.
- Terraform con backend local + `terraform state push` despues: mas
  complejo, mas error-prone.
- CloudFormation seed stack: agrega otra herramienta a la pila.

Elegimos (1) porque son <40 lineas de bash, ejecutables UNA vez,
auditable a simple vista, y el "perdes el state" se mitiga con
versioning del bucket S3 (paso 2.4 lo valida).

Lo mismo aplica al **OIDC provider** de GitHub: si lo creas con
Terraform y haces destroy, el proximo GH Actions falla. Por eso se
bootstrap-ea aparte en 2.5.

## 2.2 Script de bootstrap (bash)

Crear el archivo `infra/bootstrap.sh` con el contenido completo de
abajo. Si ya existe (este repo lo tiene), comparar con `diff` antes de
sobreescribir.

> **Convencion de copy-paste**: cada bloque de codigo en esta guia
> esta precedido por un encabezado con el path destino (`infra/X.sh`,
> `infra/modules/Y/main.tf`, etc.). Crear el archivo en ese path con
> editor o `cat > path <<'EOF' ... EOF` y pegar el contenido del
> bloque. NO mezclar bloques de archivos distintos.

> **Equivalente en AWS Console** — esto es lo que el script hace por vos, paso a paso, si lo hicieras click-a-click:
>
> | Paso del script | Servicio AWS | Que estarias haciendo en Console |
> |---|---|---|
> | 1) `s3api create-bucket` | **S3** | `S3 > Create bucket` con nombre `ml-training-tfstate-<sufijo>` en `us-east-1`. Es donde Terraform va a guardar el archivo `.tfstate` (el "mapa" de que recursos AWS pertenecen a esta infra) **y** el lock file `<key>.tflock` (objeto efimero que Terraform crea/borra en cada apply para evitar applies concurrentes). |
> | 2) `put-bucket-versioning` | **S3** | Dentro del bucket → `Properties > Bucket Versioning > Enable`. Guarda cada cambio del `.tfstate` como version nueva — si un `terraform apply` rompe el state, podes restaurar la version anterior. |
> | 3) `put-bucket-encryption` + `put-public-access-block` | **S3** | `Properties > Default encryption > AES-256` y `Permissions > Block public access > All ON`. El state file tiene secrets en plano (passwords RDS, etc.); cifrarlo y bloquear acceso publico es mandatorio. |
> | 4) `create-service-linked-role` (x3) | **IAM** | NO hay wizard "Create role" para esto — las **Service Linked Roles (SLR)** son especiales. En Console aparecen en `IAM > Roles` ya creadas (`AWSServiceRoleForEC2Spot`, `AWSServiceRoleForECS`, `AWSServiceRoleForBatch`) cuando AWS las genera **automaticamente** al primer uso del servicio. El script las pre-crea via API (`iam:CreateServiceLinkedRole`) para que el primer `terraform apply` (que las asume implicitamente al lanzar Spot/ECS/Batch) no falle con `role does not exist yet`. Son distintas a los roles "normales" porque solo pueden ser asumidas por el service AWS exacto que las nombra (no por usuarios), y AWS las gestiona internamente. |
>
> **Que paso con DynamoDB**: hasta Terraform 1.9 el lock estaba en una tabla `${PROJECT}-tflock` y el backend se inicializaba con `-backend-config=dynamodb_table=…`. Desde 1.10 esa via fue deprecada en favor de `use_lockfile=true` (lock nativo S3, objeto `<key>.tflock` en el mismo bucket). Esta guia ya usa el modo nuevo: **no se crea tabla DynamoDB**. Si vienes de un stack viejo y tenes `ml-training-tflock` huerfana, borrala con `aws dynamodb delete-table --table-name ml-training-tflock --region us-east-1`.
>
> **Por que no lo haces desde Console**: estos recursos son la "base que sostiene a Terraform mismo". Si los crearas a mano y los borraras sin querer, perderias el state entero y Terraform no sabria que recursos AWS le pertenecen (los huerfanaria, pagandolos sin poder destruirlos). El script los hace **idempotentes** (re-ejecutar es seguro) y deja un audit trail claro.

```bash
#!/usr/bin/env bash
# infra/bootstrap.sh — Bootstrap del backend Terraform.
# UNA VEZ por cuenta + region. Idempotente: re-ejecutar es seguro.
#
# Crea:
#   1) S3 bucket  ${PROJECT}-tfstate-${ACCOUNT_SUFFIX}  (state file Terraform)
#   2) Service Linked Roles para Spot / ECS / Batch     (pre-creadas)
#
# Locking: usamos `use_lockfile=true` (locking nativo S3, Terraform >= 1.10).
# El lock vive como objeto `<key>.tflock` en el mismo bucket de tfstate, asi
# que NO necesitamos una tabla DynamoDB separada. Si vienes de un bootstrap
# antiguo con `ml-training-tflock`, puedes borrarla con:
#   aws dynamodb delete-table --table-name ${PROJECT}-tflock --region $REGION
#
# El bucket S3 se crea via scripts/ensure-s3-bucket.sh (mismo helper que
# tasks/local.yml usa para data/artifacts). Asi el hardening
# (versioning + AES256 + public-access-block) vive en UN solo lugar.
#
# El sufijo se calcula via scripts/aws-suffix.sh (fuente unica). Los buckets
# de prod (data, artifacts, archive) usan el mismo sufijo de 7 digitos.

set -euo pipefail

PROJECT="${PROJECT:-ml-training}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
# Reusa ACCOUNT_SUFFIX de la sesion si ya esta exportado (Capitulo 3.5); sino
# lo calcula con el mismo script que tasks/local.yml -> garantiza coherencia
# entre buckets locales y de prod.
ACCOUNT_SUFFIX="${ACCOUNT_SUFFIX:-$(bash scripts/aws-suffix.sh)}"
TFSTATE_BUCKET="${PROJECT}-tfstate-${ACCOUNT_SUFFIX}"

# 1) S3 bucket tfstate (delegado al helper compartido)
bash scripts/ensure-s3-bucket.sh "$TFSTATE_BUCKET" "$REGION"

# 2) Service Linked Roles (errores "ya existe" se ignoran)
aws iam create-service-linked-role --aws-service-name spot.amazonaws.com   2>/dev/null || true
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com    2>/dev/null || true
aws iam create-service-linked-role --aws-service-name batch.amazonaws.com  2>/dev/null || true

echo "==> BOOTSTRAP COMPLETADO"
echo "    bucket=$TFSTATE_BUCKET  region=$REGION  (lock: nativo S3 via use_lockfile)"
```

> **Por qué reusa `scripts/ensure-s3-bucket.sh`**: ese helper ya encapsula
> creación + hardening (versioning + AES256 + public-access-block) y lo usa
> `tasks/local.yml` para `data`/`artifacts`; duplicar la lógica inline divergiría
> con el tiempo. Igual con el sufijo: `scripts/aws-suffix.sh` es la fuente única
> (POSIX `${acct#?????}` ⇒ últimos 7 chars), evitando que bootstrap calcule `-6`
> y el resto `-7`.

> **Gotcha #2.2**: `bash -n infra/bootstrap.sh` valida sintaxis sin tocar AWS (atrapa heredocs rotos `<<EOF` sin cierre o `$` sin escapar). Commit: `chore(infra): bootstrap script para S3 tfstate (lock nativo)`.

## 2.3 Ejecutar UNA vez

> [!IMPORTANT]
> Este bootstrap **se corre exactamente una vez por cuenta + región**. Crea
> el bucket `tfstate` (que guarda el state de TODA la infra). El locking
> entre applies concurrentes se hace nativo via `use_lockfile=true` —
> Terraform escribe un objeto `<key>.tflock` en el mismo bucket y lo borra
> al terminar. Es idempotente, así que re-ejecutarlo no rompe nada, pero
> **destruir el bucket tfstate** sólo se hace en `aws:nuke` (#8.7) — y si
> lo borrás a mano sin nuke, pierdes el historial Terraform y el próximo
> `terraform plan` ve toda la infra como "a crear" aunque ya exista.

```bash
# Desde la raiz del repo (WSL Ubuntu o bash nativo Linux/Mac)
cd /mnt/c/Users/CarlosAlexanderAbant/Documents/Proyectos/ml_random_forest/ml_training

# Crear el directorio infra/ si no existe
mkdir -p infra

# Verificar que el script existe (lo creaste en #2.2)
ls -la infra/bootstrap.sh
# Si no existe -> volver a #2.2 y pegar el contenido en infra/bootstrap.sh

# Dar permiso ejecutable + ejecutar
chmod +x infra/bootstrap.sh
bash infra/bootstrap.sh
```

Salida esperada (el script es silencioso; solo imprime el resumen final):

```
==> BOOTSTRAP COMPLETADO
    bucket=ml-training-tfstate-789012  region=us-east-1  (lock: nativo S3 via use_lockfile)
```

Si NO ves esa linea final, algun `aws` command fallo silenciosamente
arriba (revisa con `bash -x infra/bootstrap.sh` para verlo paso a paso).

**Si re-ejecutas el script**: es idempotente. Va a decir "Ya existe.
Skip create." en los pasos donde el recurso ya esta, y los SLR no
fallan (estan filtrados con `2>/dev/null`).

> **Gotcha #2.3**: si ya corriste bootstrap antes y el bucket tfstate existe con nombre distinto (account-id o region distinta), el script es idempotente PERO no detecta drift de nombre — vas a terminar con **dos buckets tfstate**. Revisar con `aws s3 ls | grep tfstate` antes de re-correr.

## 2.4 Verificacion post-bootstrap (3 checks)

Despues de que el script termine, valida que TODO quedo bien:

```bash
# Variables (recreadas para que esta seccion sea standalone)
export PROJECT="ml-training"
export ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export ACCOUNT_SUFFIX="${ACCOUNT_ID: -7}"
TFSTATE_BUCKET="${PROJECT}-tfstate-${ACCOUNT_SUFFIX}"

# Check 1: bucket existe y tiene versioning ON
aws s3api get-bucket-versioning --bucket "$TFSTATE_BUCKET" --query Status --output text
# Esperado: Enabled

# Check 2: bucket tiene encryption AES256
aws s3api get-bucket-encryption --bucket "$TFSTATE_BUCKET" \
    --query 'ServerSideEncryptionConfiguration.Rules[0].ApplyServerSideEncryptionByDefault.SSEAlgorithm' \
    --output text
# Esperado: AES256

# Check 3: bucket bloquea acceso publico
aws s3api get-public-access-block --bucket "$TFSTATE_BUCKET" \
    --query 'PublicAccessBlockConfiguration.BlockPublicAcls' --output text
# Esperado: True
```

Si los 3 dan los valores esperados, **el bootstrap esta OK**. Si alguno
falla, leelo despacio: la causa mas comun es region mal seteada
(creaste en `us-east-1` pero estas consultando con perfil que default
es otra).

> **Nota — locking nativo S3**: no hay un "check 4" para DynamoDB porque
> ya no usamos tabla de locks. El lock se materializa como un objeto
> efimero `envs/prod/terraform.tfstate.tflock` dentro del mismo bucket,
> creado por `terraform plan/apply` y borrado al terminar. Si un apply
> aborta abruptamente puede quedar huerfano — se libera con
> `task infra:force-unlock LOCK_ID=<id>`.

## 2.5 OIDC provider para GitHub Actions (pre-Terraform)

Mismo motivo que 2.1: el OIDC provider tiene que existir antes de que
Terraform pueda crear los IAM roles que confian en el. Lo creamos a mano.

> **Equivalente en AWS Console** — esto es lo que el script crea, si lo hicieras click-a-click:
>
> | Paso del script | Servicio AWS | Que estarias haciendo en Console |
> |---|---|---|
> | `create-open-id-connect-provider` | **IAM** | `IAM > Identity providers > Add provider`. **Provider type**: OpenID Connect. **Provider URL**: `https://token.actions.githubusercontent.com` (clickear `Get thumbprint` — Console lo deriva sola). **Audience**: `sts.amazonaws.com`. **Thumbprint**: `6938fd4d98bab03faadb97b34396831e3780aea1`. **Warning** — desde **mid-2023** AWS valida internamente el certificado de GitHub contra una CA pinneada, asi que el thumbprint pasa a ser un campo "vestigial" — la API lo sigue requiriendo, pero AWS no lo usa para validar. El script lo pasa hardcodeado por compatibilidad. Si la API te rechaza ese valor en el futuro, basta con cualquier hex valido de 40 chars. |
>
> **Que es esto conceptualmente**: es el "puente de confianza" entre GitHub Actions y tu cuenta AWS. Cuando un workflow corre en GitHub, GH emite un **JWT firmado** que dice "este job corre en el repo X, branch Y, ambiente Z". El provider OIDC le dice a AWS: "confio en los JWT firmados por `token.actions.githubusercontent.com`". Despues, los IAM Roles del modulo `cicd` (Parte 3.11) declaran su trust policy: "permito que asuma este rol cualquiera que venga con un JWT del repo `mi-org/ml_training` en branch `main`". Resultado: GHA puede hacer `aws ecr push` **sin necesitar un Access Key + Secret Key guardado como secret** (que sera la pesadilla de seguridad clasica).
>
> **Por que es shared a nivel cuenta**: AWS solo permite UN OIDC provider por URL en toda la cuenta. Si ya lo creaste para otro repo (ej: `mi-otra-app`), reusalo — no recrees ni borres. Lo que **distingue** que repo puede asumir que rol es el `sub:` claim del trust policy (definido en Parte 3.11.2), no el provider en si.

### Script `infra/bootstrap-oidc.sh`

Crear el archivo `infra/bootstrap-oidc.sh` con el contenido siguiente
(si ya existe, comparar con `diff` antes de sobreescribir):

```bash
#!/usr/bin/env bash
# infra/bootstrap-oidc.sh — OIDC provider de GitHub Actions. UNA VEZ por cuenta.
set -euo pipefail

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
PROVIDER="arn:aws:iam::${ACCOUNT}:oidc-provider/token.actions.githubusercontent.com"

if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$PROVIDER" >/dev/null 2>&1; then
    echo "OIDC provider ya existe: $PROVIDER"
else
    aws iam create-open-id-connect-provider \
        --url "https://token.actions.githubusercontent.com" \
        --client-id-list "sts.amazonaws.com" \
        --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1" >/dev/null
    echo "OIDC provider creado: $PROVIDER"
fi
```

### Ejecutar UNA vez

```bash
chmod +x infra/bootstrap-oidc.sh
bash infra/bootstrap-oidc.sh
```

### Verificacion

```bash
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
aws iam get-open-id-connect-provider \
    --open-id-connect-provider-arn "arn:aws:iam::${ACCOUNT}:oidc-provider/token.actions.githubusercontent.com" \
    --query 'Url' --output text
# Esperado: https://token.actions.githubusercontent.com
```

> **Atencion**: el OIDC provider es **shared a nivel cuenta**. Si tu
> cuenta de AWS ya lo usaba para otro repo, no lo recrees — verifica
> que existe con el check de arriba y segui. La condicion `aud` del
> trust policy (que se define en Parte 3.12 del modulo `cicd`) es lo
> que limita el acceso a tu repo especifico.

> **Gotcha #2.5**: el thumbprint hardcoded cambia raramente, pero si AWS lo rota antes del bootstrap el `create-open-id-connect-provider` falla con `InvalidInput`. Comparar contra el thumbprint publicado por AWS en su doc oficial OIDC + GHA.

## 2.6 Snapshot del estado bootstrapped (commit + tag)

El bootstrap es irreversible y no esta versionado en Terraform. Marcalo
con un commit + tag para tener un punto de retorno claro:

```bash
# Por ahora solo los scripts. terraform.tfvars (con valores sensibles)
# se agrega al .gitignore en Parte 3.2.4 — no existe todavia.
git add infra/bootstrap.sh infra/bootstrap-oidc.sh
git commit -m "infra: bootstrap scripts para S3 tfstate (lock nativo) + OIDC provider"
git tag -a "infra/bootstrap-done" -m "Bootstrap ejecutado en cuenta $ACCOUNT_ID region $AWS_DEFAULT_REGION"
git push origin main --tags   # opcional pero recomendado
```

A partir de este punto, **toda la infra es Terraform**. Los `.sh` del
bootstrap no se vuelven a tocar salvo que destruyas la cuenta entera
(#8.7).

---

# Parte 3 — Modulos Terraform

> **Filosofia de la Parte 3**: cada modulo es una caja con interface
> publica (variables.tf + outputs.tf). El `envs/prod/main.tf` solo
> compone — no contiene `resource "aws_..."` directos. Esto te deja:
>
> - Tocar `modules/batch/` sin re-aplicar el resto.
> - Crear `envs/dev/` o `envs/staging/` copiando `envs/prod/` y
>   cambiando solo `terraform.tfvars`.
> - Hacer reviews de PR donde el diff de un cambio chico es chico
>   (no 200 lineas mezcladas).

## 3.1 Layout — el arbol de archivos

Al final de la Parte 3 tu repo tiene este arbol (los `.sh` del bootstrap
ya estan desde la Parte 2):

```
ml_training/
├── infra/
│   ├── bootstrap.sh                       # Parte 2.2
│   ├── bootstrap-oidc.sh                  # Parte 2.5
│   ├── envs/
│   │   └── prod/
│   │       ├── versions.tf                 # 3.2.1
│   │       ├── backend.tf                  # 3.2.2
│   │       ├── variables.tf                # 3.2.3
│   │       ├── terraform.tfvars            # 3.2.4 (gitignored)
│   │       ├── main.tf                     # 3.2.5
│   │       └── outputs.tf                  # 3.2.6
│   ├── modules/
│   │   ├── _shared/                        # 3.4.5 (trust policies JSON compartidos)
│   │   │   ├── README.md
│   │   │   ├── assume-ecs-tasks.json
│   │   │   ├── assume-lambda.json
│   │   │   ├── assume-ec2.json
│   │   │   ├── assume-batch-service.json
│   │   │   └── assume-github-oidc.json.tftpl
│   │   ├── network/                        # 3.3 (split: main.tf + security_groups.tf)
│   │   ├── storage/                        # 3.4
│   │   ├── mlflow/                         # 3.5 (split: main.tf + alb.tf + ecs.tf + iam.tf + rds.tf)
│   │   ├── reports/                        # 3.6
│   │   ├── api/                            # 3.12 (Capa 4.5 — FastAPI)
│   │   ├── ui/                             # 3.12 (Capa 4.5 — Streamlit)
│   │   ├── batch/                          # 3.7 (split: main.tf + iam.tf)
│   │   ├── monitoring/                     # 3.8
│   │   ├── lambdas/                        # 3.9 (split: dispatcher.tf + notifier.tf, sin main.tf)
│   │   ├── scheduler/                      # 3.10
│   │   ├── cicd/                           # 3.11
│   │   └── consumer-iam/                   # 3.11.5 (Patch 13.5 — rol OIDC repo consumer)
│   └── lambdas/                            # Codigo Python de las Lambdas
│       ├── dispatcher.py                   # 3.9.5
│       ├── notifier.py                     # 3.9.6
│       └── scheduler.py                    # 3.10.4
├── docker/
│   ├── mlflow/Dockerfile                   # ya existe (custom MLflow)
│   ├── reports/Dockerfile                  # 3.6.5 (nginx + s3-sync sidecar)
│   └── nginx-reports.conf                  # ya existe (local) + version cloud (3.6.6)
├── (resto del proyecto: src/, main.py, Dockerfile, ...)
```

Crear el esqueleto vacio:

```bash
# Desde la raiz del repo
dirs=(
    "infra/envs/prod"
    "infra/modules/_shared"
    "infra/modules/network"
    "infra/modules/storage"
    "infra/modules/mlflow"
    "infra/modules/reports"
    "infra/modules/api"
    "infra/modules/ui"
    "infra/modules/batch"
    "infra/modules/monitoring"
    "infra/modules/lambdas"
    "infra/modules/scheduler"
    "infra/modules/cicd"
    "infra/modules/consumer-iam"
    "infra/lambdas"
    "docker/reports"
)
for d in "${dirs[@]}"; do mkdir -p "$d"; done

# Verificar
find infra/ docker/reports -type d
```

> **🔄 Validación por módulo (#3.2-#3.11.5)**: después de pegar cada
> módulo, validá la sintaxis con:
>
> ```bash
> # Para módulos en infra/modules/<mod> (no tienen backend)
> terraform -chdir=infra/modules/<mod> init -backend=false && \
>   terraform -chdir=infra/modules/<mod> validate
> # Esperado: "Success! The configuration is valid."
> ```
>
> Para `envs/prod` (que sí tiene backend), la validación va en #4.2 vía
> `task infra:validate` — no acá, porque entre escribir módulos y aplicar
> pueden pasar días y el init contra el backend remoto cambia. Commit
> sugerido al final de cada módulo: `feat(infra/<mod>): <descripción>`.
> Si `validate` falla con "Module not installed", repetir el `init`. Los
> gotchas específicos por módulo (cuando los hay) están al pie de la
> subsección correspondiente.

## 3.2 `envs/prod/` — la composicion

### 3.2.1 `infra/envs/prod/versions.tf`

Locks de versiones — toda la guia esta probada con estas versiones. Si
las cambias, vas a tener que ajustar sintaxis (p.ej. `for_each` map en
v5 vs v4 del provider AWS).

```hcl
terraform {
  required_version = ">= 1.6.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "Terraform"
      Env       = "prod"
    }
  }
}
```

### 3.2.2 `infra/envs/prod/backend.tf`

```hcl
terraform {
  backend "s3" {
    # Valores se inyectan desde -backend-config en `terraform init`.
    # Asi el bucket no queda hardcoded en el repo (depende del account suffix).
    encrypt = true
  }
}
```

Como se usa (referencia — **NO correr a mano**): este es el patrón que
inyecta la task `infra:_init` (`tasks/infra.yml`) automáticamente cada vez
que disparás `task infra:plan`/`apply`/`validate`. Se muestra acá solo para
entender de dónde salen los 4 valores. En producción, el único comando
manual es `task infra:apply` (Parte 4.3 en adelante).

```bash
# Fuente: tasks/infra.yml :: _init (NO ejecutar copy-paste; lo hace la task)
BUCKET="${PROJECT}-tfstate-${ACCOUNT_SUFFIX}"
terraform init \
    -backend-config="bucket=${BUCKET}" \
    -backend-config="key=envs/prod/terraform.tfstate" \
    -backend-config="region=${AWS_DEFAULT_REGION}" \
    -backend-config="use_lockfile=true"
```

> **Nota — `use_lockfile=true`**: reemplaza al deprecado
> `-backend-config="dynamodb_table=${PROJECT}-tflock"`. Requiere Terraform
> >= 1.10. El lock se materializa como objeto `envs/prod/terraform.tfstate.tflock`
> en el mismo bucket — un PUT con header `If-None-Match: *` (preconditions
> nativas de S3) que falla si otro apply ya tiene el lock. Sin DynamoDB.

### 3.2.3 `infra/envs/prod/variables.tf`

```hcl
variable "project" {
  description = "Slug del proyecto (prefijo de todos los recursos)."
  type        = string
  default     = "ml-training"
}

variable "region" {
  description = "Region AWS para todo el deployment."
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR de la VPC. /16 da espacio para 65k IPs."
  type        = string
  default     = "10.20.0.0/16"
}

variable "alert_email" {
  description = "Email que recibe notificaciones SNS (job FAILED, MAPE high)."
  type        = string
}

variable "github_org" {
  description = "Organizacion / usuario GitHub que aloja el repo (para OIDC trust)."
  type        = string
}

variable "github_repo" {
  description = "Nombre del repo (sin la org). Para trust policy OIDC."
  type        = string
}

variable "varieties_allowed" {
  description = "Allow-list defensivo para el Lambda dispatcher (rechaza submits con variety no listada). NO define las variedades del modelo: la verdad esta en las hojas del Excel (data/BD_HISTORICO_ACUMULADO.xlsx) y se descubre dinamicamente con src/step_01_load/data_loader.py::list_varieties(). Esta lista solo previene typos en `aws lambda invoke`."
  type        = list(string)
  default     = ["POP", "JUPITER", "VENTURA", "SEKOYA", "ALLISON", "STELLA"]
}

variable "spot_max_vcpus" {
  description = "Maximo de vCPUs simultaneas en la queue Spot."
  type        = number
  default     = 16
}

variable "ondemand_max_vcpus" {
  description = "Maximo de vCPUs simultaneas en la queue On-Demand (solo prod_xl)."
  type        = number
  default     = 16
}

variable "batch_instance_type" {
  description = "Tipo de instancia EC2 que arranca Batch."
  type        = string
  default     = "c6i.2xlarge"
}

variable "rds_instance_class" {
  description = "Clase RDS. Hostea DOS bases: MLflow backend + `forecasts` de la API (Capa 4.5). db.t4g.small da holgura de RAM/conexiones para el stack completo; db.t4g.micro alcanza a muy bajo trafico."
  type        = string
  default     = "db.t4g.small"
}

variable "mlflow_image_tag" {
  description = "Tag de la imagen MLflow en ECR (build manual una vez)."
  type        = string
  default     = "v3.12.0"
}

variable "reports_image_tag" {
  description = "Tag de la imagen reports (nginx + s3-sync) en ECR."
  type        = string
  default     = "stable"
}

variable "trainer_image_tag" {
  description = "Tag de la imagen del trainer. CI/CD lo sobreescribe por commit SHA."
  type        = string
  default     = "latest"
}

variable "mape_alarm_threshold" {
  description = "Umbral de MAPE (%) para disparar alarma CloudWatch."
  type        = number
  default     = 25
}

variable "log_retention_days" {
  description = "Dias que CloudWatch retiene logs."
  type        = number
  default     = 14
}

variable "work_start_hour_local" {
  description = "Hora local de arranque del scheduler (PET, UTC-5)."
  type        = number
  default     = 8
}

variable "work_end_hour_local" {
  description = "Hora local de apagado del scheduler."
  type        = number
  default     = 12
}

variable "consumer_org" {
  description = "OPCIONAL/LEGACY (Patch 13.5): org de GitHub de un repo serving EXTERNO que asume el rol cross-repo via OIDC. La API+UI in-repo (Capa 4.5) NO usan esto."
  type        = string
}

variable "consumer_repo" {
  description = "Nombre del repo consumer (ej. ml_serving). Junto con consumer_org arma el subject del trust policy."
  type        = string
}

# ── App stack: API (FastAPI) + UI (Streamlit) ──────────────────────────────
variable "api_image_tag" {
  description = "Tag de la imagen de la API en ECR. CI/CD lo sobreescribe por commit SHA."
  type        = string
  default     = "latest"
}

variable "ui_image_tag" {
  description = "Tag de la imagen de la UI en ECR. CI/CD lo sobreescribe por commit SHA."
  type        = string
  default     = "latest"
}

variable "model_registry_prefix" {
  description = "Prefijo del registered model en MLflow. Debe coincidir con MODEL_REGISTRY_PREFIX del trainer (src/config.py)."
  type        = string
  default     = "rnd-forest-"
}

variable "api_preload_models" {
  description = "Precargar TODOS los modelos al boot de la API. false = lazy (recomendado en prod: arranque rapido y menos RAM)."
  type        = bool
  default     = false
}

# --- Capacidad: dimensionar segun necesidad (ver analisis de costo en GUIA) ---
# Combos Fargate validos: 1 vCPU (1024) admite 2-8 GB; 2 vCPU (2048) admite 4-16 GB.
# Default API 1 vCPU / 2 GB cubre lazy-load de ~6 variedades. Subir memory a 4096
# si se activa api_preload_models con muchas variedades, o cpu a 2048 para mas
# concurrencia. Cambiar aqui en tfvars NO requiere tocar codigo.
variable "api_cpu" {
  type    = number
  default = 1024
}
variable "api_memory" {
  type    = number
  default = 2048
}
variable "ui_cpu" {
  type    = number
  default = 512
}
variable "ui_memory" {
  type    = number
  default = 1024
}
```

### 3.2.4 `infra/envs/prod/terraform.tfvars` (NO COMMITEAR)

```hcl
alert_email = "abantodca@gmail.com"
github_org  = "abantodca"
github_repo = "ml_training"

# App stack (Capa 4.5: API + UI). Todas estas vars tienen default sano, asi que
# este bloque es OPCIONAL — descomenta solo lo que quieras overridear.
# rds_instance_class = "db.t4g.small"   # default ya es small (hostea MLflow + forecasts)
# api_cpu            = 1024              # subir a 2048 si activas preload con muchas variedades
# api_memory         = 2048             # subir a 4096 con api_preload_models = true
# api_preload_models = false            # true = carga todos los modelos al boot (mas RAM)

# Patch 13.5 — repo consumer EXTERNO que consume el Model Registry via OIDC.
# OPCIONAL / LEGACY: la API+UI ahora viven en ESTE monorepo (Capa 4.5), no en un
# repo aparte. Manten consumer_org/consumer_repo solo si todavia tenes un repo
# serving externo (ej. ml_serving) que descarga modelos read-only. Si no, podes
# quitar el `module "consumer_iam"` de main.tf (#3.11.5) y estas dos vars.
consumer_org  = "abantodca"      # <- tu org GH
consumer_repo = "ml_serving"     # <- nombre del repo consumer
```

Agregar a `.gitignore`:

```bash
cat >> .gitignore <<'EOF'

# Terraform
**/terraform.tfvars
**/.terraform/
**/.terraform.lock.hcl
*.tfstate
*.tfstate.*
.terraformrc
terraform.rc

# Lambdas .zip (Terraform los crea desde Python source)
infra/modules/lambdas/*.zip
infra/modules/scheduler/*.zip
EOF
```

### 3.2.5 `infra/envs/prod/main.tf` (esqueleto — crece incrementalmente)

> **Como leer esta seccion**: a diferencia de los `variables.tf` /
> `outputs.tf` que se pegan completos, este `main.tf` se **construye
> en partes** a medida que avanzas por la guia. Aca pegas solo el
> esqueleto (los `data` sources). Cada `module "X" {}` se va
> apendeando al final del archivo cuando llegues a la seccion del
> modulo correspondiente:
>
> | Bloque                | Se agrega en | Modulo creado en |
> |---|---|---|
> | `module "network"`    | #3.3.4       | #3.3             |
> | `module "storage"`    | #3.4.4       | #3.4             |
> | `module "mlflow"`     | #3.5.4       | #3.5             |
> | `module "reports"`    | #3.6.7       | #3.6             |
> | `module "batch"`      | #3.7.5       | #3.7             |
> | `module "monitoring"` | #3.8.4       | #3.8             |
> | `module "lambdas"`    | #3.9.7       | #3.9             |
> | `module "scheduler"`  | #3.10.5      | #3.10            |
> | `module "cicd"`       | #3.11.4      | #3.11            |
> | `module "consumer_iam"` (opcional) | #3.11.5.4 | #3.11.5  |
> | `module "api"` (Capa 4.5) | #3.12.9  | #3.12            |
> | `module "ui"` (Capa 4.5)  | #3.12.9  | #3.12            |
>
> **Por que incremental y no de un saque**: cada `module "X" {}`
> referencia outputs de modulos anteriores (e.g. `module.network.vpc_id`).
> Si pegas el `main.tf` completo antes de crear los modulos, `terraform
> validate` truena con "module not found" en cada uno. Apendear por
> capas mantiene el archivo siempre **valido** al terminar cada seccion
> — podes correr `terraform fmt` / `validate` checkpoint por checkpoint.
> La verificacion final integrada esta en #4.2 (justo antes del primer apply).

Pegar **solo** este contenido inicial:

```hcl
# OIDC provider de GitHub (creado en Parte 2.5, NO creado por Terraform).
# Si saltaste #2.5, este `data` falla con "no resource found" en plan.
# Pre-check antes de `terraform plan`:
#   aws iam list-open-id-connect-providers --query 'OpenIDConnectProviderList[?contains(Arn,`token.actions.githubusercontent.com`)]'
# Si devuelve [], correr `bash infra/bootstrap-oidc.sh` (#2.5).
data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}
```

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `data "aws_iam_openid_connect_provider"` | **IAM** | `IAM > Identity providers` → veras `token.actions.githubusercontent.com` (creado por `bootstrap-oidc.sh` en #2.5). El `data` lo "lee" para que `module.cicd` (#3.11) pueda asignar trust policies a los roles GHA sin hardcodear el ARN. |
>
> **Nota**: el root `main.tf` ya **no** declara `data "aws_caller_identity"` ni
> `data "aws_region"` compartidos — quedaron sin uso al nivel de composicion.
> Cada modulo que necesita Account ID / region (storage, mlflow, reports, batch,
> lambdas, cicd) declara su propio `data` internamente para construir ARNs y
> regiones de logs.
>
> **Conceptualmente — `data` vs `resource`**: `data` source = lectura
> de algo que **ya existe** (creado fuera de Terraform o por otro
> stack). `resource` = Terraform **gestiona el ciclo de vida**
> (create/update/destroy). Por eso el OIDC provider esta como `data`:
> lo creamos a mano en #2.5 con `bootstrap-oidc.sh` para que cualquier
> `terraform destroy` accidental no te tire la confianza GHA-AWS
> (recrearlo cuesta rotar `vars.AWS_GHA_DEPLOY_ROLE_ARN` y arreglar
> branch protection — friccion innecesaria).

### 3.2.6 `infra/envs/prod/outputs.tf`

```hcl
output "alb_dns" {
  description = "DNS publico del ALB (MLflow + Reports + API + UI)."
  value       = module.mlflow.alb_dns
}

output "tracking_uri" {
  description = "URL completa para MLFLOW_TRACKING_URI."
  value       = module.mlflow.tracking_uri
}

output "ecr_trainer_url" {
  description = "URL del repo ECR del trainer (para docker push)."
  value       = module.storage.ecr_trainer_url
}

output "ecr_mlflow_url" {
  description = "URL del repo ECR del MLflow custom."
  value       = module.storage.ecr_mlflow_url
}

output "ecr_reports_url" {
  description = "URL del repo ECR del reports nginx."
  value       = module.storage.ecr_reports_url
}

output "ecr_api_url" {
  description = "URL del repo ECR de la API (FastAPI)."
  value       = module.storage.ecr_api_url
}

output "ecr_ui_url" {
  description = "URL del repo ECR de la UI (Streamlit)."
  value       = module.storage.ecr_ui_url
}

output "ui_url" {
  description = "URL publica de la UI (Streamlit) detras del ALB."
  value       = "http://${module.mlflow.alb_dns}${module.ui.app_path}"
}

output "api_docs_url" {
  description = "URL publica del Swagger de la API."
  value       = "http://${module.mlflow.alb_dns}/docs"
}

output "data_bucket" {
  value = module.storage.data_bucket
}

output "artifacts_bucket" {
  value = module.storage.artifacts_bucket
}

output "job_queue_spot" {
  value = module.batch.job_queue_spot
}

output "job_queue_ondemand" {
  value = module.batch.job_queue_ondemand
}

output "job_definition_name" {
  value = module.batch.job_definition_name
}

output "dispatcher_function_name" {
  value = module.lambdas.dispatcher_function_name
}

output "sns_topic_arn" {
  value = module.monitoring.sns_topic_arn
}

output "gha_deploy_role_arn" {
  description = "Role que asume GitHub Actions para `terraform apply`."
  value       = module.cicd.gha_deploy_role_arn
}

output "gha_train_role_arn" {
  description = "Role que asume GitHub Actions para invocar Lambda dispatcher."
  value       = module.cicd.gha_train_role_arn
}

# Patch 13.5
output "consumer_role_arn" {
  description = "Role que asume el repo consumer (ml_serving) via OIDC para descargar artifacts."
  value       = module.consumer_iam.consumer_role_arn
}
```

> **Gotcha #3.2**: `terraform.tfvars` debe existir con valores reales y NO commitearse (`.gitignore` debe contener `**/terraform.tfvars`). Sin el archivo `validate` aún pasa, pero `plan` falla con "No value for required variable".

## 3.3 `modules/network/` — VPC + subnets + NAT + SGs

Single-AZ a proposito (Sec 0.2 lockeada). El SG matrix es:

- `sg_alb`: ingress :80 from 0.0.0.0/0 (futuro: WAF + TLS, hardening)
- `sg_mlflow`: **dos reglas de ingress desde `sg_alb`**: :5000 (MLflow
  task escucha ahi) y :80 (modulo reports reusa este SG porque comparte
  ECS cluster; el container nginx escucha en :80). Nada desde 0.0.0.0/0.
- `sg_rds`: ingress :5432 from `sg_mlflow` + `sg_batch` (Batch necesita
  conectar a RDS para registrar runs via MLflow Python client)
- `sg_batch`: egress 443 a internet (S3, ECR, MLflow ALB)

### 3.3.1 `modules/network/variables.tf`

```hcl
variable "project" { type = string }
variable "vpc_cidr" { type = string }
```

### 3.3.2 `modules/network/main.tf`

Pegar los bloques siguientes **uno a continuacion del otro** en el
mismo archivo `modules/network/main.tf`. La separacion en sub-bloques
con `### 3.3.2.X` es solo para que puedas leer el "por que" de cada
pieza sin perderte; el archivo final es la concatenacion de los 5
bloques.

#### 3.3.2.a — Discovery de AZs + VPC

Necesitamos saber que AZs tiene esta region disponibles (sin
hardcodear `us-east-1a/b`, asi la guia funciona en cualquier region).
La VPC propia evita choques con default-VPC.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `data "aws_availability_zones"` | **EC2** | `EC2 > Account attributes > Availability Zones`. Lista las AZs disponibles (ej: `us-east-1a`, `us-east-1b`, `us-east-1c`...). El `data` es un "read-only lookup" — no crea nada, solo lee. |
> | `aws_vpc.main` | **VPC** | `VPC > Your VPCs > Create VPC`. **Name tag**: `ml-training-vpc`. **IPv4 CIDR**: `10.20.0.0/16` (var.vpc_cidr). **Tenancy**: default. **DNS hostnames + DNS resolution**: enabled. Una VPC es tu "red privada en AWS" — todo lo demas (subnets, EC2, RDS, Fargate) vive adentro. |
>
> **Conceptualmente**: una VPC es tu red privada en AWS; adentro definís subnets, route tables y security groups. El CIDR `10.20.0.0/16` da 65536 IPs. Usamos VPC propia (no la default) para aislamiento y no chocar con recursos preexistentes.

```hcl
data "aws_availability_zones" "available" { state = "available" }

locals {
  # Solo 2 AZs (la "AZ secundaria" se reserva por RDS multi-AZ futuro)
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "${var.project}-vpc" }
}
```

#### 3.3.2.b — Subnets (public + private, x2 AZs)

2 public (ALB y NAT) + 2 private (Fargate, Batch, RDS). El offset `+10`
en cidrsubnet evita que los rangos public y private se toquen — facilita
debugging cuando ves una IP en CloudTrail.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_subnet.public[0..1]` | **VPC** | `VPC > Subnets > Create subnet`. **VPC**: la que creaste arriba. **Name**: `ml-training-public-0` / `-1`. **AZ**: una distinta por subnet (`us-east-1a`, `us-east-1b`). **CIDR**: `10.20.0.0/24` y `10.20.1.0/24`. Despues edit → `Auto-assign IPv4`: ON. |
> | `aws_subnet.private[0..1]` | **VPC** | Mismo wizard pero **CIDR**: `10.20.10.0/24` y `10.20.11.0/24`. **Auto-assign IPv4**: OFF. |
>
> **Conceptualmente** — la distinción public/private es CRÍTICA:
> - **Public** (`0.0.0.0/0 → IGW`): sale a Internet Y es alcanzable desde Internet. Aquí viven el ALB y la NAT.
> - **Private** (`0.0.0.0/0 → NAT`): sale a Internet (`docker pull` de ECR, CloudWatch) pero **no es alcanzable** desde afuera. Aquí viven MLflow Fargate, Batch y RDS.
> - **Por qué 2 AZs**: el ALB exige 2 subnets en AZs distintas; el resto es single-AZ a propósito (NAT, RDS).

```hcl
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index) # 10.20.0.0/24, 10.20.1.0/24
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.project}-public-${count.index}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10) # 10.20.10.0/24, 10.20.11.0/24
  availability_zone = local.azs[count.index]
  tags              = { Name = "${var.project}-private-${count.index}" }
}
```

#### 3.3.2.c — Internet Gateway + NAT (single, en public[0])

IGW para que las public subnets salgan a Internet. NAT (single, no HA)
para que las private salgan SIN ser alcanzables. NAT es **single porque
es el item caro** (~$32/mes); HA exigiria 2 NATs = $64/mes.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_internet_gateway.igw` | **VPC** | `VPC > Internet gateways > Create internet gateway`. **Name**: `ml-training-igw`. Despues `Actions > Attach to VPC > [tu VPC]`. Es el "portero de salida" para que cualquier IP publica de tu VPC pueda hablar con Internet. |
> | `aws_eip.nat` | **EC2** | `EC2 > Elastic IPs > Allocate Elastic IP address`. Es una IP publica fija — necesaria porque la NAT Gateway debe tener una IP estable para que el trafico de salida siempre se vea con el mismo origen. |
> | `aws_nat_gateway.main` | **VPC** | `VPC > NAT gateways > Create NAT gateway`. **Subnet**: la public-0 (tiene que estar en una subnet publica para acceder al IGW). **Elastic IP**: la que acabas de allocar. **Connectivity type**: Public. |
>
> **Conceptualmente — por qué IGW Y NAT**: hacen cosas opuestas. **IGW** = tráfico **bidireccional** (sirve al ALB). **NAT** = **solo saliente** (Fargate/Batch en private salen a `docker pull`/CloudWatch sin aceptar conexiones entrantes). Sin NAT, Batch no podría pullear de ECR ni postear runs.
>
> **Por qué NAT es el item caro**: ~$32/mes encendida + $0.045/GB procesado (~$0.27 por job que baja 6 GB). Reemplazable con **VPC Endpoints** (gratis para S3/ECR) → casi cero (hardening, futuro).

```hcl
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project}-igw" }
}

resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "${var.project}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  tags          = { Name = "${var.project}-nat" }
  depends_on    = [aws_internet_gateway.igw]
}
```

#### 3.3.2.d — Route tables

Public RT: 0.0.0.0/0 → IGW. Private RT: 0.0.0.0/0 → NAT. La
asociacion x2 vincula las subnets a su RT correspondiente.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_route_table.public` | **VPC** | `VPC > Route tables > Create route table`. **Name**: `ml-training-rt-public`. **VPC**: la tuya. Despues editar `Routes > Edit routes > Add route`: **Destination**: `0.0.0.0/0`, **Target**: Internet Gateway → seleccionar el IGW. |
> | `aws_route_table.private` | **VPC** | Mismo wizard, **Name**: `ml-training-rt-private`. **Route**: `0.0.0.0/0` → NAT Gateway. |
> | `aws_route_table_association.*` | **VPC** | En cada subnet: `Subnet > Edit route table association > [seleccionar RT]`. Esto le dice a cada subnet "para salir a Internet, usa este camino". |
>
> **Conceptualmente**: las route tables son el "GPS" de la VPC. Toda subnet tiene una RT (si no, hereda la main RT). Una subnet es "public" PORQUE su RT apunta `0.0.0.0/0 → IGW`, no por nada en la subnet misma.

```hcl
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "${var.project}-rt-public" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
  tags = { Name = "${var.project}-rt-private" }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}
```

#### 3.3.2.e — Security Groups (6: alb, mlflow, batch, rds, api, ui)

6 SGs en cascada (alb es la unica que acepta 0.0.0.0/0; las otras
solo aceptan trafico desde la anterior, formando una cadena de
defense-in-depth):

- `sg-alb`: Internet → :80.
- `sg-mlflow`: sg-alb → :5000 (MLflow) y :80 (reports), **+ sg-api → :5000**
  (la API consulta el Model Registry interno via service discovery, sin
  pasar por el ALB). Compartido por MLflow y reports porque ambos viven
  detras del mismo ALB.
- `sg-batch`: solo egress (el trainer pulea de S3/ECR, escribe logs,
  postea a MLflow ALB). No acepta ingress de ningun lado.
- `sg-rds`: 5432 desde sg-mlflow + sg-batch **+ sg-api** (la API persiste
  pronosticos en la base `forecasts`, que vive en el MISMO RDS de MLflow).
- `sg-api` (App stack, Capa 4.5): :8000 desde sg-alb (`/api/*` y `/docs`)
  **+ desde sg-ui** (la UI llama a la API server-side por service discovery).
- `sg-ui` (App stack, Capa 4.5): :8501 desde sg-alb (`/app/*`). La UI no
  recibe trafico de nadie mas.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_security_group.alb` | **VPC** | `VPC > Security groups > Create security group`. **Name**: `ml-training-sg-alb`. **VPC**: la tuya. **Inbound rules > Add rule**: Type=HTTP, Source=Anywhere-IPv4 (`0.0.0.0/0`). **Outbound rules**: All traffic → 0.0.0.0/0 (default). |
> | `aws_security_group.mlflow` | **VPC** | Mismo wizard. **Inbound rules**: dos reglas → (1) Custom TCP :5000 con Source=`sg-alb` (escribis el ID, no un CIDR); (2) HTTP :80 con Source=`sg-alb`. |
> | `aws_security_group.batch` | **VPC** | **Inbound rules**: **vacio** (nadie debe poder conectarse a los jobs). **Outbound**: All traffic. |
> | `aws_security_group.rds` | **VPC** | **Inbound rules**: dos reglas → (1) PostgreSQL :5432 con Source=`sg-mlflow`; (2) PostgreSQL :5432 con Source=`sg-batch`. **Outbound**: vacio (RDS no necesita salir a nada). |
>
> **Conceptualmente — SGs son firewalls "stateful" a nivel recurso**:
> - **Stateful**: si permitís el ingreso, la respuesta saliente se permite sola (a diferencia de los NACLs stateless).
> - En `Source` podés poner **otro SG** en vez de un CIDR: `sg-rds` acepta a `sg-mlflow`/`sg-batch` aunque sus IPs cambien (Fargate las reasigna).
> - **Cadena de defense-in-depth**: Internet → ALB → MLflow → RDS. Romper el ALB no da acceso directo a RDS — corta el lateral movement a nivel red.

```hcl
resource "aws_security_group" "alb" {
  name        = "${var.project}-sg-alb"
  description = "ALB: 80/HTTP desde Internet (TLS futuro en Parte 10.1)"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "mlflow" {
  name   = "${var.project}-sg-mlflow"
  vpc_id = aws_vpc.main.id

  ingress {
    description     = "MLflow server desde ALB"
    from_port       = 5000
    to_port         = 5000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  ingress {
    description     = "Reports nginx desde ALB"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  ingress {
    description     = "MLflow desde la API (service discovery interno)"
    from_port       = 5000
    to_port         = 5000
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "batch" {
  name   = "${var.project}-sg-batch"
  vpc_id = aws_vpc.main.id

  egress {
    description = "Egress libre (S3, ECR, MLflow ALB, CloudWatch Logs)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "rds" {
  name   = "${var.project}-sg-rds"
  vpc_id = aws_vpc.main.id

  ingress {
    description     = "Postgres desde MLflow Fargate"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.mlflow.id]
  }
  ingress {
    description     = "Postgres desde Batch (trainer logging directo)"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.batch.id]
  }
  ingress {
    description     = "Postgres desde la API (base forecasts en el mismo RDS)"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
  }
}

# ── SGs de la app stack (API + UI) ─────────────────────────────────────────
# Dedicados (no se reusa sg-mlflow) para reglas explicitas y minimas.
resource "aws_security_group" "api" {
  name        = "${var.project}-sg-api"
  description = "API FastAPI: 8000 desde ALB y desde la UI"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "API desde ALB (/api/* y /docs)"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  ingress {
    description     = "API desde la UI (llamada interna server-side)"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.ui.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ui" {
  name        = "${var.project}-sg-ui"
  description = "UI Streamlit: 8501 desde ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "UI desde ALB (/app/*)"
    from_port       = 8501
    to_port         = 8501
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

> **Checkpoint despues de pegar los 7 bloques**: ejecuta
> `terraform fmt infra/modules/network/main.tf` para confirmar que el
> archivo es sintacticamente valido. Si reformatea, OK; si imprime
> error de parse, falta pegar un `}` o uniste dos resources sin
> separador.

### 3.3.3 `modules/network/outputs.tf`

```hcl
output "vpc_id" { value = aws_vpc.main.id }
output "public_subnet_ids" { value = aws_subnet.public[*].id }
output "private_subnet_ids" { value = aws_subnet.private[*].id }
output "sg_alb_id" { value = aws_security_group.alb.id }
output "sg_mlflow_id" { value = aws_security_group.mlflow.id }
output "sg_batch_id" { value = aws_security_group.batch.id }
output "sg_rds_id" { value = aws_security_group.rds.id }
output "sg_api_id" { value = aws_security_group.api.id }
output "sg_ui_id" { value = aws_security_group.ui.id }
```

> **En consola AWS veras** despues del apply:
> - VPC → Your VPCs → `ml-training-vpc` (CIDR 10.20.0.0/16).
> - VPC → Subnets → 4 subnets: 2 public (`ml-training-public-0/1`) en
>   AZ-a/AZ-b + 2 private (`ml-training-private-0/1`).
> - VPC → NAT Gateways → 1 NAT en public[0] (`state=available`,
>   `eip=<X>`). **Cuesta ~$32/mes** + traffic — es el item caro de
>   este modulo.
> - VPC → Internet Gateways → 1 IGW.
> - VPC → Route Tables → 2 (public via IGW, private via NAT).
> - EC2 → Security Groups → 6 con tag `Project=ml-training`:
>   `sg-alb` (ingress 80 desde 0.0.0.0/0), `sg-mlflow` (ingress 80
>   desde sg-alb + 5000 desde sg-api), `sg-batch` (egress all),
>   `sg-rds` (ingress 5432 desde sg-mlflow + sg-batch + sg-api),
>   `sg-api` (ingress 8000 desde sg-alb + sg-ui), `sg-ui` (ingress
>   8501 desde sg-alb).

### 3.3.4 Apendear `module "network"` en `infra/envs/prod/main.tf`

Ahora que el modulo `network` esta definido (#3.3.2) y expone sus
outputs (#3.3.3), lo **cableamos desde el env `prod`**. Pegar AL
FINAL de `infra/envs/prod/main.tf` (a continuacion del bloque
`data` de #3.2.5):

```hcl
# -------------------------------------------------------------------------
# Capa 1: Red (VPC + subnets + NAT + SGs)
# -------------------------------------------------------------------------
module "network" {
  source   = "../../modules/network"
  project  = var.project
  vpc_cidr = var.vpc_cidr
}
```

> **Checkpoint**: con esto el `main.tf` ya es valido. Podes correr
> `terraform fmt` y `terraform validate` (no `plan` todavia — falta
> el resto de los modulos). Si valida → seguir a #3.4.

> **Gotcha #3.3**: `security_groups.tf` debe quedar como archivo separado (no fundido en `main.tf`). `validate` pasa igual pero el diff vs repo crece (ver #3.1 layout).

---

## 3.4 `modules/storage/` — S3 buckets + ECR repos

### 3.4.1 `modules/storage/variables.tf`

```hcl
variable "project" { type = string }
```

### 3.4.2 `modules/storage/main.tf`

Pegar los bloques siguientes **uno a continuacion del otro** en el
mismo archivo `modules/storage/main.tf`. La separacion en sub-bloques
con `### 3.4.2.X` es solo para que puedas leer el "por que" de cada
pieza sin perderte; el archivo final es la concatenacion de los 5
bloques.

#### 3.4.2.a — Header: account suffix discovery

Calcula el sufijo de 7 chars que comparten todos los buckets (data,
artifacts) y el bucket de tfstate creado a mano por `bootstrap.sh`.
Equivalente bash: `${ACCOUNT: -7}` (idéntico a `scripts/aws-suffix.sh`,
fuente única usada por `task local:ensure-buckets`). Asi un mismo
`account_id` produce el mismo sufijo en todos los buckets — local y
prod comparten nombre, y operativamente no te confundis entre "cual
era el bucket de este proyecto".

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `data "aws_caller_identity"` | **IAM / STS** | `IAM > Dashboard` muestra arriba a la derecha tu **Account ID** de 12 digitos. El `data` es read-only — no crea nada, solo "pregunta a AWS quien soy" via STS (`sts:GetCallerIdentity`). |
> | `locals.account_suffix` | — | No tiene UI: es compute puro de Terraform. Toma los ultimos 7 chars del account_id para usarlos de sufijo de bucket. |
>
> **Conceptualmente — por qué un sufijo y no el nombre crudo**: los nombres de bucket S3 son **globalmente únicos** en todo S3. Sin sufijo, un segundo `terraform apply` con `project=ml-training` fallaría con "bucket already exists". El sufijo de 7 chars del account_id lo hace **prácticamente único** y a la vez **determinístico** dentro de una cuenta (no random).

```hcl
data "aws_caller_identity" "current" {}

locals {
  # ${ACCOUNT: -7} en bash. substr(...,5,7) toma chars 5-11 (indices 0-based)
  # = los ultimos 7 chars de un account_id estandar de 12 digitos.
  # Coincide con scripts/aws-suffix.sh (POSIX `${acct#?????}`).
  account_suffix = substr(data.aws_caller_identity.current.account_id, 5, 7)
}
```

#### 3.4.2.b — S3 bucket `data` (input Excels) + hardening

El bucket donde subis el Excel acumulado (`BD_HISTORICO_ACUMULADO.xlsx`)
antes de cada training. Los 3 sub-recursos (versioning, encryption,
public-block) son **obligatorios** en cualquier bucket post-2023 —
defaults seguros + auditoria.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_s3_bucket.data` | **🪣 S3** | `S3 > Buckets > Create bucket`. **Name**: `ml-training-data-<sufijo>`. **Region**: `us-east-1`. **ACLs**: disabled. **Object Ownership**: ACLs disabled, Bucket owner enforced. |
> | `aws_s3_bucket_versioning.data` | **🪣 S3** | Mismo bucket → `Properties > Bucket Versioning > Edit > Enable`. Mismo mecanismo de rollback que ya explicado para el tfstate bucket en #2.2 (Equivalente en AWS Console, paso 2). |
> | `aws_s3_bucket_server_side_encryption_configuration.data` | **🪣 S3** | `Properties > Default encryption > Edit > AES-256 (SSE-S3)`. **Bucket Key**: Enable (reduce costos de KMS si en el futuro migras a SSE-KMS). |
> | `aws_s3_bucket_public_access_block.data` | **🪣 S3** | `Permissions > Block public access > Edit > Block all public access`. Las 4 sub-opciones ON: bloquea ACLs publicas y bucket policies publicas, tanto las que existen como las que se intenten crear. |
>
> **Conceptualmente — por qué 4 recursos Terraform para "un bucket"**: la API REST de S3 expone cada faceta como un sub-endpoint (`PUT /?versioning`, `/?encryption`, `/?publicAccessBlock`) y Terraform la refleja 1-a-1. **Versioning** = rollback si suben un Excel roto; **AES-256** = cumple compliance sin costo; **public access block** = la red #1 contra el bucket público por error.

```hcl
resource "aws_s3_bucket" "data" {
  bucket = "${var.project}-data-${local.account_suffix}"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

#### 3.4.2.c — S3 bucket `artifacts` (modelos + reportes + MLflow store) + lifecycle

Almacen central de outputs: modelos `.joblib`, JSONs de metricas,
dashboards HTML, y el "artifact store" de MLflow (cuando
`mlflow.log_artifact()` sube algo, va aca). La lifecycle policy borra
**versiones no-current** a los 90 dias para que el bill S3 no se infle
indefinidamente.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_s3_bucket.artifacts` | **🪣 S3** | `Create bucket` con nombre `ml-training-artifacts-<sufijo>`. Mismas settings que el bucket data. |
> | `aws_s3_bucket_versioning.artifacts` + `_server_side_encryption_configuration` + `_public_access_block` | **🪣 S3** | Mismos 3 sub-recursos de hardening (versioning, encryption AES-256, public access block) — identico al bloque .b. |
> | `aws_s3_bucket_lifecycle_configuration.artifacts` | **🪣 S3** | `Management > Lifecycle rules > Create lifecycle rule`. **Name**: `expire-noncurrent`. **Scope**: Apply to all objects. **Permanently delete noncurrent versions**: after **90 days**. **Abort incomplete multipart uploads**: after 7 days. |
>
> **Que guarda este bucket** — estructura tipica:
> - `artifacts/POP/final_pipeline_POP_v1.joblib` (el modelo entrenado)
> - `artifacts/POP/run_summary_POP.json` (metricas del run)
> - `reports/POP/dashboard.html` (dashboards interactivos)
> - Internamente MLflow tambien apunta sus `artifact_uri` aca.
>
> **Por qué 90 días y no 30 ni 365**: tres meses cubren un ciclo de A/B testing entre modelos. Más corto pierde auditoría de incidentes; más largo infla el bill sin valor. Nota: la lifecycle solo borra **noncurrent versions** — la versión actual del modelo nunca se borra automáticamente.
>
> **Por qué `abort_incomplete_multipart_upload`**: si la subida de un archivo grande se corta a la mitad, S3 cobra storage por los chunks parciales invisibles. La regla los limpia a los 7 días — seguro barato contra "fantasmas" en el bill.

```hcl
resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project}-artifacts-${local.account_suffix}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-noncurrent"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration { noncurrent_days = 90 }
    abort_incomplete_multipart_upload { days_after_initiation = 7 }
  }
}
```

#### 3.4.2.d — ECR `trainer` + lifecycle policy

El "Docker Hub privado" donde vive la imagen del trainer
(`ml-training:v0.1.0`, `:sha-abc123`, `:latest`). El job de Batch hace
`docker pull` desde aca cuando arranca. La lifecycle policy evita que
ECR acumule decenas de GB de imagenes viejas.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_ecr_repository.trainer` | **ECR** | `ECR > Private repositories > Create repository`. **Name**: `ml-training`. **Tag immutability**: `Mutable`. **Scan on push**: Enabled. **Encryption**: AES-256. |
> | `aws_ecr_lifecycle_policy.trainer` | **ECR** | `ECR > [repo trainer] > Lifecycle Policy > Edit > Add rule`. Definir 2 reglas: Priority 1 = keep last 10 tagged `v*`/`sha-*`, Priority 2 = expire untagged > 7 days. La consola muestra preview de "que imagenes se borrarian con esta regla". |
>
> **Conceptualmente — MUTABLE vs IMMUTABLE**: `Mutable` deja que la **misma tag** apunte a otra imagen (ej: `latest` se mueve en cada push) — útil para CI/CD, pero un `docker pull :latest` hoy y mañana trae imágenes distintas (trampa para debug). Por eso siempre tagueamos también con el sha del commit (`sha-abc123`), inmutable de facto.
>
> **Por qué la lifecycle**: cada imagen pesa ~1-2 GB; 50 versiones sin limpiar = ~75 GB (~$7.50/mes). La policy mantiene ~20 GB (~$2/mes). Las reglas con `rulePriority` se evalúan ascendente: primero "keep last 10 tagged", el resto cae en "expire untagged > 7 days".
>
> **Sobre `scan_on_push`**: ECR escanea CVEs de la imagen subida (`ECR > [repo] > Images > Vulnerabilities`), pero **no bloquea el push** — es informativo. Bloquear requiere un step extra en CI (`aws ecr describe-image-scan-findings`).

```hcl
resource "aws_ecr_repository" "trainer" {
  name                 = var.project
  image_tag_mutability = "MUTABLE" # CI/CD reusa tag "latest" + sha
  force_delete         = true      # destroy borra el repo aunque tenga imagenes
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
}

resource "aws_ecr_lifecycle_policy" "trainer" {
  repository = aws_ecr_repository.trainer.name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 tagged"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v", "sha-"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Expire untagged > 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = { type = "expire" }
      }
    ]
  })
}
```

#### 3.4.2.e — ECR `mlflow` (IMMUTABLE) + `reports` + `api` + `ui` (MUTABLE)

Cuatro repos mas. `mlflow` va con politica **opuesta** de tag immutability
(deliberado: el binario oficial nunca debe sobrescribirse); `reports`, `api`
y `ui` son codigo nuestro que iteramos seguido → MUTABLE con lifecycle (keep
last 10 tags + expira untagged). Los repos `api`/`ui` son parte del App stack
(Capa 4.5) que se levanta junto con MLflow.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_ecr_repository.mlflow` | **ECR** | `Create repository`. **Name**: `ml-training-mlflow`. **Tag immutability**: **`IMMUTABLE`** (importante!). Scan on push: Enabled. Encryption: AES-256. |
> | `aws_ecr_repository.reports` | **ECR** | `Create repository`. **Name**: `ml-training-reports`. **Tag immutability**: `Mutable`. Scan on push: Enabled. Encryption: AES-256. |
>
> **Por qué MLflow va IMMUTABLE**: es un release oficial verificado. Si alguien sobrescribiera la tag `v3.12.0`, el ALB serviría una versión no auditada. **IMMUTABLE** = AWS rechaza cualquier push que reuse una tag existente, a nivel API — ni un admin puede sobrescribir.
>
> **Por qué reports queda MUTABLE**: es código nuestro (nginx + entrypoint.sh) que iteramos seguido; re-pushear `:latest` es normal y no sirve tráfico crítico como MLflow.

```hcl
resource "aws_ecr_repository" "mlflow" {
  name                 = "${var.project}-mlflow"
  image_tag_mutability = "IMMUTABLE" # v3.12.0 nunca cambia
  force_delete         = true        # destroy borra el repo aunque tenga imagenes
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
}

resource "aws_ecr_repository" "reports" {
  name                 = "${var.project}-reports"
  image_tag_mutability = "MUTABLE" # iteramos nginx.conf seguido
  force_delete         = true      # destroy borra el repo aunque tenga imagenes
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
}

resource "aws_ecr_repository" "api" {
  name                 = "${var.project}-api"
  image_tag_mutability = "MUTABLE" # CI/CD reusa "latest" + sha por commit
  force_delete         = true
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 tagged"
        selection    = { tagStatus = "tagged", tagPrefixList = ["v", "sha-"], countType = "imageCountMoreThan", countNumber = 10 }
        action       = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Expire untagged > 7 days"
        selection    = { tagStatus = "untagged", countType = "sinceImagePushed", countUnit = "days", countNumber = 7 }
        action       = { type = "expire" }
      }
    ]
  })
}

resource "aws_ecr_repository" "ui" {
  name                 = "${var.project}-ui"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
}

resource "aws_ecr_lifecycle_policy" "ui" {
  repository = aws_ecr_repository.ui.name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 tagged"
        selection    = { tagStatus = "tagged", tagPrefixList = ["v", "sha-"], countType = "imageCountMoreThan", countNumber = 10 }
        action       = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Expire untagged > 7 days"
        selection    = { tagStatus = "untagged", countType = "sinceImagePushed", countUnit = "days", countNumber = 7 }
        action       = { type = "expire" }
      }
    ]
  })
}
```

> **Checkpoint despues de pegar los bloques**: ejecuta
> `terraform fmt infra/modules/storage/main.tf` para confirmar que el
> archivo es sintacticamente valido. Si reformatea, OK; si imprime
> error de parse, falta pegar un `}` o uniste dos resources sin
> separador.

### 3.4.3 `modules/storage/outputs.tf`

```hcl
output "data_bucket" { value = aws_s3_bucket.data.bucket }
output "data_bucket_arn" { value = aws_s3_bucket.data.arn }
output "artifacts_bucket" { value = aws_s3_bucket.artifacts.bucket }
output "artifacts_bucket_arn" { value = aws_s3_bucket.artifacts.arn }

output "ecr_trainer_url" { value = aws_ecr_repository.trainer.repository_url }
output "ecr_mlflow_url" { value = aws_ecr_repository.mlflow.repository_url }
output "ecr_reports_url" { value = aws_ecr_repository.reports.repository_url }
output "ecr_api_url" { value = aws_ecr_repository.api.repository_url }
output "ecr_ui_url" { value = aws_ecr_repository.ui.repository_url }
```

> **En consola AWS veras**:
> - S3 → Buckets → `ml-training-data-<suffix>` (vacio; Excel se sube
>   en Ola A) y `ml-training-artifacts-<suffix>` (artifacts + reports
>   + MLflow artifact store). Ambos con Versioning=Enabled, Encryption
>   AES256, Block public access ON.
> - S3 → Bucket `ml-training-artifacts-...` → Management → Lifecycle
>   rule → "expira versiones non-current a los 90 dias".
> - ECR → Repositories → 5: `ml-training`, `ml-training-mlflow`,
>   `ml-training-reports`, `ml-training-api`, `ml-training-ui` (vacios
>   hasta Ola B). Cada uno con scan-on-push y lifecycle policy (keep
>   last 10 tags + borrar untagged >7 dias); solo `ml-training-mlflow`
>   es IMMUTABLE.

### 3.4.4 Apendear `module "storage"` en `infra/envs/prod/main.tf`

Mismo patron: pegar AL FINAL de `infra/envs/prod/main.tf`
(despues del bloque `module "network"` de #3.3.4):

```hcl
# -------------------------------------------------------------------------
# Capa 2: Storage (S3 buckets + ECR repos)
# -------------------------------------------------------------------------
module "storage" {
  source  = "../../modules/storage"
  project = var.project
}
```

> **Checkpoint**: `terraform fmt && terraform validate` debe pasar.
> Storage es independiente de network (no comparte inputs) — por eso
> en Parte 4 hay una "Ola A" que aplica storage **solo**, antes que
> todo lo demas (#4.2). Asi tenes ECR listo para hacer `docker push`
> antes de levantar ECS.

> **Gotcha #3.4**: nombres de buckets S3 con uppercase o underscores rompen `validate` con `BucketName ... is not valid`. Mantener `lowercase` + guión.

---

## 3.4.5 `modules/_shared/` — Trust policies compartidos

Documentos de assume-role JSON que varios modulos repetian. Cada modulo los
carga con `file()` o `templatefile()` en vez de redeclarar el mismo `data
"aws_iam_policy_document"`. Cambio puramente de organizacion: AWS provider
normaliza JSON, asi que `terraform plan` queda no-op.

**Archivos a crear en este paso** (los contenidos completos estan en #3.4.5.1):

- `infra/modules/_shared/assume-ecs-tasks.json`      — Fargate / ECS task roles (mlflow, reports, batch job role)
- `infra/modules/_shared/assume-lambda.json`         — Lambda execution roles (dispatcher, notifier, scheduler)
- `infra/modules/_shared/assume-ec2.json`            — EC2 instance profile (batch compute env)
- `infra/modules/_shared/assume-batch-service.json`  — AWS Batch service role
- `infra/modules/_shared/assume-github-oidc.json.tftpl` — GHA OIDC trust (cicd + consumer-iam), parametriza provider_arn/org/repo
- `infra/modules/_shared/README.md`                  — documentacion inline del directorio

> **⚠️ Importante**: en esta seccion solo creas los **6 archivos** de arriba.
> Los dos bloques HCL que siguen son **ejemplos de referencia** — te muestran
> como los modulos `mlflow`, `batch`, `lambdas`, `cicd` y `consumer-iam` van
> a consumir estos JSON mas adelante (#3.5, #3.7, #3.8, #3.11). NO los pegues
> en ningun `.tf` ahora; el codigo de cada modulo ya viene con esas referencias
> cableadas cuando llegues a su seccion.

**Ejemplo de uso (referencia — aparece dentro de cada modulo en #3.5+, no lo crees aqui):**

```hcl
resource "aws_iam_role" "ejemplo" {
  name               = "${var.project}-ejemplo"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}
```

**Ejemplo del trust GHA-OIDC con variables interpoladas (tambien referencia, vive en `cicd`/`consumer-iam`):**

```hcl
locals {
  gha_oidc_trust = templatefile("${path.module}/../_shared/assume-github-oidc.json.tftpl", {
    provider_arn = var.oidc_provider_arn
    org          = var.github_org
    repo         = var.github_repo
  })
}
```

Si en el futuro necesitas un trust nuevo (ej. RDS, EventBridge), agregalo aqui
en vez de inlinearlo en el modulo.

### 3.4.5.1 Contenido de los archivos

`modules/_shared/assume-ecs-tasks.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      }
    }
  ]
}
```

`modules/_shared/assume-lambda.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      }
    }
  ]
}
```

`modules/_shared/assume-ec2.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      }
    }
  ]
}
```

`modules/_shared/assume-batch-service.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Principal": {
        "Service": "batch.amazonaws.com"
      }
    }
  ]
}
```

`modules/_shared/assume-github-oidc.json.tftpl` (template — `${provider_arn}`,
`${org}` y `${repo}` se interpolan via `templatefile()`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Principal": {
        "Federated": "${provider_arn}"
      },
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:${org}/${repo}:*"
        }
      }
    }
  ]
}
```

> **Gotcha #3.4.5**: los JSON deben validar como JSON estricto (sin comentarios, sin trailing commas). Diagnostico: `jq . infra/modules/_shared/assume-ecs-tasks.json` no debe errorizar.

---

## 3.5 `modules/mlflow/` — RDS + ECS Fargate + ALB

Este modulo es el mas pesado: arma el backend de tracking (RDS Postgres),
el server MLflow en Fargate y el ALB que expone todo. Aca lockean dos
contratos criticos del codigo del trainer:

- El argumento `--allowed-hosts '*'` (wildcard) evita el bug #11.2 de
  V1 (403 "Invalid Host header"). Es **wildcard intencional** porque
  el ALB DNS no se conoce en tiempo de `terraform plan`. Refinable
  post-stand-up: una vez que tenes el ALB DNS, podrias pasar a lista
  especifica `--allowed-hosts <alb-dns>,mlflow.ml-training.local,localhost` (ver
  hardening, futuro).
- El usuario Postgres se llama `mlflow` y la DB se llama `mlflow`
  (igual que en docker-compose local, para que el trainer no tenga que
  cambiar la connection string entre local y prod).

### 3.5.0 Decisiones de arquitectura (rationale para experto)

Tres elecciones marcan el caracter de este modulo y vale la pena
justificarlas antes de leer el HCL:

**Fargate vs EC2 para el server MLflow.** El server MLflow es un
proceso Python single-threaded con throughput modesto (decenas de
requests por minuto en nuestro uso: el trainer lo golpea unas pocas
veces por run, y los humanos abren la UI esporadicamente). Fargate
da: zero gestion de hosts, scale-to-zero para el scheduler, e
integracion nativa con IAM Task Roles (credenciales rotadas
automaticamente, sin secrets en disco). El trade-off contra EC2 es
~30% mas caro por vCPU/hora **mientras corre**, pero como el
scheduler lo apaga 80% del tiempo (16h/dia + fines de semana), en
total Fargate sale **mas barato** que una EC2 t3.small reservada 24/7.
EC2 solo gana cuando necesitas GPU o sostenidamente >10 req/s — no
es nuestro caso.

**RDS Postgres single-AZ vs Aurora Serverless v2.** Aurora Serverless
v2 fue el candidato natural: escala a 0.5 ACU minima y matchea bien
el patron stop/start. El descarte fue economico: ACU minima costaba
~$45/mes solo en compute, mas storage; RDS `db.t4g.small` cuesta
~$23/mes corriendo y $0 stopped (que es como pasa la mayoria del
tiempo gracias al scheduler de #3.10). Para un Postgres que recibe
~1000 inserts por entrenamiento y queries esporadicas de la UI, la
elasticidad de Aurora no compensa el piso de precio. Multi-AZ
tampoco aplica: el trainer es idempotente y reproducible, la perdida
de tracking metadata no rompe modelos productivos — solo perdes
historial, recuperable desde los `joblib` + `run_summary.json` en S3.
Si en el futuro el equipo crece y la UI se vuelve critica, migrar a
Aurora es trivial (`engine = "aurora-postgresql"` + cambio de
instance_class).

**ALB publico vs interno.** El ALB nace **publico** (`internal =
false`) por simplicidad operativa: el dev abre `http://<alb-dns>/`
desde el browser sin VPN, GitHub Actions golpea `<alb-dns>/api/...`
desde sus runners managed sin route adicional. La superficie de
ataque se mitiga con Security Groups (solo :80 desde Internet) y
auth basica en la app (futuro). **Post-stand-up es candidato a
hardening**: una vez que tenes VPN corporativa o AWS Client VPN
configurada, pasar a `internal = true` + Route53 record privado es
una operacion de 30 min (hardening, futuro). La promesa de
"ALB interno-only" se cumple **entonces**, no antes.

### 3.5.1 `modules/mlflow/variables.tf`

```hcl
variable "project" { type = string }
variable "vpc_id" { type = string }
variable "public_subnet_ids" { type = list(string) }
variable "private_subnet_ids" { type = list(string) }
variable "sg_alb_id" { type = string }
variable "sg_mlflow_id" { type = string }
variable "sg_rds_id" { type = string }
variable "rds_instance_class" { type = string }
variable "rds_allocated_storage_gb" {
  type    = number
  default = 20
}
variable "mlflow_image" { type = string }
variable "artifacts_bucket" { type = string }
variable "artifacts_bucket_arn" { type = string }
variable "log_retention_days" { type = number }
```

### 3.5.2 `modules/mlflow/` (split en 5 archivos)

Este modulo es el mas grande de la guia (~270 lineas). En el **repo
real** esta **split en 5 archivos** (consistente con #3.1):

- `rds.tf` ← sub-bloque 3.5.2.a (`data "aws_region"` + Postgres + Secrets Manager)
- `alb.tf` ← sub-bloque 3.5.2.b (ALB + target group + listener)
- `iam.tf` ← sub-bloque 3.5.2.d (roles ECS exec/task + policies)
- `ecs.tf` ← sub-bloques 3.5.2.c + 3.5.2.e (cluster + service discovery + task def + service)
- `main.tf` ← **vacio en el repo real** (solo existe como placeholder por convencion; el `data "aws_region" "current" {}` vive en `rds.tf`). Podes dejarlo sin crear o crearlo vacio.

Lee los 5 sub-bloques en orden, pegando cada uno **en su archivo
correspondiente** (cada bloque trae como primera linea un comentario
`# infra/modules/mlflow/<archivo>.tf` que indica el destino). Si
preferis un archivo unico, podes concatenar todos en `main.tf` y
borrar los otros — Terraform los procesa igual.

#### 3.5.2.a — `rds.tf` — RDS Postgres (password + subnet group + instance)

> 📂 **Pegar este bloque en**: `infra/modules/mlflow/rds.tf`
> (incluye el `data "aws_region" "current" {}` — en el repo real ese
> data source vive aca, no en `main.tf`).

`random_password` + Secrets Manager evita escribir el password en
tfstate (queda solo en SM). `subnet_group` en private subnets x2
porque RDS exige 2 AZs aunque sea single-AZ. `skip_final_snapshot=true`
+ `deletion_protection=false` son **inseguros para prod con datos
reales**: nos los dejamos asi mientras hay solo experimentos
descartables. Al primer modelo que importe en Production, ambos
flags **deben cambiarse** (hardening, futuro).

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `random_password.rds` | **(local Terraform)** | No es un recurso AWS — Terraform genera un string aleatorio en memoria. En Console serias VOS quien tipea un password en el wizard de RDS. |
> | `aws_secretsmanager_secret` + `_version` | **Secrets Manager** | `Secrets Manager > Secrets > Store a new secret > Other type`. **Key/value**: pega el password. **Name**: `ml-training-rds-password`. Encryption: `aws/secretsmanager` (default). |
> | `aws_db_subnet_group.mlflow` | **RDS** | `RDS > Subnet groups > Create DB subnet group`. **Name**: `ml-training-rds-subnets`. **VPC**: la tuya. **AZs**: las 2 que tenes. **Subnets**: las 2 private. |
> | `aws_db_instance.mlflow` | **RDS** | `RDS > Databases > Create database`. **Standard create > PostgreSQL > 15.x**. **Templates**: Production (o Dev/Test si querés single-AZ). **DB identifier**: `ml-training-mlflow`. **Master username**: `mlflow`. **Master password**: pega el secret. **Instance class**: `db.t4g.small`. **Storage**: 20 GB gp3. **Connectivity > VPC**: la tuya. **Subnet group**: el creado arriba. **VPC SGs**: `sg-rds`. **Public access**: No. **Backup retention**: 7 days. |
>
> **Conceptualmente**:
> - **RDS** = Postgres **managed** (backups, parches, replicación a cargo de AWS); vos solo usás la endpoint que te da.
> - **Por qué Postgres acá**: MLflow lo usa como **backend store** (metadata de runs: params, metrics, tags). Los artifacts pesados (`.joblib`, `.html`) van a **S3**, no a Postgres — así la DB no crece a TB.
> - **Por qué Secrets Manager y no env var**: el password queda **rotable** sin re-deploy y no aparece en plano en `terraform.tfstate` (solo el ARN).
> - **`skip_final_snapshot=true` + `deletion_protection=false`**: dejan destruir el lab sin fricción. **Cambialos a `false`/`true` antes de meter datos reales** (hardening, futuro).

```hcl
# infra/modules/mlflow/rds.tf
data "aws_region" "current" {}

resource "random_password" "rds" {
  length  = 32
  special = false # algunos chars rompen connection strings -> evitar
}

resource "aws_secretsmanager_secret" "rds" {
  name = "${var.project}-rds-password"
}

resource "aws_secretsmanager_secret_version" "rds" {
  secret_id     = aws_secretsmanager_secret.rds.id
  secret_string = random_password.rds.result
}

resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.project}-rds-subnets"
  subnet_ids = var.private_subnet_ids
}

resource "aws_db_instance" "mlflow" {
  identifier              = "${var.project}-mlflow"
  engine                  = "postgres"
  engine_version          = "15"
  instance_class          = var.rds_instance_class
  allocated_storage       = var.rds_allocated_storage_gb
  storage_type            = "gp3"
  storage_encrypted       = true
  db_name                 = "mlflow"
  username                = "mlflow"
  password                = random_password.rds.result
  db_subnet_group_name    = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids  = [var.sg_rds_id]
  publicly_accessible     = false
  skip_final_snapshot     = true # OK para prod single-AZ; ajustar en Parte 10.4
  apply_immediately       = true
  deletion_protection     = false # cambiar a true cuando hay datos productivos
  backup_retention_period = 7
  backup_window           = "06:00-07:00"
  maintenance_window      = "Mon:07:00-Mon:08:00"

  tags = { Name = "${var.project}-mlflow" }
}
```

#### 3.5.2.b — `alb.tf` — ALB (load balancer + target group + listener)

> 📂 **Pegar este bloque en**: `infra/modules/mlflow/alb.tf`

Un solo ALB sirve MLflow y reports — el listener default va a MLflow,
reports agrega una `listener_rule` desde #3.6. `idle_timeout=60` es
suficiente para ML training UI; subir si subis dashboards pesados.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_lb.main` | **EC2 > Load Balancers** | `Create Load Balancer > Application Load Balancer`. **Name**: `ml-training-alb`. **Scheme**: Internet-facing. **VPC**: la tuya. **Mappings**: ambas public subnets. **SGs**: `sg-alb`. **Listener**: HTTP :80 (HTTPS: hardening, futuro). |
> | `aws_lb_target_group.mlflow` | **EC2 > Target Groups** | `Create target group > IP addresses` (no Instances — Fargate usa IPs, no EC2 IDs). **Name**: `ml-training-tg-mlflow`. **Protocol/port**: HTTP/5000. **VPC**: la tuya. **Health check**: HTTP path `/health`, port 5000, healthy=2, unhealthy=5. |
> | `aws_lb_listener.http` | **EC2 > Load Balancers > [tu ALB] > Listeners** | `Add listener`. **Protocol/port**: HTTP/80. **Default action**: Forward to → target group `ml-training-tg-mlflow`. |
>
> **Conceptualmente**:
> - **ALB** = el portero público: recibe HTTP :80 desde Internet y lo enruta a un target group según reglas (path, host, query).
> - **Target Group** = lista de IPs con **health check** propio (`/health` cada 30s; 5 fallos → unhealthy). Por eso un scale-up tarda ~3 min: el ALB manda tráfico recién cuando el check pasa 2 veces.
> - **Listener** = qué hacer con el tráfico de un puerto. El default forwarea todo a MLflow; en #3.6 se agregan **listener rules** por path (`/reports/*` → otro TG). Así UN ALB sirve varios services.
> - **Por qué un solo ALB**: ~$16/mes base; con reglas por path sirve varios services. Multi-ALB solo si necesitás aislamiento total.

```hcl
# infra/modules/mlflow/alb.tf
resource "aws_lb" "main" {
  name               = "${var.project}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [var.sg_alb_id]
  subnets            = var.public_subnet_ids
  idle_timeout       = 60
}

# Default target group (MLflow). Reports agrega su propio TG en modulo
# reports y se asocia al listener via rule.
resource "aws_lb_target_group" "mlflow" {
  name        = "${var.project}-tg-mlflow"
  port        = 5000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  health_check {
    enabled             = true
    path                = "/health"
    port                = "5000"
    matcher             = "200"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }

  deregistration_delay = 30
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow.arn
  }
}
```

#### 3.5.2.c — `ecs.tf` (parte 1/2) — ECS cluster + Service Discovery

> 📂 **Pegar este bloque en**: `infra/modules/mlflow/ecs.tf`
> (es la **primera mitad** del archivo; la segunda mitad —log group +
> task def + service— viene en #3.5.2.e y se pega **a continuacion**
> en el mismo `ecs.tf`).

Cluster compartido por MLflow y reports (un cluster, dos services).
`containerInsights=disabled` ahorra ~$2/mes (logs custom-metricas a
CloudWatch); activar si necesitas tracing detallado. Service Discovery
publica `mlflow.ml-training.local:5000` internamente para que el trainer en Batch
no tenga que conocer el ALB DNS.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_ecs_cluster.main` | **ECS** | `ECS > Clusters > Create cluster`. **Name**: `ml-training-cluster`. **Infrastructure**: AWS Fargate (no EC2). **Monitoring > Container Insights**: disabled (Off). |
> | `aws_service_discovery_private_dns_namespace.main` | **Cloud Map** | `AWS Cloud Map > Namespaces > Create namespace`. **Name**: `ml-training.local`. **Type**: API calls and DNS queries in VPC. **VPC**: la tuya. |
> | `aws_service_discovery_service.mlflow` | **Cloud Map** | Dentro del namespace: `Create service`. **Name**: `mlflow`. **DNS records**: A record, TTL 10s. **Routing policy**: MULTIVALUE. |
>
> **Conceptualmente**:
> - **ECS Cluster** = entidad organizadora de services + tasks. Aunque sea Fargate (sin EC2), igual necesitás un cluster. Es gratis: pagás las tasks, no el cluster.
> - **Cloud Map / Service Discovery** = DNS interno automático de la VPC. El service `mlflow` (3.5.2.e) registra solo la IP de cada task, así otro container resuelve `mlflow.ml-training.local` sin importar los re-deploys.
> - **Por qué no el ALB DNS directo**: clientes EXTERNOS (browser, GHA) → ALB; clientes INTERNOS (trainer en Batch) → Cloud Map, conexión task-to-task más barata y rápida (sin roundtrip por el ALB).

```hcl
# infra/modules/mlflow/ecs.tf  (parte 1/2 — cluster + service discovery)
resource "aws_ecs_cluster" "main" {
  name = "${var.project}-cluster"
  setting {
    name  = "containerInsights"
    value = "disabled" # ahorra ~$2/mes; activar si necesitas tracing detallado
  }
}

# Service discovery namespace para que reports/batch resuelvan "mlflow.local"
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "${var.project}.local"
  description = "Service discovery interno"
  vpc         = var.vpc_id
}

resource "aws_service_discovery_service" "mlflow" {
  name = "mlflow"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }
  # AWS provider v6: `health_check_custom_config { failure_threshold = 1 }`
  # quedó deprecated — AWS lo enforza siempre a 1 implicitamente.
}
```

#### 3.5.2.d — `iam.tf` — IAM (execution role + task role)

> 📂 **Pegar este bloque en**: `infra/modules/mlflow/iam.tf`

ECS Fargate distingue dos roles:
- **exec role**: lo asume el agente Fargate ANTES del container —
  permite pullear de ECR, escribir logs, leer secrets.
- **task role**: lo asume el container — permite acceso a S3
  artifacts. Por que separados: si el container es comprometido, el
  atacante solo obtiene los perms del task role (no ECR/Secrets).

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_iam_role.mlflow_exec` + attachment + inline policy | **IAM** | `IAM > Roles > Create role`. **Trusted entity**: AWS service > **Elastic Container Service Task**. **Permissions**: `AmazonECSTaskExecutionRolePolicy` (managed) + inline policy custom para `secretsmanager:GetSecretValue` sobre el secret RDS. **Role name**: `ml-training-mlflow-exec`. |
> | `aws_iam_role.mlflow_task` + inline policy | **IAM** | Mismo wizard. **Role name**: `ml-training-mlflow-task`. **Permissions**: inline policy con `s3:GetObject/PutObject/DeleteObject/ListBucket` sobre el bucket `ml-training-artifacts-*`. |
>
> **Conceptualmente — exec role vs task role es defense-in-depth**:
> - **Exec role**: lo usa el agente Fargate ANTES de tu código — `docker pull` de ECR, `kms:Decrypt` del secret, logs a CloudWatch. El código de MLflow nunca ve esas creds.
> - **Task role**: lo usa **tu container** corriendo (creds temporales vía metadata endpoint). Un `boto3.client('s3')` adentro usa estas.
> - **El ataque que bloquea**: un RCE en MLflow **solo** obtiene el task role (S3 artifacts), **no** Secrets Manager ni ECR (esos son del exec role, fuera del container).
> - **Trust policy `ecs-tasks.amazonaws.com`**: solo ECS Fargate puede asumir el rol (no usuarios ni otros services).

```hcl
# infra/modules/mlflow/iam.tf
resource "aws_iam_role" "mlflow_exec" {
  name               = "${var.project}-mlflow-exec"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

resource "aws_iam_role_policy_attachment" "mlflow_exec" {
  role       = aws_iam_role.mlflow_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Permitir leer el secret del RDS password
resource "aws_iam_role_policy" "mlflow_exec_secret" {
  role = aws_iam_role.mlflow_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = aws_secretsmanager_secret.rds.arn
    }]
  })
}

resource "aws_iam_role" "mlflow_task" {
  name               = "${var.project}-mlflow-task"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

resource "aws_iam_role_policy" "mlflow_task_s3" {
  role = aws_iam_role.mlflow_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject", "s3:PutObject", "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        var.artifacts_bucket_arn,
        "${var.artifacts_bucket_arn}/*"
      ]
    }]
  })
}
```

> **Nota — assume policy centralizada**: los trust policies (`ecs-tasks`, `ec2`,
> `lambda`, `batch-service`, GHA-OIDC) viven como JSON estático en
> `infra/modules/_shared/`; cada módulo hace `file("${path.module}/../_shared/<archivo>.json")`.
> Antes se redeclaraban copy-paste en mlflow/reports/batch — ahora una sola fuente, sin drift.

#### 3.5.2.e — `ecs.tf` (parte 2/2) — Log group + Task Definition + Service

> 📂 **Pegar este bloque en**: `infra/modules/mlflow/ecs.tf`
> **a continuacion** de #3.5.2.c (mismo archivo, debajo de los
> recursos `aws_ecs_cluster` + `aws_service_discovery_*`).

El task-def encapsula la receta del container (imagen, comando,
healthcheck, secrets). El service mantiene N replicas corriendo
(`desiredCount=1`) y se conecta al ALB target group. `ignore_changes
= [desired_count]` permite al scheduler bajar a 0 sin que el siguiente
`terraform apply` lo vuelva a subir.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_cloudwatch_log_group.mlflow` | **CloudWatch** | `CloudWatch > Log groups > Create log group`. **Name**: `/ecs/ml-training/mlflow`. **Retention**: 30 days (var.log_retention_days). |
> | `aws_ecs_task_definition.mlflow` | **ECS** | `ECS > Task definitions > Create new task definition (JSON)`. **Family**: `ml-training-mlflow`. **Launch type**: Fargate. **OS/Arch**: Linux/x86_64. **CPU/Memory**: 2 vCPU / 4 GB. **Task role**: el `mlflow_task` que creaste. **Task exec role**: el `mlflow_exec`. **Container**: name `mlflow`, image `<ecr-url>:v3.12.0`, port 5000, command `mlflow server ...`, env vars + secret `RDS_PASSWORD` desde Secrets Manager, log config awslogs, healthcheck `python urllib /health`. |
> | `aws_ecs_service.mlflow` | **ECS** | `Cluster > Create service`. **Launch type**: Fargate. **Task definition**: el de arriba (latest revision). **Service name**: `mlflow`. **Desired tasks**: 1. **Networking > VPC**: la tuya, **subnets**: private, **SG**: `sg-mlflow`, **Public IP**: Disabled. **Load balancing**: enable, target group: `ml-training-tg-mlflow`. **Service discovery**: enable, namespace `ml-training.local`, service `mlflow`. |
>
> **Conceptualmente — la trinidad ECS**:
> - **Log group**: los stdout/stderr del container en CloudWatch (un stream por task, 30 días). Para debug post-mortem.
> - **Task definition**: la "receta" inmutable (imagen, recursos, network, env). Cada cambio crea una revisión (`:1`, `:2`…); rollback = apuntar el service a una anterior.
> - **Service**: el "manager" que mantiene N tasks vivas (self-healing) y hace **rolling deployment** (arranca la nueva, espera healthcheck, recién mata la vieja).
> - **`ignore_changes = [desired_count]`**: clave — el scheduler (#3.10) baja a `0`/sube a `1`; sin esto, el próximo `terraform apply` lo re-encendería deshaciendo el scheduler.

```hcl
# infra/modules/mlflow/ecs.tf  (parte 2/2 — log group + task def + service)
resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/ecs/${var.project}/mlflow"
  retention_in_days = var.log_retention_days
}

# Task definition
resource "aws_ecs_task_definition" "mlflow" {
  family                   = "${var.project}-mlflow"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "2048" # 2 vCPU
  memory                   = "4096" # 4 GB
  execution_role_arn       = aws_iam_role.mlflow_exec.arn
  task_role_arn            = aws_iam_role.mlflow_task.arn

  container_definitions = jsonencode([
    {
      name         = "mlflow"
      image        = var.mlflow_image
      essential    = true
      portMappings = [{ containerPort = 5000, protocol = "tcp" }]
      command = [
        "sh", "-c",
        join(" ", [
          "mlflow server",
          "--host 0.0.0.0 --port 5000",
          # Allowed-hosts wildcard: MLflow 3.x rechaza con 403 si el
          # Host: header no coincide. ALB DNS no se conoce en plan-time;
          # wildcard es la opcion mas simple. Hardening en #10.
          "--allowed-hosts '*'",
          # CORS: check SEPARADO de allowed-hosts. MLflow 3.5+ valida el
          # header `Origin` del navegador y su default es solo `localhost:*`,
          # asi que TODO POST del UI servido via ALB (runs/search, etc.) cae
          # en 403 "Cross-origin request blocked" -> la lista de runs sale
          # vacia. El DNS del ALB SI es referenciable en apply-time, asi que
          # lo pasamos explicito en vez de '*'. Ver #3.5.x.
          "--cors-allowed-origins http://${aws_lb.main.dns_name}",
          # Single `$` a propósito: Terraform solo escapa `$${`→`${`; un `$$`
          # suelto se pasa literal y `sh` lo interpreta como su PID ($$=PID),
          # mandando "<pid>RDS_PASSWORD" como password. Con un solo `$` el shell
          # expande la env var inyectada desde Secrets Manager. Ver #3.5.2.
          "--backend-store-uri postgresql://mlflow:$RDS_PASSWORD@${aws_db_instance.mlflow.address}:5432/mlflow",
          "--default-artifact-root s3://${var.artifacts_bucket}/artifacts",
          "--serve-artifacts"
        ])
      ]
      secrets = [{
        name      = "RDS_PASSWORD"
        valueFrom = aws_secretsmanager_secret.rds.arn
      }]
      environment = [
        { name = "AWS_DEFAULT_REGION", value = data.aws_region.current.region }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.mlflow.name
          awslogs-region        = data.aws_region.current.region
          awslogs-stream-prefix = "mlflow"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "python -c 'import urllib.request,sys; sys.exit(0 if urllib.request.urlopen(\"http://localhost:5000/health\",timeout=3).status==200 else 1)'"]
        interval    = 30
        timeout     = 5
        retries     = 5
        startPeriod = 60
      }
    }
  ])
}

resource "aws_ecs_service" "mlflow" {
  name            = "mlflow"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.mlflow.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  propagate_tags  = "SERVICE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.sg_mlflow_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.mlflow.arn
    container_name   = "mlflow"
    container_port   = 5000
  }

  service_registries {
    registry_arn = aws_service_discovery_service.mlflow.arn
  }

  # Ignore desired_count para que el scheduler lo pueda manejar sin drift
  lifecycle {
    ignore_changes = [desired_count]
  }

  depends_on = [aws_lb_listener.http]
}
```

> **Checkpoint despues de pegar 3.5.2.a-e**: ejecutar
> `terraform fmt infra/modules/mlflow/` (sobre el **directorio**, no
> un archivo — asi formatea los 4 archivos: `rds.tf`, `alb.tf`,
> `ecs.tf`, `iam.tf`). Si reformatea sin errores, los bloques quedaron
> bien pegados. Despues `terraform -chdir=infra/modules/mlflow init
> -backend=false && terraform -chdir=infra/modules/mlflow validate`
> debe terminar con `Success! The configuration is valid.`

### 3.5.3 `modules/mlflow/outputs.tf`

```hcl
output "tracking_uri" { value = "http://${aws_lb.main.dns_name}" }
output "alb_dns" { value = aws_lb.main.dns_name }
output "alb_arn_suffix" { value = aws_lb.main.arn_suffix } # para CloudWatch dimensions
output "alb_listener_arn" { value = aws_lb_listener.http.arn }
output "cluster_id" { value = aws_ecs_cluster.main.id }
output "cluster_name" { value = aws_ecs_cluster.main.name }
output "service_name" { value = aws_ecs_service.mlflow.name }
output "rds_instance_id" { value = aws_db_instance.mlflow.id }

# --- Wiring para los modulos api / ui (Capa 4.5) ---
output "service_discovery_namespace_id" {
  description = "ID del namespace privado <project>.local para registrar la API."
  value       = aws_service_discovery_private_dns_namespace.main.id
}
output "rds_address" {
  description = "Host del RDS (la API monta su DATABASE_URL hacia la base forecasts)."
  value       = aws_db_instance.mlflow.address
}
output "rds_password_secret_arn" {
  description = "ARN del secret con el password del RDS (lo inyecta la API)."
  value       = aws_secretsmanager_secret.rds.arn
}
```

> **En consola AWS veras**:
> - RDS → Databases → `ml-training-mlflow` (engine=postgres15.4,
>   db.t4g.small, 20GB gp3, Single-AZ, Status=Available). **Es la unica
>   pieza con storage persistente del Model Registry** — hostea la base de
>   MLflow Y la base `forecasts` de la API. Apagarla con `task sleep` no
>   borra data, solo deja de cobrar compute.
> - EC2 → Load Balancers → `ml-training-alb` (internet-facing). DNS
>   `ml-training-alb-XXXX.us-east-1.elb.amazonaws.com` — este es el
>   `MLFLOW_ALB_DNS` que va a las GitHub vars.
> - EC2 → Target Groups → `ml-training-tg-mlflow` (health=200 en `/health`).
> - ECS → Clusters → `ml-training-cluster` → Services → `mlflow`
>   (desiredCount=1, runningCount=1, healthy).
> - ECS → Task Definitions → `ml-training-mlflow:N` (imagen custom de
>   ECR, env vars MLFLOW_BACKEND_STORE_URI + ARTIFACT_STORE).
> - Cloud Map (Service Discovery) → Namespaces → `<project>.local`
>   (interno, para que reports/batch resuelvan `mlflow.ml-training.local:5000`).
> - Secrets Manager → `ml-training-rds-password` (con KMS aws/secretsmanager).
> - CloudWatch → Log groups → `/ecs/ml-training/mlflow` (logs Fargate).

### 3.5.4 Apendear `module "mlflow"` en `infra/envs/prod/main.tf`

Pegar AL FINAL de `infra/envs/prod/main.tf` (despues de
`module "storage"` de #3.4.4):

```hcl
# -------------------------------------------------------------------------
# Capa 3: MLflow (RDS + ECS Fargate + ALB)
# -------------------------------------------------------------------------
module "mlflow" {
  source = "../../modules/mlflow"

  project              = var.project
  vpc_id               = module.network.vpc_id
  public_subnet_ids    = module.network.public_subnet_ids
  private_subnet_ids   = module.network.private_subnet_ids
  sg_alb_id            = module.network.sg_alb_id
  sg_mlflow_id         = module.network.sg_mlflow_id
  sg_rds_id            = module.network.sg_rds_id
  rds_instance_class   = var.rds_instance_class
  mlflow_image         = "${module.storage.ecr_mlflow_url}:${var.mlflow_image_tag}"
  artifacts_bucket     = module.storage.artifacts_bucket
  artifacts_bucket_arn = module.storage.artifacts_bucket_arn
  log_retention_days   = var.log_retention_days
}
```

> **Checkpoint**: este es el **primer modulo con dependencias** —
> consume outputs de `module.network` (VPC, subnets, SGs) y
> `module.storage` (ECR url, bucket de artifacts). Si `terraform
> validate` falla con "Unsupported attribute" o "Reference to
> undeclared module", revisa que pegaste #3.3.4 y #3.4.4 antes que
> esto.

> **Gotcha #3.5**: el módulo está SPLIT en 5 archivos (`alb.tf` / `ecs.tf` / `iam.tf` / `rds.tf` / `main.tf`). Pegarlos todos en `main.tf` funciona pero diverge del repo (ver #3.5.2).

---

## 3.6 `modules/reports/` — Fargate nginx sirviendo S3

Sirve `s3://artifacts/{reports,artifacts}/` como sitio estatico bajo
`http://<ALB>/reports/*` y `http://<ALB>/artifacts/*`. Usa el mismo ALB
listener via `path-pattern` rules.

Mecanismo: container = nginx + sidecar de `aws s3 sync` que copia el
bucket a `/usr/share/nginx/html/` cada 60s. Costo: $0.50/mes Fargate +
trafico S3 GET despreciable.

### 3.6.1 `modules/reports/variables.tf`

```hcl
variable "project" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "sg_mlflow_id" { type = string } # SG con ingress :80 desde sg-alb; reports lo reusa
variable "ecs_cluster_id" { type = string }
variable "alb_listener_arn" { type = string }
variable "artifacts_bucket" { type = string }
variable "artifacts_bucket_arn" { type = string }
variable "reports_image" { type = string }
variable "log_retention_days" { type = number }
```

> **Por qué reusa `sg_mlflow_id` y no `sg_alb_id`**: el task de reports necesita
> ingress :80 desde el ALB, y `sg_mlflow` ya abre :80 desde `sg_alb`. Pasarle
> `sg_alb_id` directo haría que aceptara :80 desde `0.0.0.0/0` (la regla de
> `sg_alb`) — peligroso y no coincide con lo que el ALB realmente envía.

### 3.6.2 `modules/reports/main.tf`

> **Equivalente en AWS Console — vista general del modulo reports**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_lb_target_group.reports` | **EC2 > Target Groups** | `Create target group > IP addresses`. **Name**: `ml-training-tg-reports`. HTTP/80. Health check path: `/healthz`. |
> | `aws_lb_listener_rule.reports_path` | **EC2 > Load Balancers > [ALB] > Listeners > HTTP:80 > Manage rules** | `Insert rule`. **Priority**: 100 (menor = mas prioritario). **IF Path is `/reports/*` OR `/reports` OR `/artifacts/*` OR `/artifacts`** → **THEN Forward to** `ml-training-tg-reports`. Los 4 paths son necesarios: el `/*` matchea `/reports/foo.html` pero NO matchea el listado raw `/reports` (sin slash final); por eso se incluyen ambas variantes. Default action (forward a MLflow TG) queda como fallback. |
> | `aws_iam_role.reports_exec` + `reports_task` | **IAM** | Mismo wizard que MLflow exec/task roles, pero el task role tiene `s3:GetObject + ListBucket` solo (no PUT — reports es read-only sobre artifacts). |
> | `aws_cloudwatch_log_group.reports` | **CloudWatch** | `Create log group`. Name: `/ecs/ml-training/reports`. |
> | `aws_ecs_task_definition.reports` | **ECS > Task definitions** | Mismo wizard. CPU/Mem: 0.5 vCPU / 1 GB (es solo nginx). Image: `<ecr-url>/ml-training-reports:latest`. Env: `S3_BUCKET=<artifacts-bucket>`. |
> | `aws_ecs_service.reports` | **ECS > Cluster > Services** | Mismo wizard. **Cluster**: el `ml-training-cluster` ya existente (NO crear otro). **Service name**: `reports`. **Target group**: el de reports. |
>
> **Conceptualmente — el patrón "Fargate sidecar de S3"**:
> - Reports es nginx sirviendo HTML estático que viene de S3. Tres formas: (1) **CloudFront + S3** (más barato, pero dashboards públicos u OAC); (2) **API GW + Lambda + S3** (serverless, $$$$ por request); (3) **Fargate nginx + sidecar `aws s3 sync`** (este — $4-5/mes, reusa el ALB y mantiene los reports privados tras el SG).
> - **El truco**: reusa el ALB (ahorra $16/mes), el cluster ECS y el SG `sg_mlflow`.
> - **Listener rules ORDENADAS por priority**: el ALB evalúa de menor a mayor; `priority=100` corre antes del default action (siempre último).

```hcl
data "aws_region" "current" {}

# Target group
resource "aws_lb_target_group" "reports" {
  name        = "${var.project}-tg-reports"
  port        = 80
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  health_check {
    path                = "/healthz"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 5
    matcher             = "200"
  }
}

# Listener rules: /reports/* y /artifacts/* -> reports TG
resource "aws_lb_listener_rule" "reports_path" {
  listener_arn = var.alb_listener_arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.reports.arn
  }
  condition {
    path_pattern { values = ["/reports/*", "/reports", "/artifacts/*", "/artifacts"] }
  }
}

# IAM (assume policy compartida en infra/modules/_shared/)
resource "aws_iam_role" "reports_exec" {
  name               = "${var.project}-reports-exec"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

resource "aws_iam_role_policy_attachment" "reports_exec" {
  role       = aws_iam_role.reports_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "reports_task" {
  name               = "${var.project}-reports-task"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

resource "aws_iam_role_policy" "reports_task_s3" {
  role = aws_iam_role.reports_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [var.artifacts_bucket_arn, "${var.artifacts_bucket_arn}/*"]
    }]
  })
}

resource "aws_cloudwatch_log_group" "reports" {
  name              = "/ecs/${var.project}/reports"
  retention_in_days = var.log_retention_days
}

resource "aws_ecs_task_definition" "reports" {
  family                   = "${var.project}-reports"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"  # 0.5 vCPU
  memory                   = "1024" # 1 GB
  execution_role_arn       = aws_iam_role.reports_exec.arn
  task_role_arn            = aws_iam_role.reports_task.arn

  container_definitions = jsonencode([
    {
      name         = "reports"
      image        = var.reports_image
      essential    = true
      portMappings = [{ containerPort = 80, protocol = "tcp" }]
      environment = [
        { name = "S3_BUCKET", value = var.artifacts_bucket },
        { name = "AWS_DEFAULT_REGION", value = data.aws_region.current.region }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.reports.name
          awslogs-region        = data.aws_region.current.region
          awslogs-stream-prefix = "reports"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "reports" {
  name            = "reports"
  cluster         = var.ecs_cluster_id
  task_definition = aws_ecs_task_definition.reports.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.sg_mlflow_id] # mismo SG que mlflow: ingress :80 desde sg-alb
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.reports.arn
    container_name   = "reports"
    container_port   = 80
  }

  lifecycle {
    ignore_changes = [desired_count]
  }
}
```

### 3.6.3 `modules/reports/outputs.tf`

```hcl
output "service_name" { value = aws_ecs_service.reports.name }
```

> **En consola AWS veras**:
> - EC2 → Load Balancers → `ml-training-alb` → Listeners → HTTP:80 →
>   Rules: 2 nuevas con `path-pattern=/reports/*` y `/artifacts/*` que
>   ruteo al target group `ml-training-tg-reports`. Default (/) sigue
>   yendo al de MLflow.
> - EC2 → Target Groups → `ml-training-tg-reports` (health=200 en
>   `/healthz`).
> - ECS → Cluster `ml-training-cluster` → Services → `reports` (segundo
>   service, mismo cluster que MLflow). Task definition con imagen
>   custom de ECR `ml-training-reports`.
> - CloudWatch → Log groups → `/ecs/ml-training/reports`.

### 3.6.4 `docker/reports/Dockerfile`

Imagen custom: nginx + `aws s3 sync` cada 60s en background.

> **NO usar `nginx:*-alpine` + `apk add aws-cli`** aqui: en esas imagenes el
> `pyexpat`/`expat` esta desalineado y tumba CUALQUIER invocacion de `aws` (el
> `aws s3 sync` del entrypoint falla en silencio y `/reports` queda vacio). El
> porque esta en el comentario del propio Dockerfile. (El compose local en
> #4.5.2 SI usa `nginx:alpine` porque ahi se montan dirs del host y NO hay
> aws-cli.)

```dockerfile
# Base Debian (glibc), NO alpine: el paquete `aws-cli` de Alpine venia con un
# expat roto en la imagen (pyexpat/minidom: "XML_SetAllocTrackerActivationThreshold:
# symbol not found"), lo que tumbaba CUALQUIER invocacion de `aws` con un
# traceback de Python -> el `aws s3 sync` del entrypoint nunca corria y
# /reports + /artifacts quedaban vacios aunque S3 tuviera los archivos.
# AWS CLI v2 oficial es glibc, self-contained y soportado: no depende del
# expat/prompt_toolkit del sistema.
FROM nginx:1.27

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         curl unzip ca-certificates dumb-init \
    && curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip \
    && apt-get purge -y unzip \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# config nginx que sirve /usr/share/nginx/html con autoindex
COPY docker/reports/nginx.conf /etc/nginx/conf.d/default.conf
COPY docker/reports/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/entrypoint.sh"]
```

### 3.6.5 `docker/reports/nginx.conf`

```nginx
server {
  listen 80 default_server;
  server_name _;

  # Health check para ALB target group
  location = /healthz {
    access_log off;
    return 200 "ok\n";
    add_header Content-Type text/plain;
  }

  # /reports/* -> /usr/share/nginx/html/reports/*
  # /artifacts/* -> /usr/share/nginx/html/artifacts/*
  location / {
    root /usr/share/nginx/html;
    autoindex on;
    autoindex_exact_size off;
    autoindex_localtime on;
    add_header Cache-Control "no-store";
  }
}
```

### 3.6.6 `docker/reports/entrypoint.sh`

```bash
#!/bin/bash
set -e

: "${S3_BUCKET:?S3_BUCKET requerido}"
: "${AWS_DEFAULT_REGION:?AWS_DEFAULT_REGION requerido}"

mkdir -p /usr/share/nginx/html/reports /usr/share/nginx/html/artifacts

# Sync inicial (bloqueante: arrancamos nginx con data ya cargada)
aws s3 sync "s3://${S3_BUCKET}/reports/"   /usr/share/nginx/html/reports/   --no-progress || true
aws s3 sync "s3://${S3_BUCKET}/artifacts/" /usr/share/nginx/html/artifacts/ --no-progress || true

# Sync loop en background (cada 60s)
(
  while true; do
    sleep 60
    aws s3 sync "s3://${S3_BUCKET}/reports/"   /usr/share/nginx/html/reports/   --delete --no-progress >/dev/null 2>&1 || true
    aws s3 sync "s3://${S3_BUCKET}/artifacts/" /usr/share/nginx/html/artifacts/ --delete --no-progress >/dev/null 2>&1 || true
  done
) &

# Foreground: nginx
exec nginx -g 'daemon off;'
```

### 3.6.7 Apendear `module "reports"` en `infra/envs/prod/main.tf`

Pegar AL FINAL de `infra/envs/prod/main.tf` (despues de
`module "mlflow"` de #3.5.4):

```hcl
# -------------------------------------------------------------------------
# Capa 4: Reports (Fargate nginx, mismo cluster + ALB que MLflow)
# -------------------------------------------------------------------------
module "reports" {
  source = "../../modules/reports"

  project              = var.project
  vpc_id               = module.network.vpc_id
  private_subnet_ids   = module.network.private_subnet_ids
  sg_mlflow_id         = module.network.sg_mlflow_id
  ecs_cluster_id       = module.mlflow.cluster_id
  alb_listener_arn     = module.mlflow.alb_listener_arn
  artifacts_bucket     = module.storage.artifacts_bucket
  artifacts_bucket_arn = module.storage.artifacts_bucket_arn
  reports_image        = "${module.storage.ecr_reports_url}:${var.reports_image_tag}"
  log_retention_days   = var.log_retention_days
}
```

> **Checkpoint**: `reports` reusa el `cluster_id` y `alb_listener_arn`
> de `module.mlflow` — por eso #3.5.4 **tiene que estar antes**. Si
> intentas validar sin haber pegado mlflow, vas a ver "Reference to
> undeclared module module.mlflow".

> **Gotcha #3.6**: la imagen `reports` requiere `docker/reports/{Dockerfile,nginx.conf,entrypoint.sh}` (#3.6.4-3.6.6). Si no los creaste, `task ecr:build` fallará luego con "Cannot locate specified Dockerfile" (`terraform validate` pasa porque no toca Docker).

---

## 3.7 `modules/batch/` — Compute envs + queues + job-def + IAM

Pieza critica donde se respeta el contrato del trainer: el container
recibe via CMD `--varieties X --tuning Y` (matchea
`src/orchestration/cli.py:parse_args`) y las env vars S3 que
`main.py:_hydrate_data_from_s3` lee.

### 3.7.1 `modules/batch/variables.tf`

```hcl
variable "project" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "sg_batch_id" { type = string }
variable "ecr_trainer_url" { type = string }
variable "trainer_image_tag" { type = string }

variable "spot_max_vcpus" { type = number }
variable "ondemand_max_vcpus" { type = number }
variable "spot_bid_percentage" {
  type    = number
  default = 70
}
variable "instance_type" { type = string }

variable "tracking_uri" { type = string }
variable "artifacts_bucket" { type = string }
variable "artifacts_bucket_arn" { type = string }
variable "data_bucket_arn" { type = string }

variable "job_attempt_seconds" {
  type    = number
  default = 28800 # 8h hard ceiling (incluye prod_xl)
}

variable "log_retention_days" { type = number }
```

### 3.7.2 `modules/batch/iam.tf`

> 📂 **Pegar este bloque en**: `infra/modules/batch/iam.tf`
> (el `main.tf` del mismo modulo lo cubrimos en #3.7.3 a continuacion).
>
> **Equivalente en AWS Console — los 4 roles IAM del modulo batch**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_iam_role.batch_instance` + `aws_iam_instance_profile.batch` | **IAM** | `IAM > Roles > Create role`. **Trusted entity**: AWS service > **EC2**. **Permissions**: `AmazonEC2ContainerServiceforEC2Role` (managed). **Name**: `ml-training-batch-instance`. Despues `IAM > Instance Profiles` (deprecated en Console moderna — el wizard de role crea el instance profile automaticamente). |
> | `aws_iam_role.job` + 2 inline policies (S3 + CloudWatch) | **IAM** | `Create role`. **Trusted entity**: AWS service > **Elastic Container Service Task**. **Permissions**: 2 inline policies — (a) S3: Get/Put/List sobre buckets data y artifacts; (b) CloudWatch: PutMetricData. **Name**: `ml-training-job-role`. |
> | `aws_iam_role.exec` | **IAM** | Mismo wizard. **Permissions**: `AmazonECSTaskExecutionRolePolicy` (managed). **Name**: `ml-training-job-exec`. |
> | `aws_iam_role.batch_service` | **IAM** | `Create role`. **Trusted entity**: AWS service > **AWS Batch**. **Permissions**: `AWSBatchServiceRole` (managed). **Name**: `ml-training-batch-service`. |
>
> **Conceptualmente — por qué CUATRO roles distintos**:
> - **instance**: lo asume **la EC2** que Batch arranca; le deja reportar al cluster ECS subyacente.
> - **job (task role)**: lo asume **tu container** (el trainer). Permisos S3 read/write + CloudWatch PutMetric — lo único que tu código Python ve.
> - **exec**: lo asume **el agente ECS** ANTES del container; pull de ECR + logs.
> - **batch_service**: lo asume **AWS Batch** para gestionar los compute environments (crear/destruir/escalar EC2s).
> - **Por qué tanta separación**: **least privilege**. Comprometer el trainer (job role) no da destruir EC2s (batch_service), modificar el cluster (instance) ni leer secrets.

```hcl
# infra/modules/batch/iam.tf
# Trust policies viven como JSON estatico en infra/modules/_shared/
# (assume-ec2.json, assume-ecs-tasks.json, assume-batch-service.json).

# Role asumido por la EC2 que lanza Batch (instance profile)
resource "aws_iam_role" "batch_instance" {
  name               = "${var.project}-batch-instance"
  assume_role_policy = file("${path.module}/../_shared/assume-ec2.json")
}

resource "aws_iam_role_policy_attachment" "batch_instance" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch" {
  name = "${var.project}-batch-instance"
  role = aws_iam_role.batch_instance.name
}

# Role asumido por el container (task) durante el job
resource "aws_iam_role" "job" {
  name               = "${var.project}-job-role"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

# S3: el trainer necesita:
#  - GetObject en s3://data/ (hydrate del Excel acumulado)
#  - PutObject en s3://artifacts/{artifacts,reports}/ (sync de outputs)
#  - PutObject en s3://artifacts/artifacts/ (MLflow log_artifact)
resource "aws_iam_role_policy" "job_s3" {
  role = aws_iam_role.job.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [var.data_bucket_arn, "${var.data_bucket_arn}/*"]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"
        ]
        Resource = [var.artifacts_bucket_arn, "${var.artifacts_bucket_arn}/*"]
      }
    ]
  })
}

# CloudWatch: para emitir custom metric MAPE desde el trainer (Parte 5)
resource "aws_iam_role_policy" "job_cloudwatch" {
  role = aws_iam_role.job.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["cloudwatch:PutMetricData"]
      Resource = "*"
    }]
  })
}

# Execution role (pull image, write logs) — usado por Batch para arrancar
resource "aws_iam_role" "exec" {
  name               = "${var.project}-job-exec"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

resource "aws_iam_role_policy_attachment" "exec" {
  role       = aws_iam_role.exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Service role de Batch (gestion de CE).
# Antes era inline `jsonencode({...})`; ahora consume el JSON shared.
resource "aws_iam_role" "batch_service" {
  name               = "${var.project}-batch-service"
  assume_role_policy = file("${path.module}/../_shared/assume-batch-service.json")
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}
```

### 3.7.3 `modules/batch/main.tf`

> 📂 **Pegar los 4 sub-bloques (a/b/c/d) en**: `infra/modules/batch/main.tf`
> (concatenados, en orden — cada sub-bloque trae el comentario
> `# infra/modules/batch/main.tf (parte N/4)` como recordatorio).

#### 3.7.3.a — Log group (parte 1/4 de `main.tf`)

Logs de TODOS los Batch jobs en un solo group. Retencion configurable
desde `envs/prod/terraform.tfvars` (default 14 dias).

```hcl
# infra/modules/batch/main.tf  (parte 1/4 — data + log group)
data "aws_region" "current" {}

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/${var.project}"
  retention_in_days = var.log_retention_days
}
```

#### 3.7.3.b — Compute Environments (Spot + OnDemand) (parte 2/4 de `main.tf`)

2 CEs: Spot (default, ~70% mas barato, puede interrumpir) y OnDemand
(reservado para `prod_xl` que tarda 5-6h y no tolera Spot kills).
`min_vcpus=0` permite que Batch escale a 0 cuando no hay jobs (ahorro
total fuera de horario). `ignore_changes = desired_vcpus` evita
drift cuando Batch scaling lo cambia entre apply y apply.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_batch_compute_environment.spot` | **AWS Batch** | `Batch > Compute environments > Create`. **Type**: Managed. **Name**: `ml-training-ce-spot`. **Provisioning model**: **Spot** (importante). **Bid percentage**: 70 (paga max 70% del precio On-Demand, default de `var.spot_bid_percentage`). **Allocation strategy**: SPOT_CAPACITY_OPTIMIZED. **Min/Max vCPUs**: 0 / 16. **Instance types**: `c6i.2xlarge`. **VPC**: la tuya, **Subnets**: las private, **SG**: `sg-batch`. **Instance role**: `ml-training-batch-instance`. **Service role**: `ml-training-batch-service`. |
> | `aws_batch_compute_environment.ondemand` | **AWS Batch** | Mismo wizard pero **Provisioning model**: **On-Demand** (EC2). **Allocation strategy**: BEST_FIT_PROGRESSIVE. Resto igual. |
>
> **Conceptualmente — Compute Environment = "pool autoscaleable de EC2s"**:
> - Al submitear un job, Batch mira la queue → su CE → si no hay capacidad **arranca una EC2** del tipo configurado, y la **apaga** al terminar (escala a `min_vcpus=0`). Pagás EC2 solo durante el job (~$0.03/h c6i.2xlarge Spot).
> - **Spot vs On-Demand**: Spot = EC2 sobrantes a 60-90% off, pero AWS puede **interrumpir** con 2 min de aviso (Batch re-encola según `retry_strategy`). On-Demand = precio full, sin interrupción. Jobs cortos (<30 min) → Spot; largos (>4h) o HPO costoso → On-Demand. El dispatcher (#3.9) elige según `tuning`.
> - **`min_vcpus=0`**: escala a cero sin jobs (sin esto, EC2s idle cuestan $$$).
> - **`ignore_changes = desired_vcpus`**: Batch lo ajusta según carga; Terraform no debe "arreglarlo" en cada apply.

```hcl
# infra/modules/batch/main.tf  (parte 2/4 — compute environments)
resource "aws_batch_compute_environment" "spot" {
  # AWS provider v6+ usa `name`. El atributo `compute_environment_name`
  # fue deprecado en v5 y eliminado en v6 -> con `aws ~> 6.0` lockeado
  # en #3.2.1, `terraform validate` falla si se usa el nombre viejo.
  name         = "${var.project}-ce-spot"
  service_role = aws_iam_role.batch_service.arn
  type         = "MANAGED"
  state        = "ENABLED"

  compute_resources {
    type                = "SPOT"
    bid_percentage      = var.spot_bid_percentage
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"
    min_vcpus           = 0
    max_vcpus           = var.spot_max_vcpus
    desired_vcpus       = 0
    instance_type       = [var.instance_type]
    subnets             = var.private_subnet_ids
    security_group_ids  = [var.sg_batch_id]
    instance_role       = aws_iam_instance_profile.batch.arn
    tags                = { Name = "${var.project}-batch-spot" }
  }

  lifecycle {
    create_before_destroy = true
    ignore_changes        = [compute_resources[0].desired_vcpus]
  }
}

resource "aws_batch_compute_environment" "ondemand" {
  name         = "${var.project}-ce-ondemand" # ver nota en bloque "spot" sobre v6
  service_role = aws_iam_role.batch_service.arn
  type         = "MANAGED"
  state        = "ENABLED"

  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    min_vcpus           = 0
    max_vcpus           = var.ondemand_max_vcpus
    desired_vcpus       = 0
    instance_type       = [var.instance_type]
    subnets             = var.private_subnet_ids
    security_group_ids  = [var.sg_batch_id]
    instance_role       = aws_iam_instance_profile.batch.arn
    tags                = { Name = "${var.project}-batch-od" }
  }

  lifecycle {
    create_before_destroy = true
    ignore_changes        = [compute_resources[0].desired_vcpus]
  }
}
```

#### 3.7.3.c — Job queues (1 por CE) (parte 3/4 de `main.tf`)

Una queue por CE. El Lambda dispatcher (#3.9.5) elige queue por
`tuning`: `prod_xl → ondemand`, resto → spot. Priority=1 en ambas
(no hay queueing entre ellas, son disjuntas).

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_batch_job_queue.spot` | **AWS Batch** | `Batch > Job queues > Create`. **Name**: `ml-training-job-queue-spot`. **State**: Enabled. **Priority**: 1. **Connected compute environments > Add**: `ml-training-ce-spot`, order 1. |
> | `aws_batch_job_queue.ondemand` | **AWS Batch** | Mismo wizard, name `-ondemand`, conectado a `ml-training-ce-ondemand`. |
>
> **Conceptualmente — Queue es donde "depositás" jobs**:
> - El dispatcher hace `aws batch submit-job --job-queue …`; el job pasa `SUBMITTED → PENDING → RUNNABLE` y, cuando hay capacidad en el CE, `STARTING → RUNNING → SUCCEEDED/FAILED`.
> - **Por qué 2 queues y no una con 2 CEs**: una queue multi-CE haría spillover (si CE-A se llena, usa CE-B). Con 2 queues separadas el dispatcher elige explícito — `prod_xl` SIEMPRE OD, resto SIEMPRE Spot, sin riesgo de spillover.

```hcl
# infra/modules/batch/main.tf  (parte 3/4 — job queues)
resource "aws_batch_job_queue" "spot" {
  name     = "${var.project}-job-queue-spot"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.spot.arn
  }
}

resource "aws_batch_job_queue" "ondemand" {
  name     = "${var.project}-job-queue-ondemand"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.ondemand.arn
  }
}
```

#### 3.7.3.d — Job definition (contrato con el trainer) (parte 4/4 de `main.tf`)

El campo `command = ["--varieties","POP","--tuning","smoke"]` es default
— el dispatcher (#3.9.5) lo sobreescribe por job. `retry_strategy`
solo reintenta cuando AWS Spot mata el host (no en error del trainer).
`timeout = job_attempt_seconds` (default 28800 = 8h) corta jobs colgados.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_batch_job_definition.trainer` | **AWS Batch** | `Batch > Job definitions > Create`. **Type**: Single-node. **Platform type**: EC2 (no Fargate — necesitamos c6i.2xlarge). **Name**: `ml-training-trainer`. **Container properties**: **Image**: `<ecr-url>/ml-training:v0.1.0`, **vCPUs**: 8, **Memory**: 14000 MiB, **Command**: `["--varieties","POP","--tuning","smoke"]` (default; el dispatcher lo sobreescribe). **Job role**: `ml-training-job-role`. **Execution role**: `ml-training-job-exec`. **Network**: sin `networkConfiguration` (assignPublicIp es solo Fargate; en EC2 la IP publica la define la subnet/compute environment). **Environment variables**: `MLFLOW_TRACKING_URI`, `S3_ARTIFACTS_BUCKET`, etc. **Log configuration**: awslogs, group `/aws/batch/ml-training`. **Timeout**: 28800 sec (8h). **Retry strategy**: attempts=2 con `evaluate_on_exit` para reintentar solo en interrupciones Spot. |
>
> **Conceptualmente — Job Definition = "receta inmutable de cómo correr el trainer"**:
> - Plantilla: al submitear, Batch lanza un container basado en ella; podés sobreescribir campos por submit (el dispatcher cambia `command` para variar `--varieties`).
> - **Cada cambio crea una REVISION** (`:1`, `:2`…); rollback = apuntar a una anterior.
> - **`retry_strategy` con `evaluate_on_exit`**: sin reglas, AWS reintenta cualquier fallo (incluido un bug) y gastás $$$. Con reglas: `Host EC2*` (Spot mató la EC2) → RETRY; `*` (cualquier otro, incl. exit code del trainer) → EXIT.
> - **`timeout = 28800`** (8h): mata jobs colgados para no pagar EC2 de más.

```hcl
# infra/modules/batch/main.tf  (parte 4/4 — job definition)
resource "aws_batch_job_definition" "trainer" {
  name = "${var.project}-trainer"
  type = "container"

  retry_strategy {
    attempts = 2
    # Auto-retry solo cuando Spot interrumpe el host (preserva exit codes
    # del trainer; un error real no se reintenta)
    evaluate_on_exit {
      action           = "RETRY"
      on_status_reason = "Host EC2*"
    }
    evaluate_on_exit {
      action    = "EXIT"
      on_reason = "*"
    }
  }

  timeout {
    attempt_duration_seconds = var.job_attempt_seconds
  }

  container_properties = jsonencode({
    image            = "${var.ecr_trainer_url}:${var.trainer_image_tag}"
    vcpus            = 8     # c6i.2xlarge tiene 8 vCPU
    memory           = 14000 # de los 16 GB, dejamos ~2 GB para kernel + Batch agent
    jobRoleArn       = aws_iam_role.job.arn
    executionRoleArn = aws_iam_role.exec.arn
    # networkConfiguration (assignPublicIp) es solo Fargate; en EC2 la IP
    # publica se define en la subnet/compute environment, no aqui.
    # Sobreescrito por Lambda dispatcher (Sec 3.9.5) en cada submit.
    command = ["--varieties", "POP", "--tuning", "smoke"]
    environment = [
      { name = "MLFLOW_TRACKING_URI", value = var.tracking_uri },
      { name = "S3_ARTIFACTS_BUCKET", value = var.artifacts_bucket },
      { name = "S3_ARTIFACTS_PREFIX", value = "artifacts" },
      { name = "S3_REPORTS_PREFIX", value = "reports" },
      # S3_DATA_BUCKET / S3_DATA_KEY se inyectan por job (varia por submit)
      { name = "AWS_DEFAULT_REGION", value = data.aws_region.current.region },
      { name = "PYTHONUNBUFFERED", value = "1" }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.batch.name
        awslogs-region        = data.aws_region.current.region
        awslogs-stream-prefix = "trainer"
      }
    }
  })

  propagate_tags = true
  tags = {
    Project = var.project
  }
}
```

> **Checkpoint despues de 3.7.3.a-d**: `terraform fmt
> infra/modules/batch/` (sobre el **directorio** — formatea `iam.tf` +
> `main.tf` + `outputs.tf` + `variables.tf`).

### 3.7.4 `modules/batch/outputs.tf`

```hcl
output "job_queue_spot" { value = aws_batch_job_queue.spot.name }
output "job_queue_spot_arn" { value = aws_batch_job_queue.spot.arn }
output "job_queue_ondemand" { value = aws_batch_job_queue.ondemand.name }
output "job_queue_ondemand_arn" { value = aws_batch_job_queue.ondemand.arn }
output "job_definition_name" { value = aws_batch_job_definition.trainer.name }
output "log_group_name" { value = aws_cloudwatch_log_group.batch.name }
```

> **En consola AWS veras**:
> - Batch → Compute environments → 2: `ml-training-ce-spot` (instance
>   type `c6i.2xlarge` (var.instance_type), Spot allocationStrategy
>   SPOT_CAPACITY_OPTIMIZED) y `ml-training-ce-ondemand` (mismo
>   instance type, EC2). Ambas con `state=ENABLED, status=VALID`.
> - Batch → Job queues → 2: `ml-training-job-queue-spot` (priority=1,
>   conecta a ce-spot) y `-ondemand` (priority=1). Ambas `VALID`.
> - Batch → Job definitions → `ml-training-trainer` (revision N, type
>   container, imagen del ECR `ml-training:latest`). Es lo que el Lambda
>   dispatcher (#3.9) invoca con SubmitJob.
> - IAM → Roles → `ml-training-batch-instance` (EC2 lanzadora),
>   `ml-training-job-role` (lo asume el container del trainer:
>   S3 r/w + CloudWatch PutMetricData), `ml-training-job-exec`
>   (ECR pull + logs write), `ml-training-batch-service`
>   (gestion del CE).
> - CloudWatch → Log groups → `/aws/batch/ml-training` con
>   `retention=14 days`.

### 3.7.5 Apendear `module "batch"` en `infra/envs/prod/main.tf`

Pegar AL FINAL de `infra/envs/prod/main.tf` (despues de
`module "reports"` de #3.6.7):

```hcl
# -------------------------------------------------------------------------
# Capa 5: Batch (Spot + OD queues, job-def, IAM)
# -------------------------------------------------------------------------
module "batch" {
  source = "../../modules/batch"

  project              = var.project
  private_subnet_ids   = module.network.private_subnet_ids
  sg_batch_id          = module.network.sg_batch_id
  ecr_trainer_url      = module.storage.ecr_trainer_url
  trainer_image_tag    = var.trainer_image_tag
  spot_max_vcpus       = var.spot_max_vcpus
  ondemand_max_vcpus   = var.ondemand_max_vcpus
  instance_type        = var.batch_instance_type
  tracking_uri         = module.mlflow.tracking_uri
  artifacts_bucket     = module.storage.artifacts_bucket
  artifacts_bucket_arn = module.storage.artifacts_bucket_arn
  data_bucket_arn      = module.storage.data_bucket_arn
  log_retention_days   = var.log_retention_days
}
```

> **Checkpoint**: batch depende de network (subnets/SG), storage (ECR
> + buckets) y mlflow (`tracking_uri`). El `tracking_uri` se inyecta
> como env var al job-def para que los containers de entrenamiento
> sepan a donde loguear runs sin hardcodearlo.

> **Gotcha #3.7**: módulo split en `main.tf + iam.tf`. Usar `name` (no `compute_environment_name` deprecado) — key para AWS provider v6.

> **Gotcha #3.7.b**: en la job definition NO pongas `networkConfiguration` (`assignPublicIp`) — es **solo Fargate**. Con compute environments EC2 el `RegisterJobDefinition` falla con `ClientException: networkConfiguration not applicable for EC2.` (HTTP 400) y el apply muere despues de crear las queues. En EC2 la IP publica la define la subnet/compute environment, no la job-def. `terraform validate` pasa igual porque el rechazo ocurre del lado de AWS, en apply.

---

## 3.8 `modules/monitoring/` — SNS + alarmas

Genera **una alarma MAPE por variedad** (no hardcoded a POP como en V1).
La alarma escucha la custom metric que el trainer va a emitir en Parte 5.

### 3.8.1 `modules/monitoring/variables.tf`

```hcl
variable "project" { type = string }
variable "alert_email" { type = string }
variable "batch_job_queue_name" { type = string }
variable "alb_arn_suffix" {
  type        = string
  description = "Suffix del ALB ARN (formato 'app/<name>/<id>'). Usado por CloudWatch metrics."
}
variable "varieties" { type = list(string) }
variable "mape_alarm_threshold" { type = number }
```

> **Por qué `alb_arn_suffix` y no `alb_arn`**: CloudWatch espera el suffix exacto
> en la dimensión `LoadBalancer`, no el ARN completo. `aws_lb` expone `arn_suffix`
> nativo; extraerlo con `split()` del ARN sería frágil ante cambios de formato.

### 3.8.2 `modules/monitoring/main.tf`

> **Equivalente en AWS Console — el patron SNS + CloudWatch Alarms**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_sns_topic.alerts` | **SNS** | `SNS > Topics > Create topic`. **Type**: Standard. **Name**: `ml-training-alerts`. |
> | `aws_sns_topic_subscription.email` | **SNS** | Dentro del topic: `Create subscription`. **Protocol**: Email. **Endpoint**: `abantodca@gmail.com`. Status queda en **PendingConfirmation** hasta que clickees el mail "AWS Notification - Subscription Confirmation". |
> | `aws_cloudwatch_metric_alarm.batch_failed` | **CloudWatch** | `CloudWatch > Alarms > Create alarm > Select metric > Batch > By Job Queue`. Selecciona la metrica `FailedJobs` con dim `JobQueue=ml-training-job-queue-spot`. Statistic: Sum. Period: 5 min. Threshold: `>= 1`. Notification: SNS topic `ml-training-alerts`. |
> | `aws_cloudwatch_metric_alarm.mape_high` (1 por variedad) | **CloudWatch** | Mismo wizard pero `Custom namespace > ml-training/Training > MAPE`, dimension `variety=<X>`. Threshold: `> 20%`. **Treat missing data**: notBreaching (importante: si no hay datos, no dispares falsa alarma — por defecto Console pone "missing" que causa false positives). |
> | `aws_cloudwatch_metric_alarm.alb_5xx` | **CloudWatch** | Mismo wizard, namespace `AWS/ApplicationELB > Per AppELB Metrics`, metric `HTTPCode_Target_5XX_Count`. Threshold: `> 10` en 2 periodos consecutivos de 5 min. |
>
> **Conceptualmente — el pipeline de alertas**:
> - **SNS Topic** = canal pub/sub central: todas las alarmas (CloudWatch, dispatcher, EventBridge) publican acá y un solo subscriber (tu email) recibe todo. Agregar Slack/PagerDuty después no toca las alarmas.
> - **CloudWatch Alarm** = evalúa una métrica; al pasar `OK → ALARM` publica al SNS de `alarm_actions`.
> - **`for_each` en mape_high**: una alarma POR variedad, así el mail dice "MAPE de POP superó 20%" (debug instantáneo).
> - **`treat_missing_data = notBreaching`**: sin datos (no entrenaste hoy) queda en `INSUFFICIENT_DATA`, no dispara (el default de Console daría falsa alarma).
> - **`evaluation_periods = 2` en ALB 5xx**: exige 2 periodos consecutivos para no disparar por un spike transient. MAPE usa 1 (el trainer publica 1 valor por run).

```hcl
# ----- SNS topic + suscripcion email ----------------------------------
resource "aws_sns_topic" "alerts" {
  name = "${var.project}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ----- Alarma 1: Batch job FAILED -------------------------------------
# CloudWatch publica metricas de Batch (FailedJobs por queue) cada 5 min.
resource "aws_cloudwatch_metric_alarm" "batch_failed" {
  alarm_name          = "${var.project}-batch-job-failed"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "FailedJobs"
  namespace           = "AWS/Batch"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Al menos un Batch job fallo (no por Spot interrupt)"
  treat_missing_data  = "notBreaching"
  dimensions = {
    JobQueue = var.batch_job_queue_name
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
}

# ----- Alarma 2: MAPE alto, por variedad ------------------------------
# Custom metric "MAPE" en namespace "${project}/Training", dimension
# `variety`. Emitida por el trainer (Parte 5).
resource "aws_cloudwatch_metric_alarm" "mape_high" {
  for_each = toset(var.varieties)

  alarm_name          = "${var.project}-mape-${lower(each.value)}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "MAPE"
  namespace           = "${var.project}/Training"
  period              = 3600 # 1h (MAPE se publica al final del run)
  statistic           = "Maximum"
  threshold           = var.mape_alarm_threshold
  alarm_description   = "MAPE de ${each.value} supero ${var.mape_alarm_threshold}%"
  treat_missing_data  = "notBreaching"
  dimensions = {
    variety = each.value
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

# ----- Alarma 3: ALB 5xx -----------------------------------------------
resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${var.project}-alb-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  treat_missing_data  = "notBreaching"
  dimensions = {
    LoadBalancer = var.alb_arn_suffix
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
}
```

### 3.8.3 `modules/monitoring/outputs.tf`

```hcl
output "sns_topic_arn" { value = aws_sns_topic.alerts.arn }
```

> **En consola AWS veras**:
> - SNS → Topics → `ml-training-alerts` con 1 subscription
>   (`Protocol=email`, `Endpoint=<alert_email>`, `Status=PendingConfirmation`
>   hasta que clickees el mail de #4.6).
> - CloudWatch → Alarms → **N + 2 alarmas** con prefijo `ml-training-`,
>   donde N = `length(var.varieties)`. El conteo escala automatic si
>   agregas/quitas variedades — `for_each` se reconcilia en el proximo
>   `terraform apply`:
>   - `ml-training-batch-job-failed` (Batch FailedJobs sum > 0 en 5 min) — 1 sola.
>   - `ml-training-mape-<variety>` — **una por variedad**
>     (`for_each = toset(var.varieties)`). Custom metric
>     namespace=`ml-training/Training`, dim=`variety`, threshold default
>     20%. En `INSUFFICIENT_DATA` hasta el primer training (esperable).
>   - `ml-training-alb-5xx` (HTTPCode_Target_5XX_Count del ALB) — 1 sola.
> - Cada alarma con `AlarmActions=[<topic-arn>]`. La de Batch tambien
>   tiene `OKActions` para mandar mail al recuperar.

### 3.8.4 Apendear `module "monitoring"` en `infra/envs/prod/main.tf`

Pegar AL FINAL de `infra/envs/prod/main.tf` (despues de
`module "batch"` de #3.7.5):

```hcl
# -------------------------------------------------------------------------
# Capa 6: Monitoring (SNS + alarmas CloudWatch)
# -------------------------------------------------------------------------
module "monitoring" {
  source = "../../modules/monitoring"

  project              = var.project
  alert_email          = var.alert_email
  batch_job_queue_name = module.batch.job_queue_spot
  alb_arn_suffix       = module.mlflow.alb_arn_suffix
  varieties            = var.varieties_allowed
  mape_alarm_threshold = var.mape_alarm_threshold
}
```

> **Checkpoint**: monitoring lee el nombre de la queue de `module.batch`
> (para alarmas de jobs failed) y el `alb_arn_suffix` de `module.mlflow`
> (para alarma 5XX del ALB). El `sns_topic_arn` que genera lo consumira
> `module.lambdas` en #3.9.7 (notifier).

> **Gotcha #3.8**: si `varieties` está vacío en tfvars, el `for_each` de alarmas MAPE no crea ninguna alarma (silencioso).

---

## 3.9 `modules/lambdas/` — dispatcher + notifier

Dos Lambdas:

- **dispatcher**: validacion de payload + `boto3.client('batch').submit_job`.
  Acepta `varieties` (CSV), `tuning` (`smoke|dev|prod|prod_xl`), y opcional
  `s3_data_key`. La queue se elige por `tuning`: `prod_xl` -> ondemand,
  el resto -> spot.
- **notifier**: recibe eventos EventBridge "Batch Job State Change FAILED"
  y publica un mensaje a SNS con el log link directo.

> **Orden de pegado importante**: los `.tf` de #3.9.2 y #3.9.3 usan
> `data "archive_file"` para empaquetar `infra/lambdas/dispatcher.py` y
> `infra/lambdas/notifier.py`. Si haces `terraform plan` antes de crear
> esos `.py`, falla con "no such file or directory". Para evitarlo:
>
> 1. **Crear primero los `.py`** — saltar a #3.9.5 (dispatcher.py) y
>    #3.9.6 (notifier.py), pegar el codigo en
>    `infra/lambdas/dispatcher.py` y `infra/lambdas/notifier.py`.
> 2. **Luego volver aca** y pegar los `.tf` (3.9.1 -> 3.9.4).
>
> El orden de presentacion (`.tf` antes que `.py`) es para leer la
> arquitectura primero (variables -> resources -> outputs); el orden
> de creacion de archivos es al reves.

### 3.9.1 `modules/lambdas/variables.tf`

```hcl
variable "project" { type = string }
variable "job_queue_spot_arn" { type = string }
variable "job_queue_ondemand_arn" { type = string }
# *_name vars: el dispatcher.py / notifier.py usan los NAMES (no ARNs)
# para `batch.submit_job` / `batch.describe_jobs`. Antes el .tf construia
# `"${var.project}-job-queue-spot"` inline; ahora se reciben como input
# del envs/prod (wireado desde module.batch.job_queue_spot/ondemand).
variable "job_queue_spot_name" { type = string }
variable "job_queue_ondemand_name" { type = string }
variable "job_definition_name" { type = string }
variable "data_bucket" { type = string }
variable "varieties_allowed" { type = list(string) }
variable "sns_topic_arn" { type = string }
variable "batch_log_group_name" { type = string }
variable "log_retention_days" { type = number }
variable "lambdas_src_dir" { type = string }
```

### 3.9.2 `modules/lambdas/dispatcher.tf`

> 📂 **Pegar este bloque en**: `infra/modules/lambdas/dispatcher.tf`
> (el modulo `lambdas/` esta split en 3 archivos: `dispatcher.tf`
> (esta seccion) + `notifier.tf` (#3.9.3) + `outputs.tf` (#3.9.4)).
>
> **Equivalente en AWS Console — pieza por pieza del dispatcher**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `data "archive_file"` | **(local Terraform)** | NO es AWS — Terraform comprime localmente `dispatcher.py` → `dispatcher.zip`. En Console, vos tendrias que hacer el `zip` a mano antes de subir. |
> | `aws_iam_role.dispatcher` + inline policy | **IAM** | `IAM > Roles > Create role > AWS service > Lambda`. **Permissions**: inline policy con `batch:SubmitJob`, `batch:DescribeJobs`, `batch:TagResource` sobre las 2 queues + job-def, y `logs:Create*/PutLogEvents` para CloudWatch. **Name**: `ml-training-dispatcher`. ⚠️ El ARN de la job-def debe ir **con y sin** `:revision` (`...trainer` y `...trainer:*`): SubmitJob por NAME autoriza contra el ARN sin revision, que no matchea el `:*`. Y `TagResource` es obligatorio porque `submit_job` pasa `tags={...}`. |
> | `aws_cloudwatch_log_group.dispatcher` | **CloudWatch** | `Create log group`. Name: `/aws/lambda/ml-training-dispatcher`. (Lambda lo crea automaticamente la primera vez que loggeas, pero creandolo explicito te permite setear retention.) |
> | `aws_lambda_function.dispatcher` | **λ Lambda** | `Lambda > Functions > Create function > Author from scratch`. **Name**: `ml-training-dispatcher`. **Runtime**: Python 3.12. **Architecture**: x86_64. **Execution role**: el creado arriba. **Upload from**: `.zip file` → subir `dispatcher.zip`. **Handler**: `dispatcher.handler` (formato `<filename>.<function>`). **Configuration > General**: Timeout 60s, Memory 256 MB. **Configuration > Environment variables**: agregar las 6 vars (PROJECT, JOB_QUEUE_SPOT, etc.). |
>
> **Conceptualmente — Lambda como "REST endpoint para invocar Batch"**:
> - Lambda es **serverless compute** — pagas SOLO por ejecucion (no por estar prendido). Cuando alguien la invoca, AWS arranca un container con tu codigo, corre tu funcion, devuelve resultado, y apaga el container. ~100ms cold-start, ~10ms warm.
> - **Por que un dispatcher Lambda en vez de `aws batch submit-job` directo desde el cliente**:
>   - **Punto unico de validacion**: el dispatcher chequea que `varieties` esten en `VARIETIES_ALLOWED`, que `tuning` sea un preset valido, que `s3_data_key` exista. Si vos invocas batch directo, podrias submitir `--tuning xxx` sin querer y gastar EC2 corriendo basura.
>   - **Seleccion automatica de queue**: el dispatcher decide spot vs ondemand por `tuning`. Sin esto, cada caller tendria que conocer la regla.
>   - **Permission boundary**: el rol `gha-train` (Parte 3.11) tiene SOLO permiso de `lambda:InvokeFunction` sobre el dispatcher. No tiene `batch:SubmitJob` directo. Si comprometen GHA, el atacante solo puede invocar el dispatcher con payloads validos (no puede submitir jobs ad-hoc con imagenes raras).
> - **`source_code_hash`**: cuando cambia el .zip, Terraform lo detecta via hash y dispara redeploy. Sin esto, Terraform veria "el filename no cambio" y no haria nada (el zip se reconstruiria pero Lambda seguiria con la version vieja).

```hcl
# infra/modules/lambdas/dispatcher.tf
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Empaca el codigo Python en zip
data "archive_file" "dispatcher" {
  type        = "zip"
  source_file = "${var.lambdas_src_dir}/dispatcher.py"
  output_path = "${path.module}/dispatcher.zip"
}

# IAM (trust policy compartida en infra/modules/_shared/assume-lambda.json)
resource "aws_iam_role" "dispatcher" {
  name               = "${var.project}-dispatcher"
  assume_role_policy = file("${path.module}/../_shared/assume-lambda.json")
}

resource "aws_iam_role_policy" "dispatcher" {
  role = aws_iam_role.dispatcher.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        # TagResource: SubmitJob pasa tags={...}, lo cual requiere batch:TagResource
        Action = ["batch:SubmitJob", "batch:DescribeJobs", "batch:TagResource"]
        Resource = [
          var.job_queue_spot_arn,
          var.job_queue_ondemand_arn,
          # SubmitJob por NAME autoriza contra el ARN SIN revision; SubmitJob
          # por ARN con revision matchea el patron `:*`. Hacen falta ambos.
          "arn:aws:batch:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:job-definition/${var.job_definition_name}",
          "arn:aws:batch:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:job-definition/${var.job_definition_name}:*"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "dispatcher" {
  name              = "/aws/lambda/${var.project}-dispatcher"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "dispatcher" {
  function_name    = "${var.project}-dispatcher"
  role             = aws_iam_role.dispatcher.arn
  runtime          = "python3.12"
  handler          = "dispatcher.handler"
  filename         = data.archive_file.dispatcher.output_path
  source_code_hash = data.archive_file.dispatcher.output_base64sha256
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      PROJECT = var.project
      # AWS Batch SubmitJob/ListJobs aceptan ARN o name; usamos NAME.
      # Antes el .tf construia `"${var.project}-job-queue-spot"` inline;
      # ahora se recibe como input (var.job_queue_spot_name) wireado
      # desde module.batch.job_queue_spot — single source of truth.
      JOB_QUEUE_SPOT     = var.job_queue_spot_name
      JOB_QUEUE_ONDEMAND = var.job_queue_ondemand_name
      JOB_DEFINITION     = var.job_definition_name
      DATA_BUCKET        = var.data_bucket
      VARIETIES_ALLOWED  = join(",", var.varieties_allowed)
    }
  }

  depends_on = [aws_cloudwatch_log_group.dispatcher]
}
```

### 3.9.3 `modules/lambdas/notifier.tf`

> 📂 **Pegar este bloque en**: `infra/modules/lambdas/notifier.tf`
> (archivo distinto al `dispatcher.tf` de #3.9.2; ambos coexisten en
> el mismo modulo `lambdas/`).
>
> **Equivalente en AWS Console — Lambda + EventBridge trigger**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_iam_role.notifier` + inline policy | **IAM** | Wizard de Lambda role. **Permissions**: inline policy con `sns:Publish` (al topic), `batch:DescribeJobs`, `logs:*`. **Name**: `ml-training-notifier`. |
> | `aws_lambda_function.notifier` | **λ Lambda** | Mismo wizard que dispatcher. **Name**: `ml-training-notifier`. **Timeout**: 30s. **Memory**: 128 MB. Env: `SNS_TOPIC_ARN` + `BATCH_LOG_GROUP` (nombre real del log group, ej. `/aws/batch/ml-training`, propagado desde el output del modulo batch — antes era construido inline con `f"/aws/batch/{PROJECT}"`, frágil si cambia el patron). |
> | `aws_cloudwatch_event_rule.batch_failed` | **EventBridge** | `EventBridge > Rules > Create rule`. **Name**: `ml-training-batch-failed`. **Event bus**: default. **Rule type**: Rule with an event pattern. **Event source**: AWS services > AWS Batch > Batch Job State Change. **Specific status(es)**: FAILED. La consola te muestra un preview del JSON pattern. |
> | `aws_cloudwatch_event_target.notifier` | **EventBridge** | Dentro de la rule: `Add target > Lambda function > ml-training-notifier`. |
> | `aws_lambda_permission.notifier_eventbridge` | **λ Lambda** | NO existe en Console como recurso aparte — la consola lo crea **automaticamente** al asociar el target (te pide "Add permission"). En Terraform es explicito. |
>
> **Conceptualmente — el patrón event-driven con EventBridge**:
> - **EventBridge** = bus de eventos central de AWS; casi todo servicio publica eventos (Batch "Job State Change", S3 "Object Created"…). Publicar es gratis; pagás $1/M consumidos.
> - **Rule** = filtro (`event_pattern`: `{source: aws.batch, detail-type: Batch Job State Change, status: FAILED}`) + `target` (Lambda, SNS, SQS…).
> - **Por qué no Batch → SNS directo**: SNS no filtra/transforma el payload ni hace DescribeJobs. El notifier recibe el evento, hace `batch.describe_jobs` para sacar nombre/queue/log stream, arma un mensaje legible con link al log y `sns.publish`.
> - **`aws_lambda_permission`**: Lambda no acepta invocaciones por default; cada source (EventBridge, S3, API GW) necesita una resource policy explícita. En Console es automático; en Terraform, recurso aparte.

```hcl
# infra/modules/lambdas/notifier.tf
data "archive_file" "notifier" {
  type        = "zip"
  source_file = "${var.lambdas_src_dir}/notifier.py"
  output_path = "${path.module}/notifier.zip"
}

resource "aws_iam_role" "notifier" {
  name               = "${var.project}-notifier"
  assume_role_policy = file("${path.module}/../_shared/assume-lambda.json")
}

resource "aws_iam_role_policy" "notifier" {
  role = aws_iam_role.notifier.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = var.sns_topic_arn
      },
      {
        Effect   = "Allow"
        Action   = ["batch:DescribeJobs"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "notifier" {
  name              = "/aws/lambda/${var.project}-notifier"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "notifier" {
  function_name    = "${var.project}-notifier"
  role             = aws_iam_role.notifier.arn
  runtime          = "python3.12"
  handler          = "notifier.handler"
  filename         = data.archive_file.notifier.output_path
  source_code_hash = data.archive_file.notifier.output_base64sha256
  timeout          = 30
  memory_size      = 128

  environment {
    variables = {
      SNS_TOPIC_ARN   = var.sns_topic_arn
      BATCH_LOG_GROUP = var.batch_log_group_name
    }
  }

  depends_on = [aws_cloudwatch_log_group.notifier]
}

# EventBridge rule: Batch Job State Change FAILED -> notifier
resource "aws_cloudwatch_event_rule" "batch_failed" {
  name        = "${var.project}-batch-failed"
  description = "Captura Batch jobs en estado FAILED"
  event_pattern = jsonencode({
    source        = ["aws.batch"]
    "detail-type" = ["Batch Job State Change"]
    detail = {
      status = ["FAILED"]
    }
  })
}

resource "aws_cloudwatch_event_target" "notifier" {
  rule      = aws_cloudwatch_event_rule.batch_failed.name
  target_id = "notifier"
  arn       = aws_lambda_function.notifier.arn
}

resource "aws_lambda_permission" "notifier_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.notifier.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.batch_failed.arn
}
```

### 3.9.4 `modules/lambdas/outputs.tf`

```hcl
output "dispatcher_function_name" { value = aws_lambda_function.dispatcher.function_name }
```

### 3.9.5 `infra/lambdas/dispatcher.py`

```python
"""Lambda dispatcher: submit jobs a AWS Batch.

Payload aceptado:
{
  "varieties": "POP,JUPITER",      # CSV o "all"
  "tuning":    "prod",             # smoke|dev|prod|prod_xl
  "s3_data_key": "BD_HISTORICO_ACUMULADO.xlsx"   # opcional, default = ese mismo
}

Contrato del trainer (main.py):
- CMD ["--varieties","POP,JUPITER","--tuning","prod"]
- ENV S3_DATA_BUCKET, S3_DATA_KEY (para _hydrate_data_from_s3)
- ENV MLFLOW_TRACKING_URI, S3_ARTIFACTS_BUCKET, ... (ya en job-def)
"""

from __future__ import annotations

import json
import logging
import os
import re

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

batch = boto3.client("batch")

PROJECT            = os.environ["PROJECT"]
JOB_QUEUE_SPOT     = os.environ["JOB_QUEUE_SPOT"]
JOB_QUEUE_ONDEMAND = os.environ["JOB_QUEUE_ONDEMAND"]
JOB_DEFINITION     = os.environ["JOB_DEFINITION"]
DATA_BUCKET        = os.environ["DATA_BUCKET"]
VARIETIES_ALLOWED  = set(os.environ["VARIETIES_ALLOWED"].split(","))

TUNINGS = {"smoke", "dev", "prod", "prod_xl"}


def _normalize_varieties(raw: str) -> list[str]:
    if not raw:
        raise ValueError("varieties vacio")
    raw = raw.strip()
    if raw.lower() == "all":
        return sorted(VARIETIES_ALLOWED)
    items = [v.strip().upper() for v in raw.split(",") if v.strip()]
    bad = [v for v in items if v not in VARIETIES_ALLOWED]
    if bad:
        raise ValueError(f"variedades no permitidas: {bad}. Validas: {sorted(VARIETIES_ALLOWED)}")
    return items


def _validate_tuning(tuning: str) -> str:
    if tuning not in TUNINGS:
        raise ValueError(f"tuning invalido: {tuning}. Validos: {sorted(TUNINGS)}")
    return tuning


def _validate_key(key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._/\-]+\.xlsx", key):
        raise ValueError(f"s3_data_key invalido: {key}")
    return key


def _validate_mode(mode: str) -> str:
    # train (default) entrena; eda corre el analisis exploratorio standalone.
    if mode not in ("train", "eda"):
        raise ValueError(f"mode invalido: {mode}. Validos: eda, train")
    return mode


def handler(event, _context):
    log.info("event: %s", json.dumps(event)[:1000])

    # EventBridge envuelve el payload en `detail`; manual invoke lo pasa raw.
    payload = event.get("detail", event) or {}

    try:
        varieties = _normalize_varieties(payload.get("varieties", ""))
        tuning    = _validate_tuning(payload.get("tuning", "prod"))
        s3_key    = _validate_key(payload.get("s3_data_key", "BD_HISTORICO_ACUMULADO.xlsx"))
        mode      = _validate_mode(payload.get("mode", "train"))
    except ValueError as exc:
        log.error("validacion fallo: %s", exc)
        return {"statusCode": 400, "body": str(exc)}

    queue = JOB_QUEUE_ONDEMAND if tuning == "prod_xl" else JOB_QUEUE_SPOT
    job_name = f"{PROJECT}-{'eda' if mode == 'eda' else tuning}-{'-'.join(varieties)[:50]}"
    # sanitize: Batch acepta [a-zA-Z0-9_-], max 128
    job_name = re.sub(r"[^a-zA-Z0-9_-]", "-", job_name)[:128]

    # EDA: standalone, no entrena -> ignora tuning/modelo. Training: como siempre.
    if mode == "eda":
        command = ["--eda", "--varieties", ",".join(varieties)]
    else:
        command = ["--varieties", ",".join(varieties), "--tuning", tuning]

    response = batch.submit_job(
        jobName=job_name,
        jobQueue=queue,
        jobDefinition=JOB_DEFINITION,
        containerOverrides={
            "command": command,
            "environment": [
                {"name": "S3_DATA_BUCKET", "value": DATA_BUCKET},
                {"name": "S3_DATA_KEY",    "value": s3_key},
            ],
        },
        tags={"variety": ",".join(varieties), "tuning": tuning, "mode": mode},
    )

    log.info("submit OK: jobId=%s queue=%s mode=%s", response["jobId"], queue, mode)
    return {
        "statusCode": 200,
        "body": {
            "jobId":    response["jobId"],
            "jobName":  response["jobName"],
            "queue":    queue,
            "varieties": varieties,
            "tuning":   tuning,
            "mode":     mode,
        },
    }
```

### 3.9.6 `infra/lambdas/notifier.py`

```python
"""Lambda notifier: traduce un evento de Batch FAILED a un email SNS legible."""

from __future__ import annotations

import json
import logging
import os

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

sns   = boto3.client("sns")
batch = boto3.client("batch")

SNS_TOPIC_ARN = os.environ["SNS_TOPIC_ARN"]
# AWS_REGION lo inyecta Lambda runtime automaticamente.
AWS_REGION = os.environ["AWS_REGION"]
# BATCH_LOG_GROUP es el name real (ej. "/aws/batch/ml-training"); el modulo
# batch lo expone como output y se pasa via env var. Antes se construia con
# f"/aws/batch/{PROJECT}" -> rompia silencioso si el log group cambiaba de patron.
BATCH_LOG_GROUP = os.environ["BATCH_LOG_GROUP"]


def _cw_url_encode(s: str) -> str:
    """CloudWatch UI hace doble URL-decode del log group/stream name.

    "/" se vuelve "$252F" (% URL-encoded a %25, luego %25 + 2F = $252F).
    """
    return s.replace("/", "$252F")


def handler(event, _context):
    log.info("event: %s", json.dumps(event)[:1500])

    detail = event.get("detail", {})
    job_id = detail.get("jobId")
    if not job_id:
        return {"statusCode": 400, "body": "no jobId in event"}

    job_name   = detail.get("jobName", "?")
    queue_arn  = detail.get("jobQueue", "?")
    reason     = detail.get("statusReason", "?")
    container  = detail.get("container", {})
    exit_code  = container.get("exitCode", "?")
    log_stream = container.get("logStreamName")

    log_url = "(no log stream)"
    if log_stream:
        log_url = (
            f"https://{AWS_REGION}.console.aws.amazon.com/cloudwatch/home"
            f"?region={AWS_REGION}#logsV2:log-groups/log-group/"
            f"{_cw_url_encode(BATCH_LOG_GROUP)}/log-events/"
            f"{_cw_url_encode(log_stream)}"
        )

    subject = f"[ml-training] Job FAILED: {job_name}"
    body = "\n".join([
        f"Job ID:    {job_id}",
        f"Job name:  {job_name}",
        f"Queue:     {queue_arn.rsplit('/', 1)[-1]}",
        f"Exit code: {exit_code}",
        f"Reason:    {reason}",
        f"Logs:      {log_url}",
    ])

    sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=body)
    log.info("notified jobId=%s", job_id)
    return {"statusCode": 200, "body": "notified"}
```

### 3.9.7 Apendear `module "lambdas"` en `infra/envs/prod/main.tf`

Pegar AL FINAL de `infra/envs/prod/main.tf` (despues de
`module "monitoring"` de #3.8.4):

```hcl
# -------------------------------------------------------------------------
# Capa 7: Lambdas (dispatcher + notifier)
# -------------------------------------------------------------------------
module "lambdas" {
  source = "../../modules/lambdas"

  project                 = var.project
  job_queue_spot_arn      = module.batch.job_queue_spot_arn
  job_queue_ondemand_arn  = module.batch.job_queue_ondemand_arn
  job_queue_spot_name     = module.batch.job_queue_spot
  job_queue_ondemand_name = module.batch.job_queue_ondemand
  job_definition_name     = module.batch.job_definition_name
  data_bucket             = module.storage.data_bucket
  varieties_allowed       = var.varieties_allowed
  sns_topic_arn           = module.monitoring.sns_topic_arn
  batch_log_group_name    = module.batch.log_group_name
  log_retention_days      = var.log_retention_days
  lambdas_src_dir         = "${path.module}/../../lambdas"
}
```

> **Checkpoint**: el `lambdas_src_dir` apunta a `infra/lambdas/`
> (donde estan los `.py` de #3.9.5 y #3.9.6). Si todavia no pegaste
> los `.py`, `terraform plan` truena al hacer `archive_file` del zip.
> Por eso #3.9 te dice **primero pegar los `.py`, despues los `.tf`**.

> **Gotcha #3.9**: el módulo NO tiene `main.tf`; tiene `dispatcher.tf + notifier.tf` (split). Verificar también `python3 -m py_compile infra/lambdas/dispatcher.py infra/lambdas/notifier.py` sin errores.

---

## 3.10 `modules/scheduler/` — auto on/off RDS + Fargate

Una Lambda + 2 crons EventBridge. La Lambda hace `start`/`stop` segun
el payload. Antes de stop, chequea Batch jobs RUNNING — si hay, posterga
(no apaga). Lockeado a PET (UTC-5).

> **Orden de pegado**: igual que #3.9, `modules/scheduler/main.tf`
> empaca `infra/lambdas/scheduler.py` con `data "archive_file"`. Pegar
> **primero** #3.10.4 (`scheduler.py`) en `infra/lambdas/scheduler.py`,
> y **despues** #3.10.1-#3.10.3 (los `.tf`). Asi `task infra:validate` /
> el primer apply en Parte 4 no truena por archivo inexistente.

### 3.10.1 `modules/scheduler/variables.tf`

```hcl
variable "project" { type = string }
variable "ecs_cluster_name" { type = string }
variable "ecs_service_name_mlflow" { type = string }
variable "ecs_service_name_reports" { type = string }
variable "ecs_service_name_api" { type = string }
variable "ecs_service_name_ui" { type = string }
variable "rds_instance_id" { type = string }
# *_name vars: el scheduler.py llama batch.list_jobs(jobQueue=NAME)
# para detectar jobs RUNNING antes de apagar RDS. Antes el .tf
# construia los nombres inline; ahora se reciben como input desde
# envs/prod (module.batch.job_queue_spot / job_queue_ondemand).
variable "job_queue_spot_name" { type = string }
variable "job_queue_ondemand_name" { type = string }
variable "work_start_hour_local" { type = number }
variable "work_end_hour_local" { type = number }
variable "tz_offset_hours" {
  type    = number
  default = -5 # PET (Peru)
}
variable "workdays_cron" {
  type    = string
  default = "MON,WED,FRI" # Patch 13.1: solo L/Mi/V (antes: "MON-FRI")
}
variable "log_retention_days" { type = number }
variable "lambdas_src_dir" { type = string }
```

### 3.10.2 `modules/scheduler/main.tf`

3 sub-bloques al mismo `modules/scheduler/main.tf`:

#### 3.10.2.a — Lambda function + IAM

Empaqueta `infra/lambdas/scheduler.py` (que ya creaste antes — ver
callout arriba). IAM con scope a ECS update-service, RDS start/stop,
y Batch describe — todo `Resource="*"` porque los recursos del proyecto
son los unicos en la cuenta con esos names; refinable a ARN especifico
en hardening (futuro).

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_iam_role.scheduler` + inline policy | **IAM** | `IAM > Roles > Create role > Lambda`. **Permissions**: inline policy con `ecs:UpdateService/DescribeServices`, `rds:StartDBInstance/StopDBInstance/DescribeDBInstances`, `batch:ListJobs/DescribeJobs`, `logs:*`. **Name**: `ml-training-scheduler`. |
> | `aws_lambda_function.scheduler` | **λ Lambda** | `Create function > Author from scratch`. **Name**: `ml-training-scheduler`. **Runtime**: Python 3.12. **Execution role**: el de arriba. Subir `scheduler.zip`. **Handler**: `scheduler.handler`. **Timeout**: 900s / 15 min (cubre RDS cold start ~5-8 min + espera a MLflow running). **Memory**: 256 MB. **Env vars**: PROJECT, ECS_CLUSTER, ECS_SVC_MLFLOW, ECS_SVC_REPORTS, ECS_SVC_API, ECS_SVC_UI, RDS_INSTANCE, JOB_QUEUE_SPOT, JOB_QUEUE_ONDEMAND. |
>
> **Conceptualmente — por qué UN Lambda con payload `action`, no DOS**:
> - Dos Lambdas (`start`/`stop`) duplicarían el código compartido (auth ECS, espera de healthy, errores).
> - **El patrón**: una sola Lambda que recibe `{"action": "start|stop"}` y despacha adentro (`if action == "start": _start_all()`). Reusa helpers + 1 IAM + 1 log group.
> - **`timeout=900`**: RDS cold start no es instantáneo — la Lambda espera a `available` (loop ~8 min) antes de arrancar Fargate; con 60s/300s timeoutearía.
> - **`Resource="*"` en IAM**: laxo pero los nombres son únicos al proyecto; en hardening (futuro) → ARNs específicos.

```hcl
locals {
  start_hour_utc = (var.work_start_hour_local - var.tz_offset_hours + 24) % 24
  stop_hour_utc  = (var.work_end_hour_local - var.tz_offset_hours + 24) % 24
}

data "archive_file" "scheduler" {
  type        = "zip"
  source_file = "${var.lambdas_src_dir}/scheduler.py"
  output_path = "${path.module}/scheduler.zip"
}

# Trust policy compartida en infra/modules/_shared/assume-lambda.json
resource "aws_iam_role" "scheduler" {
  name               = "${var.project}-scheduler"
  assume_role_policy = file("${path.module}/../_shared/assume-lambda.json")
}

resource "aws_iam_role_policy" "scheduler" {
  role = aws_iam_role.scheduler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecs:UpdateService", "ecs:DescribeServices"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "rds:StartDBInstance", "rds:StopDBInstance", "rds:DescribeDBInstances"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["batch:ListJobs", "batch:DescribeJobs"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "scheduler" {
  name              = "/aws/lambda/${var.project}-scheduler"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "scheduler" {
  function_name    = "${var.project}-scheduler"
  role             = aws_iam_role.scheduler.arn
  runtime          = "python3.12"
  handler          = "scheduler.handler"
  filename         = data.archive_file.scheduler.output_path
  source_code_hash = data.archive_file.scheduler.output_base64sha256
  timeout          = 900 # Patch 13.3: 15 min (antes 300). Cubre RDS cold start (~5-8 min) + wait MLflow.
  memory_size      = 256

  environment {
    variables = {
      PROJECT         = var.project
      ECS_CLUSTER     = var.ecs_cluster_name
      ECS_SVC_MLFLOW  = var.ecs_service_name_mlflow
      ECS_SVC_REPORTS = var.ecs_service_name_reports
      ECS_SVC_API     = var.ecs_service_name_api
      ECS_SVC_UI      = var.ecs_service_name_ui
      RDS_INSTANCE    = var.rds_instance_id
      # Antes los names se construian inline (`"${var.project}-job-queue-spot"`);
      # ahora vienen como input wireado desde module.batch en envs/prod.
      JOB_QUEUE_SPOT     = var.job_queue_spot_name
      JOB_QUEUE_ONDEMAND = var.job_queue_ondemand_name
      # Patch 13.1: propagar workdays + ventana al _keepstop (sino el
      # martes/jueves queda "dentro de ventana" y nunca re-para el RDS).
      WORKDAYS_CRON  = var.workdays_cron
      WORK_START_UTC = tostring(local.start_hour_utc)
      WORK_END_UTC   = tostring(local.stop_hour_utc)
    }
  }

  depends_on = [aws_cloudwatch_log_group.scheduler]
}
```

#### 3.10.2.b — EventBridge rules (start, stop, keepstop)

3 crons: `start` (8 AM PET), `stop` (12 PM PET), `keepstop` (cada 6h
defensa contra el auto-arranque de RDS post-7-dias-stopped). El offset
PET→UTC se calcula en `locals` y se enchufa al `cron(0 H ? * MON-FRI *)`.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_cloudwatch_event_rule.start` | **EventBridge** | `EventBridge > Rules > Create rule`. **Name**: `ml-training-start`. **Event bus**: default. **Rule type**: Schedule. **Schedule pattern**: A fine-grained schedule that runs at a specific time. **Cron expression**: `cron(0 13 ? * MON,WED,FRI *)` (13 UTC = 08:00 PET; default workdays_cron). |
> | `aws_cloudwatch_event_target.start` | **EventBridge** | Dentro de la rule: `Add target > Lambda function > ml-training-scheduler`. **Configure target input**: Constant (JSON text): `{"action": "start"}`. |
> | `aws_cloudwatch_event_rule.stop` + target | **EventBridge** | Mismo wizard, name `-stop`, cron `cron(0 17 ? * MON,WED,FRI *)` (17 UTC = 12:00 PET), input `{"action":"stop"}`. |
> | `aws_cloudwatch_event_rule.rds_keepstop` + target | **EventBridge** | Mismo wizard, name `-rds-keepstop`, **Schedule pattern**: A schedule that runs at a regular rate. **Rate expression**: `rate(6 hours)`. Input `{"action":"keepstop"}`. |
>
> **Conceptualmente — por qué 3 rules y no 1 sola**:
> - Una sola rule con 3 targets ejecutaría los 3 inputs en cada tick. Cada rule tiene 1 propósito y 1 cron.
> - **Cron de EventBridge**: `cron(min hour day-of-month month day-of-week year)` (6 campos, no los 5 de Linux). El `?` = "no me importa" (mutuamente excluyente entre day-of-month y day-of-week).
> - **`MON,WED,FRI`**: este repo entrena 3 días; cualquier subset funciona.
> - **`rate(6 hours)` para keepstop**: RDS dejado `stopped` >7 días lo enciende AWS solo "por mantenimiento" (~$15 sorpresa/mes). El keepstop lo re-apaga si lo encuentra `available` fuera de ventana.

```hcl
resource "aws_cloudwatch_event_rule" "start" {
  name                = "${var.project}-start"
  description         = "L-V ${var.work_start_hour_local}:00 PET start RDS+Fargate"
  schedule_expression = "cron(0 ${local.start_hour_utc} ? * ${var.workdays_cron} *)"
}

resource "aws_cloudwatch_event_target" "start" {
  rule      = aws_cloudwatch_event_rule.start.name
  target_id = "scheduler-start"
  arn       = aws_lambda_function.scheduler.arn
  input     = jsonencode({ action = "start" })
}

# ----- EventBridge: cron STOP L-V <stop_hour_utc>:00 -----------------
resource "aws_cloudwatch_event_rule" "stop" {
  name                = "${var.project}-stop"
  description         = "L-V ${var.work_end_hour_local}:00 PET stop RDS+Fargate"
  schedule_expression = "cron(0 ${local.stop_hour_utc} ? * ${var.workdays_cron} *)"
}

resource "aws_cloudwatch_event_target" "stop" {
  rule      = aws_cloudwatch_event_rule.stop.name
  target_id = "scheduler-stop"
  arn       = aws_lambda_function.scheduler.arn
  input     = jsonencode({ action = "stop" })
}

# ----- Cron extra: cada 6h chequea RDS y lo re-stop si quedo RUNNING --
# (necesario porque RDS auto-arranca despues de 7 dias stopped)
resource "aws_cloudwatch_event_rule" "rds_keepstop" {
  name                = "${var.project}-rds-keepstop"
  description         = "Cada 6h: re-stop RDS si quedo RUNNING fuera de ventana"
  schedule_expression = "rate(6 hours)"
}

resource "aws_cloudwatch_event_target" "rds_keepstop" {
  rule      = aws_cloudwatch_event_rule.rds_keepstop.name
  target_id = "scheduler-keepstop"
  arn       = aws_lambda_function.scheduler.arn
  input     = jsonencode({ action = "keepstop" })
}
```

#### 3.10.2.c — Permissions (EventBridge → Lambda)

EventBridge no puede invocar Lambdas por defecto; cada rule necesita
su propia `lambda_permission` con `source_arn` matching. 3 rules =
3 permissions.

> **Equivalente en AWS Console**:
> 
> En Console, cuando agregas una rule como target de una Lambda, el wizard pregunta automaticamente "Add the necessary permissions for the target to be invoked by this rule?" → al hacer click en "Confirm", la consola agrega esta resource policy a la Lambda. Por eso en Console no ves recursos `aws_lambda_permission` aparte — son **invisibles, gestionados por el wizard**.
>
> En Terraform es **explicito** porque Terraform no infiere ese tipo de "side effect" — necesita un recurso declarativo. Si te olvidas el `aws_lambda_permission`, EventBridge dispara la rule, pero Lambda devuelve `403 AccessDenied` y nunca corre.
>
> **Conceptualmente — el modelo de permisos cruzados de AWS**: dos lados autorizan la invocación:
> - **Invocador (EventBridge)**: tiene `lambda:InvokeFunction` implícito, no requiere tf.
> - **Invocado (Lambda)**: una "resource policy" que permite a `events.amazonaws.com` (con `source_arn` matching) invocarla — **ESTE** es el `aws_lambda_permission`.
>
> Sin ella, Lambda rechaza aunque el caller tenga IAM. Mismo patrón que `aws_lambda_permission.notifier_eventbridge` (#3.9.3).

```hcl
resource "aws_lambda_permission" "start" {
  statement_id  = "AllowStart"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.start.arn
}

resource "aws_lambda_permission" "stop" {
  statement_id  = "AllowStop"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.stop.arn
}

resource "aws_lambda_permission" "keepstop" {
  statement_id  = "AllowKeepstop"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.rds_keepstop.arn
}
```

> **Checkpoint despues de 3.10.2.a-c**: `terraform fmt
> infra/modules/scheduler/main.tf`.

### 3.10.3 `modules/scheduler/outputs.tf`

```hcl
# scheduler no expone outputs: opera por side-effects (Lambda scheduler +
# EventBridge start/stop rules). El parent (envs/prod) no consume nada de aqui.
```

### 3.10.4 `infra/lambdas/scheduler.py`

> **NOTA**: este es el **archivo real de produccion** `infra/lambdas/scheduler.py`
> (paridad 1:1 con el repo). Incluye los patches integrados:
> - **Patch 13.1 — ventana configurable**: `_parse_workdays` parsea `WORKDAYS_CRON`
>   (`MON,WED,FRI` o `MON-FRI`) y `_keepstop` usa `WORK_START_UTC`/`WORK_END_UTC`
>   (inyectados por el modulo, #3.10.2) en vez de tener la ventana hardcodeada.
> - **Patch 13.3 — wake secuencial**: `_start` arranca **RDS → MLflow (con wait
>   hasta available/running) → Reports + API + UI**, no en paralelo. Asi el
>   container MLflow no falla el healthcheck por arrancar antes que RDS, y la API
>   (lazy-load) tolera que MLflow todavia este cargando.
> - **App stack (Capa 4.5)**: `_start`/`_stop` escalan tambien `api` + `ui`
>   (`ECS_SVC_API`/`ECS_SVC_UI`, con defaults tolerantes `"api"`/`"ui"`).

```python
"""Lambda scheduler: start/stop RDS + Fargate.

Acciones:
- start:    arranca RDS + ECS services secuencialmente (RDS -> MLflow -> Reports)
- stop:     baja ECS services a 0 + para RDS. Antes chequea Batch jobs RUNNING.
- keepstop: cada 6h. Si RDS quedo RUNNING fuera de ventana, lo re-para.
            Ventana parametrizada via WORKDAYS_CRON + WORK_START_UTC + WORK_END_UTC.
"""

from __future__ import annotations

import logging
import os
import time

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

ecs   = boto3.client("ecs")
rds   = boto3.client("rds")
batch = boto3.client("batch")

ECS_CLUSTER        = os.environ["ECS_CLUSTER"]
ECS_SVC_MLFLOW     = os.environ["ECS_SVC_MLFLOW"]
ECS_SVC_REPORTS    = os.environ["ECS_SVC_REPORTS"]
# App stack (defaults tolerantes: los servicios se llaman literalmente api/ui).
ECS_SVC_API        = os.environ.get("ECS_SVC_API", "api")
ECS_SVC_UI         = os.environ.get("ECS_SVC_UI", "ui")
RDS_INSTANCE       = os.environ["RDS_INSTANCE"]
JOB_QUEUE_SPOT     = os.environ["JOB_QUEUE_SPOT"]
JOB_QUEUE_ONDEMAND = os.environ["JOB_QUEUE_ONDEMAND"]

# Patch 13.1: workdays + horas configurables via env (default = comportamiento
# original MON-FRI 13-17 UTC). EventBridge cron usa los mismos tokens (MON,WED,FRI).
_WEEKDAY_MAP = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def _parse_workdays(cron_token: str) -> set[int]:
    """Parsea 'MON,WED,FRI' o 'MON-FRI' a un set de tm_wday (0=lunes)."""
    cron_token = cron_token.strip().upper()
    if "-" in cron_token:
        a, b = cron_token.split("-", 1)
        ia, ib = _WEEKDAY_MAP[a.strip()], _WEEKDAY_MAP[b.strip()]
        return set(range(ia, ib + 1))
    return {_WEEKDAY_MAP[tok.strip()] for tok in cron_token.split(",") if tok.strip()}


def _running_jobs() -> list[str]:
    """IDs de jobs en estado RUNNING o RUNNABLE en cualquiera de las queues."""
    ids: list[str] = []
    for queue in (JOB_QUEUE_SPOT, JOB_QUEUE_ONDEMAND):
        for status in ("RUNNING", "RUNNABLE", "STARTING"):
            resp = batch.list_jobs(jobQueue=queue, jobStatus=status)
            ids.extend(j["jobId"] for j in resp.get("jobSummaryList", []))
    return ids


def _start():
    """Wake secuencial: RDS -> MLflow -> Reports (Patch 13.3).

    Por que serializar y no lanzar todo en paralelo:
    1. El container MLflow intenta conectar a RDS al arrancar. Si RDS
       no esta available, falla healthcheck startPeriod -> ECS lo
       reinicia. Costoso en tiempo.
    2. Reports depende de S3 (no de RDS o MLflow) pero su UI vacia es
       confusa si MLflow todavia esta cargando. Mejor secuencial.
    """
    log.info("=== START (secuencial: RDS -> MLflow -> Reports) ===")

    # Etapa 1: RDS
    db = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE)["DBInstances"][0]
    if db["DBInstanceStatus"] == "stopped":
        rds.start_db_instance(DBInstanceIdentifier=RDS_INSTANCE)
        log.info("rds start_db_instance ack")

    # Esperar hasta available (max ~8 min)
    state = db["DBInstanceStatus"]
    for i in range(48):
        db = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE)["DBInstances"][0]
        state = db["DBInstanceStatus"]
        log.info("rds[%d]=%s", i, state)
        if state == "available":
            break
        time.sleep(10)
    else:
        raise RuntimeError(f"RDS no available tras 8 min (estado={state})")

    log.info("rds OK -> arrancando MLflow")

    # Etapa 2: MLflow Fargate
    ecs.update_service(cluster=ECS_CLUSTER, service=ECS_SVC_MLFLOW, desiredCount=1)
    log.info("ecs %s -> desiredCount=1", ECS_SVC_MLFLOW)

    # Esperar hasta running (max ~5 min). Si no llega, igual arrancamos reports.
    for i in range(30):
        svc = ecs.describe_services(cluster=ECS_CLUSTER, services=[ECS_SVC_MLFLOW])["services"][0]
        running = svc.get("runningCount", 0)
        log.info("mlflow[%d]: running=%d desired=%d", i, running, svc.get("desiredCount", 0))
        if running >= 1:
            break
        time.sleep(10)
    else:
        log.warning("MLflow no esta running tras 5 min, arrancamos reports igual")

    # Etapa 3: Reports + API + UI Fargate (no esperan, son no-bloqueantes).
    # La API tolera que MLflow aun no este listo (lazy-load); RDS ya esta up.
    for svc in (ECS_SVC_REPORTS, ECS_SVC_API, ECS_SVC_UI):
        ecs.update_service(cluster=ECS_CLUSTER, service=svc, desiredCount=1)
        log.info("ecs %s -> desiredCount=1", svc)
    log.info("=== START OK ===")


def _stop():
    log.info("=== STOP ===")
    running = _running_jobs()
    if running:
        log.warning(
            "Batch jobs activos (%d): %s. Postponiendo stop hasta proximo cron.",
            len(running), running[:5]
        )
        return

    # ECS: desired_count = 0 (incluye app stack api + ui)
    for svc in (ECS_SVC_MLFLOW, ECS_SVC_REPORTS, ECS_SVC_API, ECS_SVC_UI):
        ecs.update_service(cluster=ECS_CLUSTER, service=svc, desiredCount=0)
        log.info("ecs %s -> desiredCount=0", svc)

    # RDS: stop si esta RUNNING
    db = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE)["DBInstances"][0]
    state = db["DBInstanceStatus"]
    if state == "available":
        rds.stop_db_instance(DBInstanceIdentifier=RDS_INSTANCE)
        log.info("rds stop_db_instance ack")
    else:
        log.info("rds en estado %s (skip stop)", state)


def _keepstop():
    """Defense: si RDS quedo RUNNING fuera de ventana, re-pararlo (Patch 13.1)."""
    log.info("=== KEEPSTOP ===")
    workdays = _parse_workdays(os.environ.get("WORKDAYS_CRON", "MON-FRI"))
    start_utc = int(os.environ.get("WORK_START_UTC", "13"))
    end_utc = int(os.environ.get("WORK_END_UTC", "17"))

    utc_hour = time.gmtime().tm_hour
    weekday = time.gmtime().tm_wday   # 0=lunes
    in_window = (weekday in workdays) and (start_utc <= utc_hour < end_utc)
    if in_window:
        log.info("dentro de ventana (UTC=%02d:00, weekday=%d, workdays=%s), skip",
                 utc_hour, weekday, sorted(workdays))
        return

    db = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE)["DBInstances"][0]
    state = db["DBInstanceStatus"]
    if state == "available":
        running = _running_jobs()
        if running:
            log.warning("Batch jobs activos, skip keepstop")
            return
        rds.stop_db_instance(DBInstanceIdentifier=RDS_INSTANCE)
        log.info("rds re-stopped por keepstop")
    else:
        log.info("rds en estado %s (skip)", state)


def handler(event, _context):
    action = (event or {}).get("action", "stop")
    if action == "start":
        _start()
    elif action == "stop":
        _stop()
    elif action == "keepstop":
        _keepstop()
    else:
        raise ValueError(f"action desconocida: {action}")
    return {"statusCode": 200, "body": action}
```

### 3.10.5 Apendear `module "scheduler"` en `infra/envs/prod/main.tf`

Pegar AL FINAL de `infra/envs/prod/main.tf` (despues de
`module "lambdas"` de #3.9.7):

```hcl
# -------------------------------------------------------------------------
# Capa 8: Scheduler (auto on/off RDS + Fargate)
# -------------------------------------------------------------------------
module "scheduler" {
  source = "../../modules/scheduler"

  project                  = var.project
  ecs_cluster_name         = module.mlflow.cluster_name
  ecs_service_name_mlflow  = module.mlflow.service_name
  ecs_service_name_reports = module.reports.service_name
  ecs_service_name_api     = module.api.service_name
  ecs_service_name_ui      = module.ui.service_name
  rds_instance_id          = module.mlflow.rds_instance_id
  job_queue_spot_name      = module.batch.job_queue_spot
  job_queue_ondemand_name  = module.batch.job_queue_ondemand
  work_start_hour_local    = var.work_start_hour_local
  work_end_hour_local      = var.work_end_hour_local
  log_retention_days       = var.log_retention_days
  lambdas_src_dir          = "${path.module}/../../lambdas"
}
```

> **Checkpoint**: scheduler consume `cluster_name` / `service_name` /
> `rds_instance_id` para escalar Fargate a 0 y parar RDS fuera de
> horario laboral. Igual que #3.9.7, depende de que `scheduler.py`
> (#3.10.4) ya este pegado.

> **Gotcha #3.10**: la guía muestra la versión **real de producción** de `scheduler.py` (paridad 1:1 con el repo): `WORKDAYS_CRON` + wake secuencial RDS→MLflow→Reports + app stack api/ui. Validar tras pegar con `python3 -m py_compile infra/lambdas/scheduler.py`.

---

## 3.11 `modules/cicd/` — OIDC trust + GHA roles

Dos roles: `gha-deploy` (terraform apply, push ECR) y `gha-train`
(invoke Lambda dispatcher). Trust policy especifica `repo:org/repo:*`
para evitar que cualquier repo pueda asumir.

> **🔗 Orden OIDC end-to-end** (cuándo se crea qué):
>
> 1. **#2.5** crea el OIDC provider en la cuenta AWS via CLI
>    (`aws iam create-open-id-connect-provider`). **Pre-Terraform**: si
>    falta este paso, el apply de Ola C3 (#4.5.3) explota con
>    `data "aws_iam_openid_connect_provider" ... no such entity`.
> 2. **#3.11.2** (este módulo) declara los 2 roles que **asumen** ese
>    provider (`gha-deploy` / `gha-train`) usando `var.oidc_provider_arn`.
> 3. **#3.11.5** (`consumer-iam`) declara un rol cross-repo que asume el
>    OIDC del repo consumer (`ml_serving`) — requiere que ese repo
>    haya creado su propio provider, no éste.
> 4. **#5.7** explica el flujo conceptual del token JWT y por qué el
>    `sub` del trust policy debe matchear `repo:<org>/<repo>:*`.
>
> Si el repo en GitHub cambia de nombre/org, hay que **regenerar tanto**
> el OIDC provider (#2.5) **como** los roles (este módulo): el `sub`
> hardcoded del trust no se ajusta solo.

### 3.11.1 `modules/cicd/variables.tf`

```hcl
variable "project" { type = string }
variable "github_org" { type = string }
variable "github_repo" { type = string }
variable "oidc_provider_arn" { type = string }
```

### 3.11.2 `modules/cicd/main.tf`

> **Equivalente en AWS Console — los 2 IAM Roles con trust OIDC**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_iam_role.deploy` con trust OIDC | **IAM** | `IAM > Roles > Create role`. **Trusted entity type**: **Web identity** (NO "AWS service"). **Identity provider**: `token.actions.githubusercontent.com` (el OIDC creado en Parte 2.5). **Audience**: `sts.amazonaws.com`. **GitHub organization**: `<tu-org>`. **GitHub repository**: `<tu-repo>`. **GitHub branch**: deja vacio para `*`. La Console te genera el JSON con `StringLike` sobre `sub = repo:org/repo:*`. **Permissions**: inline policy con todo lo de Terraform apply + ECR push. **Name**: `ml-training-gha-deploy`. |
> | `aws_iam_role.train` con mismo trust | **IAM** | Mismo wizard de Web identity. **Permissions**: SOLO `lambda:InvokeFunction` sobre el dispatcher + `batch:Describe/ListJobs` + `logs:GetLogEvents`. **Name**: `ml-training-gha-train`. |
>
> **Conceptualmente — el flujo OIDC paso a paso**:
> 1. GHA arranca un workflow con `permissions: id-token: write`.
> 2. GH genera un **JWT** firmado con claims `iss`, `aud=sts.amazonaws.com`, `sub=repo:org/ml_training:ref:refs/heads/main`, etc.
> 3. `aws-actions/configure-aws-credentials@v4` manda el JWT a STS (`sts:AssumeRoleWithWebIdentity`, `RoleArn=ml-training-gha-deploy`).
> 4. STS valida la firma (vía el discovery endpoint de GH) y los claims contra la **trust policy** del rol (`aud` y `sub LIKE "repo:org/ml_training:*"`).
> 5. STS devuelve credenciales temporales (~1h) con los permisos del rol.
> 6. **No hay secrets de larga duración en GitHub** — la gran ventaja vs `AWS_ACCESS_KEY_ID`/`SECRET` eternos como GH Secrets.
> - **Trust `sub = repo:org/repo:*`**: limita a tu repo; con solo `aud` (shared a nivel cuenta) cualquier repo de GitHub podría asumir el rol.
> - **`gha-deploy` es PODEROSO** (`iam:*`, `ec2:*`, `rds:*`): branch protection (#6.6) + Environments con approval (#6.5) son las únicas barreras entre `git push` y `terraform destroy`.
> - **`gha-train` es MÍNIMO**: solo invoca el dispatcher; lo peor sería submitear un job (gasto acotado por `dispatcher.py`).

```hcl
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  # Trust policy GHA-OIDC compartido entre gha-deploy y gha-train.
  # El template vive en infra/modules/_shared/assume-github-oidc.json.tftpl
  # — provider_arn / org / repo se inyectan via templatefile().
  # Mismo template usado por modules/consumer-iam (otro repo, mismo shape).
  gha_oidc_trust = templatefile("${path.module}/../_shared/assume-github-oidc.json.tftpl", {
    provider_arn = var.oidc_provider_arn
    org          = var.github_org
    repo         = var.github_repo
  })
}

# ----- Role 1: gha-deploy (CI workflows que aplican terraform + push ECR)
resource "aws_iam_role" "deploy" {
  name               = "${var.project}-gha-deploy"
  assume_role_policy = local.gha_oidc_trust
}

resource "aws_iam_role_policy" "deploy" {
  role = aws_iam_role.deploy.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Terraform: state remoto + lock nativo S3 (use_lockfile en backend "s3").
      # PutObject + DeleteObject cubren tanto el .tfstate como el .tfstate.tflock
      # que Terraform escribe junto al state. NO requiere DynamoDB.
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
        Resource = [
          "arn:aws:s3:::${var.project}-tfstate-*",
          "arn:aws:s3:::${var.project}-tfstate-*/*"
        ]
      },
      # ECR: push de las 5 imagenes (trainer + mlflow + reports + api + ui)
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage"
        ]
        Resource = "*" # ECR scoping necesita el endpoint para auth, * es estandar
      },
      # Terraform: leer/escribir resources (scope intencionalmente amplio para que
      # `terraform apply` funcione sobre TODOS los modulos. En produccion mas
      # estricta, dividir en roles plan-only + apply-only).
      #
      # BLAST RADIUS de este statement: un atacante que comprometa el OIDC
      # trust (ej. fork con write a `main`, o GitHub actions de un usuario
      # con permisos en el repo) puede:
      #   - destruir TODA la infra del proyecto (terraform destroy desde CI).
      #   - crear nuevos IAM roles (iam:*) y escalar a admin de la cuenta.
      #   - leer Secrets Manager (incluido el RDS password).
      # MITIGACIONES en uso:
      #   - trust policy con `sub = "repo:org/repo:*"` (solo este repo).
      #   - branch protection en main (#6.6) + required reviewers.
      #   - GitHub Environment "production" con manual approval (#6.5).
      # Refinable en #10 (hardening): partir en deploy-plan-only + apply
      # con CODEOWNERS, o restringir Resource por modulo via tags.
      {
        Effect = "Allow"
        Action = [
          "ec2:*", "vpc:*", "iam:*", "rds:*", "logs:*",
          "ecs:*", "elasticloadbalancing:*", "servicediscovery:*",
          "batch:*", "lambda:*", "events:*", "sns:*",
          "cloudwatch:*", "secretsmanager:*", "kms:*",
          "s3:GetBucketLocation", "s3:ListAllMyBuckets",
          "s3:CreateBucket", "s3:DeleteBucket", "s3:PutBucket*", "s3:GetBucket*",
          "ecr:*"
        ]
        Resource = "*"
      }
    ]
  })
}

# ----- Role 2: gha-train (solo invocar Lambda dispatcher) -------------
resource "aws_iam_role" "train" {
  name               = "${var.project}-gha-train"
  assume_role_policy = local.gha_oidc_trust
}

resource "aws_iam_role_policy" "train" {
  role = aws_iam_role.train.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["lambda:InvokeFunction"]
        # Patch 13.2: gha-train tambien invoca scheduler para wake/stop
        # en el workflow auto-train-on-push.yml.
        Resource = [
          "arn:aws:lambda:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:function:${var.project}-dispatcher",
          "arn:aws:lambda:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:function:${var.project}-scheduler"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["batch:DescribeJobs", "batch:ListJobs"]
        Resource = "*"
      },
      {
        # Patch 13.2: chequear estado RDS antes de wake
        Effect   = "Allow"
        Action   = ["rds:DescribeDBInstances"]
        Resource = "*"
      },
      {
        # Patch 13.2: chequear estado de los services Fargate
        Effect   = "Allow"
        Action   = ["ecs:DescribeServices"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:GetLogEvents", "logs:DescribeLogStreams"]
        Resource = "arn:aws:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/batch/${var.project}*"
      }
    ]
  })
}
```

### 3.11.3 `modules/cicd/outputs.tf`

```hcl
output "gha_deploy_role_arn" { value = aws_iam_role.deploy.arn }
output "gha_train_role_arn" { value = aws_iam_role.train.arn }
```

> **En consola AWS veras**:
> - IAM → Roles → `ml-training-gha-deploy` y `ml-training-gha-train`.
>   Ambas con trust policy que confia en el OIDC provider de #2.5 y
>   limita el `sub` a `repo:<github_org>/<github_repo>:*`.
> - IAM → Roles → `gha-deploy` → Permissions tab: inline policy con
>   ec2/iam/s3/ecr/ecs/batch/lambda/cloudwatch/logs/events/sns
>   (scope amplio para que `terraform apply` pueda crear/modificar
>   cualquier modulo). **Blast radius**: si alguien compromete el OIDC
>   trust (e.g., un fork con write a `main`), puede destruir toda la
>   infra — por eso branch protection (#6.6) es load-bearing.
> - IAM → Roles → `gha-train` → Permissions: solo `lambda:InvokeFunction`
>   sobre el dispatcher. Scope minimo intencional.
> - Estos roles son los `vars.AWS_GHA_DEPLOY_ROLE_ARN` /
>   `AWS_GHA_TRAIN_ROLE_ARN` que se setean con `gh variable set` en #6.1.

---

### 3.11.4 Apendear `module "cicd"` en `infra/envs/prod/main.tf`

Ultimo bloque. Pegar AL FINAL de `infra/envs/prod/main.tf` (despues
de `module "scheduler"` de #3.10.5):

```hcl
# -------------------------------------------------------------------------
# Capa 9: CI/CD (GHA IAM roles confiando en OIDC)
# -------------------------------------------------------------------------
module "cicd" {
  source = "../../modules/cicd"

  project           = var.project
  github_org        = var.github_org
  github_repo       = var.github_repo
  oidc_provider_arn = data.aws_iam_openid_connect_provider.github.arn
}
```

> **Checkpoint final**: con este bloque pegado, el `main.tf` esta
> **completo** (los 9 modulos + el `data` source del OIDC provider). La validación
> sintáctica (`fmt -check` + `validate`) se ejecuta en #4.2, justo antes
> del primer apply — no acá, porque entre que terminás Parte 3 y arrancás
> Parte 4 pueden pasar días.

> **Gotcha #3.11**: el modulo `cicd` solo recibe `project`, `github_org`, `github_repo` y `oidc_provider_arn`. Las vars `*_arn` (buckets, ECR, queues, job-def) que en versiones previas se pasaban como pass-through fueron **eliminadas** del modulo y de su `variables.tf` por estar sin uso — el `main.tf` de cicd construye los ARNs que necesita con su propio `data "aws_caller_identity"` / `data "aws_region"`.

---

## 3.11.5 `modules/consumer-iam/` — Rol OIDC para repo consumer EXTERNO (Patch 13.5, OPCIONAL)

> **OPCIONAL / LEGACY**: este modulo nacio cuando la API+UI vivian en un repo
> APARTE (`ml_serving`) que descargaba modelos read-only. **Ahora la API+UI son
> parte de ESTE monorepo** (Capa 4.5, #3.12) y obtienen los artifacts via el
> `task role` de su propio modulo (S3 GetObject), sin necesitar este rol
> cross-repo. Mantenelo SOLO si todavia tenes un repo serving externo que
> consume el Model Registry. Si no, podes **omitir** este modulo: borra el
> bloque `module "consumer_iam"` de `main.tf`, el output `consumer_role_arn`
> (#3.2.6) y las vars `consumer_org`/`consumer_repo` (#3.2.3/#3.2.4).

Rol IAM que un repo consumer EXTERNO (otro repo que sirve modelos) asume via
GitHub OIDC para descargar artifacts de S3 read-only. Separado de `cicd/`
porque vive con permisos distintos y trust hacia otro repo.

Las vars `consumer_org` y `consumer_repo` ya estan declaradas en
`envs/prod/variables.tf` (Patch 13.5), y el OIDC provider es el mismo
`data "aws_iam_openid_connect_provider" "github"` que usa `cicd/` (#3.11),
asi que no hay que crear nada nuevo a nivel envs.

### 3.11.5.1 `modules/consumer-iam/variables.tf`

```hcl
variable "project" { type = string }
variable "artifacts_bucket_arn" { type = string }
variable "consumer_oidc_arn" { type = string }
variable "consumer_org" { type = string }
variable "consumer_repo" { type = string }
```

### 3.11.5.2 `modules/consumer-iam/main.tf`

```hcl
# Patch 13.5: rol que el repo consumer (FastAPI/Streamlit) asume via OIDC
# para descargar artifacts (modelos) desde S3 read-only.

resource "aws_iam_role" "consumer" {
  name = "${var.project}-consumer"
  assume_role_policy = templatefile("${path.module}/../_shared/assume-github-oidc.json.tftpl", {
    provider_arn = var.consumer_oidc_arn
    org          = var.consumer_org
    repo         = var.consumer_repo
  })
}

resource "aws_iam_role_policy" "consumer" {
  role = aws_iam_role.consumer.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [var.artifacts_bucket_arn, "${var.artifacts_bucket_arn}/*"]
      }
    ]
  })
}
```

### 3.11.5.3 `modules/consumer-iam/outputs.tf`

```hcl
output "consumer_role_arn" { value = aws_iam_role.consumer.arn }
```

### 3.11.5.4 Apendear `module "consumer_iam"` en `infra/envs/prod/main.tf`

Pegar AL FINAL de `infra/envs/prod/main.tf` (despues del `module "cicd"`
de #3.11.4):

```hcl
# -------------------------------------------------------------------------
# Capa 10: Consumer IAM (Patch 13.5 — repo ml_serving consume artifacts read-only)
# -------------------------------------------------------------------------
module "consumer_iam" {
  source = "../../modules/consumer-iam"

  project              = var.project
  artifacts_bucket_arn = module.storage.artifacts_bucket_arn
  consumer_oidc_arn    = data.aws_iam_openid_connect_provider.github.arn
  consumer_org         = var.consumer_org
  consumer_repo        = var.consumer_repo
}
```

> **En consola AWS veras**:
> - IAM → Roles → `ml-training-consumer` con trust policy que confia en
>   el OIDC provider de #2.5 y limita el `sub` a
>   `repo:<consumer_org>/<consumer_repo>:*` (repo distinto al de training).
> - IAM → Roles → `ml-training-consumer` → Permissions: inline policy con
>   `s3:GetObject` + `s3:ListBucket` sobre el bucket de artifacts (scope
>   minimo: el repo consumer **solo lee** modelos, no entrena ni publica).
> - El ARN se exporta como output `consumer_role_arn` (#3.2.6) — ese
>   ARN va al repo consumer como `vars.AWS_CONSUMER_ROLE_ARN` para que
>   su workflow GHA pueda hacer `aws-actions/configure-aws-credentials`
>   contra este rol.

> **Gotcha #3.11.5**: requiere `consumer_oidc_arn` válido — si el repo consumer aún no tiene OIDC trust setup, el `plan` falla con `InvalidIdentifier`. Bootstrapear el OIDC del consumer antes de aplicar este módulo.

---

## 3.12 App stack — `modules/api` (FastAPI) + `modules/ui` (Streamlit) — Capa 4.5

Esta capa levanta los **dos servicios de producto** que viven en el monorepo
`ml_training`: la **API** (FastAPI, sirve los modelos `rnd-forest-*` y persiste
pronosticos) y la **UI** (Streamlit, dashboard gerencial que consume la API).
Ambos corren en el **mismo cluster ECS + ALB que MLflow** (no se crea infra
nueva de red ni balanceador), siguiendo el mismo patron que `reports` (#3.6):
Fargate + target group + listener rule por path.

> **Por que "Capa 4.5"**: en orden de dependencias va **entre** reports (Capa 4)
> y batch (Capa 5), porque reusa el cluster, el ALB, el RDS y el namespace de
> service discovery que crea MLflow (Capa 3). En `main.tf` los `module` blocks
> se pegan en ese orden por claridad, pero Terraform resuelve referencias por
> el grafo de dependencias, no por orden textual.

> **Decisiones de arquitectura (rationale para experto)**:
> - **API y UI comparten ALB con MLflow/reports** (1 solo balanceador para todo
>   el stack → ahorra ~$16/mes vs un ALB por servicio). El ruteo es **por path**:
>   `/api/*` y `/docs` → API; `/app/*` → UI; el resto (default) → MLflow.
> - **La API reusa el RDS de MLflow** (base `forecasts`, auto-creada al boot) en
>   vez de un RDS dedicado. A bajo trafico (showcase) una instancia hostea ambas
>   bases sin problema; ver #9.1 para el dimensionamiento (db.t4g.small).
> - **La UI NO toca AWS** (sin task role): solo hace HTTP a la API por service
>   discovery interno (`api.<project>.local:8000`). La API SI tiene task role
>   (lee artifacts de S3 al cargar modelos).
> - **App code ya existe** (`api/app/`, `ui/app/`, `src/`) del Tramo I local —
>   igual que `src/` del trainer, no se re-pega aca. Esta seccion paga **solo la
>   infra (Terraform) + los Dockerfiles** que faltaban para el deploy AWS.

### 3.12.1 `modules/api/variables.tf`

```hcl
variable "project" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "sg_api_id" { type = string }

variable "ecs_cluster_id" { type = string }
variable "alb_listener_arn" { type = string }
variable "service_discovery_namespace_id" { type = string }

variable "api_image" {
  description = "URI completa de la imagen ECR de la API (repo:tag)."
  type        = string
}

# --- MLflow / modelos ---
variable "mlflow_tracking_uri" {
  description = "URI interna del MLflow (service discovery): http://mlflow.<project>.local:5000."
  type        = string
}
variable "model_registry_prefix" {
  description = "Prefijo del registered model. Debe coincidir con el trainer."
  type        = string
  default     = "rnd-forest-"
}
variable "mlflow_preload_models" {
  description = "Precargar modelos al boot (true) o lazy (false). Lazy acota RAM."
  type        = bool
  default     = false
}

# --- Base de datos (reusa el RDS de MLflow, base `forecasts`) ---
variable "rds_address" { type = string }
variable "rds_password_secret_arn" { type = string }

# --- S3 artifacts (boto3 vía cliente MLflow) ---
variable "artifacts_bucket" { type = string }
variable "artifacts_bucket_arn" { type = string }

# --- CORS (origen del ALB para Swagger / llamadas browser) ---
variable "cors_origins" {
  description = "Origenes CORS permitidos (coma-separados)."
  type        = string
  default     = "http://localhost:8501"
}

# --- Capacidad / costo (ver analisis en GUIA) ---
# Combos validos Fargate: 1 vCPU admite 2-8 GB. La API carga modelos en RAM;
# con lazy-load (preload=false) y ~6 variedades, 1 vCPU / 2 GB sobra. Subir
# memory a 4096 si se activa preload de muchas variedades.
variable "cpu" {
  type    = number
  default = 1024 # 1 vCPU
}
variable "memory" {
  type    = number
  default = 2048 # 2 GB
}
variable "desired_count" {
  description = "Replicas. El scheduler lo maneja (0 fuera de horario)."
  type        = number
  default     = 1
}

variable "log_retention_days" { type = number }
```

### 3.12.2 `modules/api/main.tf`

La API es el patron espejo de `reports` + dos extras: **service discovery**
(para que la UI la resuelva por DNS interno) y **secret del RDS** (inyecta el
password al componer `DATABASE_URL` en runtime, sin persistirlo en la imagen).

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_cloudwatch_log_group.api` | **CloudWatch** | `Log groups > Create`. **Name**: `/ecs/ml-training/api`. Retention = 14 dias. |
> | `aws_iam_role.api_exec` + `api_task` | **IAM** | Dos roles con trust `ecs-tasks.amazonaws.com`. **exec**: adjuntar `AmazonECSTaskExecutionRolePolicy` + inline `secretsmanager:GetSecretValue` sobre el secret del RDS. **task**: inline `s3:GetObject`+`s3:ListBucket` sobre el bucket de artifacts. |
> | `aws_service_discovery_service.api` | **Cloud Map** | Dentro del namespace `ml-training.local`, `Create service` → **Name**: `api`. DNS A-record, TTL 10, routing MULTIVALUE. Resuelve `api.ml-training.local`. |
> | `aws_lb_target_group.api` | **EC2 → Target Groups** | `Create` → target type **IP**, protocol HTTP :8000, VPC la tuya. Health check path `/api/health`, matcher 200. |
> | `aws_lb_listener_rule.api_functional` + `api_docs` | **EC2 → Load Balancers → Listener :80 → Rules** | Dos reglas forward al TG api: prioridad 88 con paths `/api/health* /api/forecasts* /api/varieties* /api/history*`; prioridad 89 con `/docs /openapi.json /redoc`. |
> | `aws_ecs_task_definition.api` | **ECS → Task definitions** | Fargate, `awsvpc`, cpu 1024 / mem 2048. Container `api` (imagen ECR), port 8000, secret `RDS_PASSWORD`, env MLFLOW_TRACKING_URI/EXPERIMENT_PREFIX/CORS_ORIGINS, logs a CloudWatch. |
> | `aws_ecs_service.api` | **ECS → Services** | En el cluster `ml-training-cluster`: `Create` → launch type FARGATE, task def `ml-training-api`, desired 1, subnets privadas, SG `sg-api`, sin IP publica. Asociar al TG api + al service registry de Cloud Map. |
>
> **Por que ruteo por prefijos especificos (no `/api/*` a secas)**: MLflow es el
> default del ALB y con `--serve-artifacts` expone `/api/2.0/mlflow-artifacts/*`.
> Un `/api/*` generico le robaria esa ruta y rompe el preview de artifacts del
> MLflow UI. Por eso listamos solo los prefijos reales del FastAPI.
>
> **Por que `$RDS_PASSWORD` con UN solo `$`**: el `command` compone `DATABASE_URL`
> en runtime. Terraform solo escapa `$${`; `$RDS_PASSWORD` queda literal y lo
> expande el shell con la env var que ECS inyecta desde Secrets Manager. Usar
> `$$` rompe la auth (mismo gotcha que MLflow).

```hcl
# ============================================================================
# Modulo api — FastAPI en ECS Fargate (mismo cluster + ALB que MLflow).
#
# Sirve los modelos rnd-forest-* registrados en MLflow y persiste pronosticos
# en la base `forecasts` del RDS de MLflow (la API la auto-crea al boot).
# Patron espejo del modulo reports + service discovery + secret RDS.
# ============================================================================
data "aws_region" "current" {}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project}/api"
  retention_in_days = var.log_retention_days
}

# ── IAM ────────────────────────────────────────────────────────────────────
resource "aws_iam_role" "api_exec" {
  name               = "${var.project}-api-exec"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

resource "aws_iam_role_policy_attachment" "api_exec" {
  role       = aws_iam_role.api_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# El exec role inyecta el secret del RDS password al arrancar la task.
resource "aws_iam_role_policy" "api_exec_secret" {
  role = aws_iam_role.api_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = var.rds_password_secret_arn
    }]
  })
}

resource "aws_iam_role" "api_task" {
  name               = "${var.project}-api-task"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

# Task role: leer artifacts de S3 (boto3 vía cliente MLflow al cargar modelos).
resource "aws_iam_role_policy" "api_task_s3" {
  role = aws_iam_role.api_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [var.artifacts_bucket_arn, "${var.artifacts_bucket_arn}/*"]
    }]
  })
}

# ── Service discovery (la UI llama a api.<project>.local:8000) ───────────────
resource "aws_service_discovery_service" "api" {
  name = "api"
  dns_config {
    namespace_id = var.service_discovery_namespace_id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }
}

# ── ALB target group + reglas ────────────────────────────────────────────────
resource "aws_lb_target_group" "api" {
  name        = "${var.project}-tg-api"
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  health_check {
    path                = "/api/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
    matcher             = "200"
  }
  deregistration_delay = 30
}

# Ruteo por PREFIJOS ESPECIFICOS (no `/api/*` a secas): MLflow es el default del
# ALB y con --serve-artifacts expone /api/2.0/mlflow-artifacts/*. Un `/api/*`
# generico robaria esa ruta y rompe el preview de artifacts del MLflow UI.
# Listamos solo los prefijos reales del FastAPI.
resource "aws_lb_listener_rule" "api_functional" {
  listener_arn = var.alb_listener_arn
  priority     = 88

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
  condition {
    path_pattern {
      values = ["/api/health*", "/api/forecasts*", "/api/varieties*", "/api/history*"]
    }
  }
}

# Swagger / OpenAPI publico (showcase). Mantiene la doc accesible en /docs.
resource "aws_lb_listener_rule" "api_docs" {
  listener_arn = var.alb_listener_arn
  priority     = 89

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
  condition {
    path_pattern { values = ["/docs", "/openapi.json", "/redoc"] }
  }
}

# ── ECS task definition + service ────────────────────────────────────────────
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.cpu)
  memory                   = tostring(var.memory)
  execution_role_arn       = aws_iam_role.api_exec.arn
  task_role_arn            = aws_iam_role.api_task.arn

  container_definitions = jsonencode([
    {
      name         = "api"
      image        = var.api_image
      essential    = true
      portMappings = [{ containerPort = 8000, protocol = "tcp" }]
      # Componemos DATABASE_URL en runtime para inyectar el password del RDS
      # (secret) sin persistirlo. Single `$` a proposito (igual que MLflow):
      # Terraform solo escapa `$${`; `$RDS_PASSWORD` queda literal y lo expande
      # el shell con la env var inyectada desde Secrets Manager.
      command = [
        "sh", "-c",
        "export DATABASE_URL=postgresql://mlflow:$RDS_PASSWORD@${var.rds_address}:5432/forecasts; exec uvicorn app.main:app --host 0.0.0.0 --port 8000"
      ]
      secrets = [{
        name      = "RDS_PASSWORD"
        valueFrom = var.rds_password_secret_arn
      }]
      environment = [
        { name = "MLFLOW_TRACKING_URI", value = var.mlflow_tracking_uri },
        { name = "EXPERIMENT_PREFIX", value = var.model_registry_prefix },
        { name = "MLFLOW_PRELOAD_MODELS", value = tostring(var.mlflow_preload_models) },
        { name = "CORS_ORIGINS", value = var.cors_origins },
        { name = "AWS_DEFAULT_REGION", value = data.aws_region.current.region }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = data.aws_region.current.region
          awslogs-stream-prefix = "api"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -fsS http://localhost:8000/api/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 40
      }
    }
  ])
}

resource "aws_ecs_service" "api" {
  name            = "api"
  cluster         = var.ecs_cluster_id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
  propagate_tags  = "SERVICE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.sg_api_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  service_registries {
    registry_arn = aws_service_discovery_service.api.arn
  }

  # El scheduler maneja desired_count (0 fuera de horario) -> ignore drift.
  lifecycle {
    ignore_changes = [desired_count]
  }
}
```

### 3.12.3 `modules/api/outputs.tf`

```hcl
output "service_name" {
  description = "Nombre del servicio ECS (para el scheduler on/off)."
  value       = aws_ecs_service.api.name
}
output "target_group_arn" { value = aws_lb_target_group.api.arn }
output "internal_url" {
  description = "URL interna (service discovery) que usa la UI."
  value       = "http://api.${var.project}.local:8000"
}
```

### 3.12.4 `api/Dockerfile` (contexto de build = RAÍZ del repo)

La imagen de la API necesita el paquete `src/` raiz para des-picklear los
modelos de MLflow (los transformers `LagFeatureTransformer`, `FeatureGenerator`,
etc. se serializan con rutas `src.step_03_features.*`). Por eso el **contexto de
build es la raiz** (`-f api/Dockerfile .`), no `api/`. Si ya hiciste el Tramo I
local, este archivo ya existe; se incluye aca para que la guia AWS sea standalone.

```dockerfile
# syntax=docker/dockerfile:1.7
# ============================================================================
# Imagen de la API (FastAPI) — servicio `api` del monorepo ml_training.
#
# IMPORTANTE: el contexto de build es la RAÍZ del repo (no api/), porque la
# imagen necesita el paquete `src/` raíz para des-picklear los modelos de
# MLflow (LagFeatureTransformer, FeatureGenerator, target_transform, ...).
# Una única fuente de verdad: el mismo `src/` que entrena el trainer.
#   docker build -f api/Dockerfile -t ml-training-api .
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder: instala dependencias en un venv aislado
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Toolchain mínimo para compilar wheels nativos (lightgbm/xgboost/asyncpg).
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY api/requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime: imagen final mínima
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ARG GIT_SHA=unknown
ARG VERSION=dev
LABEL org.opencontainers.image.title="ml-training-api" \
      org.opencontainers.image.description="FastAPI de pronosticos (sirve modelos rnd-forest-* de MLflow)" \
      org.opencontainers.image.source="https://github.com/abantodca/ml_training" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# libgomp1 = runtime de OpenMP (lightgbm/xgboost lo necesitan en inferencia).
# tini propaga SIGTERM correctamente cuando ECS/compose detienen la task.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libgomp1 curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 appuser \
    && useradd  --system --uid 1001 --gid appuser --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Código de la API + `src/` raíz (única fuente de verdad para unpickle).
# Orden de COPY: de mejor cache (cambia poco) a peor cache (cambia más).
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser api/app ./app

USER appuser
STOPSIGNAL SIGTERM

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.12.5 `modules/ui/variables.tf`

```hcl
variable "project" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "sg_ui_id" { type = string }

variable "ecs_cluster_id" { type = string }
variable "alb_listener_arn" { type = string }

variable "ui_image" {
  description = "URI completa de la imagen ECR de la UI (repo:tag)."
  type        = string
}

variable "api_internal_url" {
  description = "URL interna de la API (service discovery) que consume la UI."
  type        = string
}

variable "base_url_path" {
  description = "Sub-path del ALB donde sirve Streamlit (STREAMLIT_SERVER_BASE_URL_PATH)."
  type        = string
  default     = "app"
}

# --- Capacidad / costo: Streamlit es liviano ---
variable "cpu" {
  type    = number
  default = 512 # 0.5 vCPU
}
variable "memory" {
  type    = number
  default = 1024 # 1 GB
}
variable "desired_count" {
  type    = number
  default = 1
}

variable "log_retention_days" { type = number }
```

### 3.12.6 `modules/ui/main.tf`

La UI es el patron mas simple del stack: Fargate + target group + una listener
rule en `/app/*`. **No tiene task role** (no toca AWS; solo HTTP a la API por
service discovery). El sub-path `/app` se setea via env nativa de Streamlit
(`STREAMLIT_SERVER_BASE_URL_PATH`), sin rebuild.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_cloudwatch_log_group.ui` | **CloudWatch** | `Log groups > Create`. **Name**: `/ecs/ml-training/ui`. Retention 14 dias. |
> | `aws_iam_role.ui_exec` | **IAM** | Un solo rol (exec) con trust `ecs-tasks` + `AmazonECSTaskExecutionRolePolicy`. **Sin task role** (la UI no accede a AWS). |
> | `aws_lb_target_group.ui` | **EC2 → Target Groups** | target type IP, HTTP :8501. Health check path `/app/_stcore/health` (con el base-path), matcher 200. |
> | `aws_lb_listener_rule.ui` | **EC2 → Listener :80 → Rules** | prioridad 70, forward al TG ui, paths `/app` y `/app/*`. |
> | `aws_ecs_task_definition.ui` | **ECS → Task definitions** | Fargate, cpu 512 / mem 1024. Container `ui` (imagen ECR), port 8501, env `API_URL` (= service discovery de la API) + `STREAMLIT_SERVER_BASE_URL_PATH=app`. |
> | `aws_ecs_service.ui` | **ECS → Services** | cluster `ml-training-cluster`, FARGATE, desired 1, subnets privadas, SG `sg-ui`, sin IP publica, asociado al TG ui. |
>
> **Por que health check en `/app/_stcore/health`**: con base-path activo,
> Streamlit expone TODO (incluido su endpoint de salud) bajo el prefijo. Si
> apuntaras el health check a `/_stcore/health` (sin `/app`), el TG marcaria
> la task unhealthy para siempre y ECS la reciclaria en loop.

```hcl
# ============================================================================
# Modulo ui — Streamlit en ECS Fargate (mismo cluster + ALB que MLflow).
# Sirve detras del ALB en /app/* (STREAMLIT_SERVER_BASE_URL_PATH=app) y consume
# la API por service discovery interno. Patron espejo del modulo reports.
# ============================================================================
data "aws_region" "current" {}

resource "aws_cloudwatch_log_group" "ui" {
  name              = "/ecs/${var.project}/ui"
  retention_in_days = var.log_retention_days
}

resource "aws_iam_role" "ui_exec" {
  name               = "${var.project}-ui-exec"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

resource "aws_iam_role_policy_attachment" "ui_exec" {
  role       = aws_iam_role.ui_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
# Sin task role: la UI no accede a AWS (solo HTTP a la API).

# ── ALB target group + regla /app/* ─────────────────────────────────────────
resource "aws_lb_target_group" "ui" {
  name        = "${var.project}-tg-ui"
  port        = 8501
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  health_check {
    # Con base-path, Streamlit expone el health bajo el prefijo.
    path                = "/${var.base_url_path}/_stcore/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
    matcher             = "200"
  }
  deregistration_delay = 30
}

resource "aws_lb_listener_rule" "ui" {
  listener_arn = var.alb_listener_arn
  priority     = 70

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ui.arn
  }
  condition {
    path_pattern { values = ["/${var.base_url_path}", "/${var.base_url_path}/*"] }
  }
}

# ── ECS task definition + service ────────────────────────────────────────────
resource "aws_ecs_task_definition" "ui" {
  family                   = "${var.project}-ui"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.cpu)
  memory                   = tostring(var.memory)
  execution_role_arn       = aws_iam_role.ui_exec.arn

  container_definitions = jsonencode([
    {
      name         = "ui"
      image        = var.ui_image
      essential    = true
      portMappings = [{ containerPort = 8501, protocol = "tcp" }]
      environment = [
        { name = "API_URL", value = var.api_internal_url },
        { name = "STREAMLIT_SERVER_BASE_URL_PATH", value = var.base_url_path },
        { name = "LOG_LEVEL", value = "info" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ui.name
          awslogs-region        = data.aws_region.current.region
          awslogs-stream-prefix = "ui"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -fsS http://localhost:8501/${var.base_url_path}/_stcore/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }
  ])
}

resource "aws_ecs_service" "ui" {
  name            = "ui"
  cluster         = var.ecs_cluster_id
  task_definition = aws_ecs_task_definition.ui.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
  propagate_tags  = "SERVICE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.sg_ui_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ui.arn
    container_name   = "ui"
    container_port   = 8501
  }

  lifecycle {
    ignore_changes = [desired_count]
  }
}
```

### 3.12.7 `modules/ui/outputs.tf`

```hcl
output "service_name" {
  description = "Nombre del servicio ECS (para el scheduler on/off)."
  value       = aws_ecs_service.ui.name
}
output "target_group_arn" { value = aws_lb_target_group.ui.arn }
output "app_path" {
  description = "Sub-path publico del ALB donde vive la UI."
  value       = "/${var.base_url_path}/"
}
```

### 3.12.8 `ui/Dockerfile` (contexto de build = `ui/`)

A diferencia de la API, la UI **no** necesita `src/` ni nada de la raiz: su
contexto de build es la carpeta `ui/` (`-f ui/Dockerfile ui`). Si ya hiciste el
Tramo I local, este archivo ya existe.

```dockerfile
# syntax=docker/dockerfile:1.7
# ============================================================================
# Imagen de la UI (Streamlit) — servicio `ui` del monorepo ml_training.
# Contexto de build = carpeta ui/.  docker build -f ui/Dockerfile -t ml-training-ui ui
#
# Ruteo: en local sirve en la raíz (:8501). En prod, detrás del ALB en /app/*,
# se setea STREAMLIT_SERVER_BASE_URL_PATH=app (env nativa de Streamlit) en la
# task de ECS — no requiere rebuild.
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ARG GIT_SHA=unknown
ARG VERSION=dev
LABEL org.opencontainers.image.title="ml-training-ui" \
      org.opencontainers.image.description="Streamlit dashboard de pronosticos (consume la API)" \
      org.opencontainers.image.source="https://github.com/abantodca/ml_training" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 appuser \
    && useradd  --system --uid 1001 --gid appuser --create-home appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser . .

USER appuser
STOPSIGNAL SIGTERM

EXPOSE 8501

# Healthcheck sensible al base-path: /_stcore/health en local, /app/_stcore/health en prod.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -fsS "http://localhost:8501/${STREAMLIT_SERVER_BASE_URL_PATH:+${STREAMLIT_SERVER_BASE_URL_PATH}/}_stcore/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
```

### 3.12.9 Apendear `module "api"` + `module "ui"` en `infra/envs/prod/main.tf`

Pegar AL FINAL de `infra/envs/prod/main.tf`. Logicamente es **Capa 4.5** (va
entre reports y batch); como el orden textual no afecta a Terraform, podes
pegarlo al final y queda igual de valido.

```hcl
# -------------------------------------------------------------------------
# Capa 4.5: App stack — API (FastAPI) + UI (Streamlit)
# Mismo cluster + ALB que MLflow. La API reusa el RDS de MLflow (base
# `forecasts`, auto-creada al boot) y carga modelos rnd-forest-* via S3.
# -------------------------------------------------------------------------
module "api" {
  source = "../../modules/api"

  project                        = var.project
  vpc_id                         = module.network.vpc_id
  private_subnet_ids             = module.network.private_subnet_ids
  sg_api_id                      = module.network.sg_api_id
  ecs_cluster_id                 = module.mlflow.cluster_id
  alb_listener_arn               = module.mlflow.alb_listener_arn
  service_discovery_namespace_id = module.mlflow.service_discovery_namespace_id
  api_image                      = "${module.storage.ecr_api_url}:${var.api_image_tag}"
  # MLflow interno via service discovery (no pasa por el ALB).
  mlflow_tracking_uri     = "http://mlflow.${var.project}.local:5000"
  model_registry_prefix   = var.model_registry_prefix
  mlflow_preload_models   = var.api_preload_models
  rds_address             = module.mlflow.rds_address
  rds_password_secret_arn = module.mlflow.rds_password_secret_arn
  artifacts_bucket        = module.storage.artifacts_bucket
  artifacts_bucket_arn    = module.storage.artifacts_bucket_arn
  cors_origins            = "http://${module.mlflow.alb_dns}"
  cpu                     = var.api_cpu
  memory                  = var.api_memory
  log_retention_days      = var.log_retention_days
}

module "ui" {
  source = "../../modules/ui"

  project            = var.project
  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  sg_ui_id           = module.network.sg_ui_id
  ecs_cluster_id     = module.mlflow.cluster_id
  alb_listener_arn   = module.mlflow.alb_listener_arn
  ui_image           = "${module.storage.ecr_ui_url}:${var.ui_image_tag}"
  api_internal_url   = module.api.internal_url
  cpu                = var.ui_cpu
  memory             = var.ui_memory
  log_retention_days = var.log_retention_days
}
```

> **Checkpoint**: consume outputs de `module.network` (sg_api/sg_ui),
> `module.mlflow` (cluster, listener, service discovery, rds_address,
> rds_password_secret_arn) y `module.storage` (ecr_api/ui_url, artifacts).
> Si `validate` falla con "Unsupported attribute", revisa que ya pegaste los
> outputs nuevos de mlflow (#3.5.3) y los SGs api/ui (#3.3.2.e / #3.3.3).

> **Gotcha #3.12**: la API necesita que MLflow este **encendido** (RDS available
> + service discovery resolviendo `mlflow.<project>.local`) y que existan modelos
> `rnd-forest-*` registrados, o el `/api/health` devolvera degraded. En un
> stand-up de cero, primero entrena al menos 1 variedad (#4.6) — la API la
> servira en cuanto el run quede registrado.

---

> **Cierre Parte 3.**
>
> Estado actual: el repo tiene escritos `infra/envs/prod/` (6 archivos),
> 12 módulos en `infra/modules/` (incluye `api` + `ui` de la Capa 4.5),
> 3 archivos Python en `infra/lambdas/`, 3 archivos en `docker/reports/`
> y los 2 Dockerfiles de app (`api/Dockerfile`, `ui/Dockerfile`). Todo
> lockeado al contrato real del trainer (env vars S3, command `--varieties X --tuning Y`, variedades
> válidas, custom metric MAPE con dimension `variety`). La verificación
> sintáctica (`fmt -check` + `validate`) se ejecuta en #4.2, justo antes
> del primer apply — donde de verdad cobra sentido.

---

# Parte 4 — Apply incremental + smoke test

> **Filosofia de la Parte 4**: aplicar Terraform en 3 olas en vez de un
> `terraform apply` monolitico. Esto te da puntos de rollback claros y
> evita el dolor de "el apply de 25 min fallo en el ultimo recurso y
> ahora no se en que estado quedo todo".
>
> Las olas:
>
> | Ola | Modulos | Tiempo | Sirve para |
> |---|---|---|---|
> | A | `storage` (S3 + ECR) | ~1 min | Tener ECR donde pushear las imagenes |
> | B | build + push de las 5 imagenes (trainer + mlflow + reports + api + ui) | 5-15 min (10-30 min en cache frio) | Imagenes disponibles para Fargate / Batch |
> | C | resto (`network`, `mlflow`, `reports`, `batch`, `monitoring`, `lambdas`, `scheduler`, `cicd`, `consumer_iam`) | ~15-20 min | Infra operativa, ALB con health checks 200 |
>
> Si haces apply monolitico (Ola A+C juntas), Fargate intenta arrancar
> MLflow con la imagen que todavia no esta pusheada → bucle de retries
> hasta que el deployment timeout te mata.

## 4.1 Setup Task (orquestador local)

[Task](https://taskfile.dev) es un task runner single-binary (Go, ~10 MB)
que orquesta Terraform + Docker + Lambda + Batch desde una sola lista
descubrible (`task --list`). Si conocés Make, lo leés en 5 min.

Cada task es **idempotente y composable**: si falla en el paso 5/10, lo
corres de nuevo y empieza donde quedo (Terraform mismo es idempotente;
los hash de `sources:` evitan re-buildear imagenes ya construidas).

### 4.1.1 Verificar / instalar Task

```bash
task --version
# Esperado: 3.34+ (necesario para `prompt:` en tasks destructivos)
```

Si falta (ya cubierto en Capítulo 3.1; recordatorio aqui):

```bash
# Windows (WSL Ubuntu) y Linux: mismo instalador
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/bin
export PATH="$HOME/bin:$PATH"   # agregar a ~/.bashrc para persistir

# macOS
brew install go-task
```

### 4.1.2 Estructura final

Despues de seguir sección 4.1.3 a sección 4.1.8, tu proyecto va a tener:

```
Taskfile.yml                    # raiz: tasks LOCALES (Docker) + atajos AWS high-level
tasks/
├── infra.yml                   # infra:*    terraform wrapper + bootstrap
├── ecr.yml                     # ecr:*      build + push 5 imagenes
├── batch.yml                   # batch:*    submit (via Lambda dispatcher) + smoke + status + cancel
├── ops.yml                     # ops:*      lifecycle cluster + MLflow registry (Dia 2)
├── local.yml                   # local:*    sandbox dev (ensure-buckets, bucket-name)
└── lib/
    ├── batch_wait.sh           # polling Batch jobs
    ├── wake.sh                 # wake idempotente del cluster
    ├── mlflow_uri.sh           # resolver ALB DNS
    └── nuke.sh                 # helpers para destroy/nuke
```

**Por que 5 archivos `tasks/*.yml` (4 AWS + `local`) + helpers en `lib/`**:

- **Namespacing**: cada `includes:` prefija con `<nombre>:`. `task build`
  (Docker local) no choca con `task ecr:build` (AWS).
- **Atajos high-level en root** (`deploy`, `smoke`, `wake`, `sleep`,
  `status`, `teardown`, `rebuild`, `destroy`, `nuke`): el flujo de runbook
  tipico esta a un comando. Detras llaman al namespace que corresponda.
  Orden = ciclo de vida: stand-up → day-2 ops → cleanup/recovery.
- **Bash largo extraido a `tasks/lib/*.sh`**: polling, wake, mlflow URI
  resolution y nuke viven como funciones sourceables. El YAML queda
  declarativo; la logica imperativa, testeable con `bash -n`.

### 4.1.3 `tasks/infra.yml` — Terraform wrapper + bootstrap

```yaml
# =============================================================================
# tasks/infra.yml  -  Terraform wrapper + bootstrap del backend
# =============================================================================
# Incluido por Taskfile.yml raiz con namespace "infra:".
#
# USO TIPICO:
#   task infra:bootstrap                              UNA VEZ por cuenta+region
#   task infra:bootstrap-oidc                         UNA VEZ (rol GHA)
#   task infra:plan [TARGET=module.X]                 ver cambios
#   task infra:apply [TARGET=module.X]                aplicar (parcial o full)
#   task infra:output                                 outputs (alb_dns, ecr_urls, ...)
#   task infra:validate                               fmt -check + validate (pre-commit)
#   task infra:destroy                                DESTRUCTIVO: todo
#   task infra:destroy-target TARGET=module.X         DESTRUCTIVO: parcial
# =============================================================================

version: "3"

vars:
  TF_DIR:  '{{.TF_DIR | default "infra/envs/prod"}}'
  # Ultimos 7 chars del account ID (sufijo de buckets para evitar colisiones).
  # Misma fuente que tasks/local.yml -> coherencia local/prod en nombres de bucket.
  # Resuelto una vez por invocacion del include.
  SUFFIX:
    sh: bash scripts/aws-suffix.sh

tasks:

  # ═══ Bootstrap (one-shot, idempotente) ══════════════════════════════════════

  bootstrap:
    desc: "Backend Terraform (S3 tfstate + SLRs; lock nativo S3). UNA VEZ por cuenta+region"
    cmds:
      - bash infra/bootstrap.sh

  bootstrap-oidc:
    desc: "Rol IAM para GitHub Actions via OIDC. UNA VEZ"
    cmds:
      - bash infra/bootstrap-oidc.sh

  # ═══ Init (interno, dep de plan/apply/destroy) ══════════════════════════════

  _init:
    internal: true
    cmds:
      # use_lockfile=true => locking nativo S3 (reemplaza el deprecado
      # `dynamodb_table`). El lock vive como `<key>.tflock` en el mismo bucket
      # de tfstate. Requiere Terraform >= 1.10.
      - terraform -chdir={{.TF_DIR}} init
        -backend-config=bucket={{.PROJECT}}-tfstate-{{.SUFFIX}}
        -backend-config=key=envs/prod/terraform.tfstate
        -backend-config=region={{.REGION}}
        -backend-config=use_lockfile=true
        -reconfigure

  # ═══ Plan / Apply / Destroy ═════════════════════════════════════════════════

  plan:
    desc: "terraform plan. Var opcional: TARGET=module.X"
    deps: [_init]
    cmds:
      - terraform -chdir={{.TF_DIR}} plan {{if .TARGET}}-target={{.TARGET}}{{end}}

  apply:
    desc: "terraform apply -auto-approve. Var opcional: TARGET=module.X (apply parcial por oleadas)"
    deps: [_init]
    cmds:
      - terraform -chdir={{.TF_DIR}} apply {{if .TARGET}}-target={{.TARGET}}{{end}} -auto-approve

  destroy:
    desc: "DESTRUCTIVO: terraform destroy completo. Considerar `task ops:teardown` antes (preserva storage)"
    prompt: "Esto borrara TODA la infra de envs/prod (incluso S3 + ECR). Continuar?"
    deps: [_init]
    cmds:
      - terraform -chdir={{.TF_DIR}} destroy -auto-approve

  destroy-target:
    desc: "terraform destroy parcial. Vars: TARGET=module.X (REQ)"
    prompt: "Destruir {{.TARGET}}? Asegurate que no tenga dependencias activas"
    requires:
      vars: [TARGET]
    deps: [_init]
    cmds:
      - terraform -chdir={{.TF_DIR}} destroy -target={{.TARGET}} -auto-approve

  # ═══ Inspeccion ═════════════════════════════════════════════════════════════

  output:
    desc: "Mostrar outputs de envs/prod (alb_dns, ecr_urls, rds_endpoint, ...)"
    cmds:
      - terraform -chdir={{.TF_DIR}} output

  output-raw:
    desc: "Mostrar UN output crudo (para scripts). Var: NAME=alb_dns (REQ)"
    silent: true
    requires:
      vars: [NAME]
    cmds:
      - terraform -chdir={{.TF_DIR}} output -raw {{.NAME}}

  urls:
    desc: "Imprime las URLs publicas de PROD (derivadas del ALB DNS, que varia por deploy)."
    silent: true
    cmds:
      - |
        ALB=$(terraform -chdir={{.TF_DIR}} output -raw alb_dns 2>/dev/null)
        if [ -z "$ALB" ]; then
          echo "  (sin outputs: la infra no esta aplicada todavia -> corre 'task deploy')"
          exit 0
        fi
        cat <<EOF

        ════════════════════════════════════════════════════════════════════
         Servicios PROD (ALB: $ALB)
           UI (dashboard gerencial)   http://$ALB/app/
           API (Swagger)              http://$ALB/docs
           MLflow runs                http://$ALB/
           MLflow Model Registry      http://$ALB/#/models
           Reports (campeon HTML)     http://$ALB/reports/
           Artifacts                  http://$ALB/artifacts/
        ────────────────────────────────────────────────────────────────────
         Todo cuelga del :80 del ALB por paths (a diferencia de local, que usa
         un puerto por servicio). El DNS del ALB cambia si recreas el LB.
        ════════════════════════════════════════════════════════════════════
        EOF

  validate:
    desc: "terraform fmt -check + validate (init + providers; usable cold antes del primer apply)"
    deps: [_init]
    cmds:
      - terraform -chdir={{.TF_DIR}} fmt -check -recursive
      - terraform -chdir={{.TF_DIR}} validate

  # ═══ Recovery ═══════════════════════════════════════════════════════════════

  force-unlock:
    desc: "Liberar state lock huerfano. Var: LOCK_ID=<id> (REQ)"
    requires:
      vars: [LOCK_ID]
    deps: [_init]
    cmds:
      - terraform -chdir={{.TF_DIR}} force-unlock -force {{.LOCK_ID}}
```

`_init` es `internal: true` (no aparece en `task --list`) y se dispara via `deps:` cuando hace falta. `SUFFIX` se resuelve dinamicamente con `aws sts get-caller-identity` — si cambias de cuenta AWS, el backend bucket name cambia con vos.

### 4.1.4 `tasks/ecr.yml` — build + push 5 imagenes

Cinco imagenes: `trainer` + `mlflow` + `reports` (training stack) + `api` + `ui`
(App stack, Capa 4.5). El switch `IMG` resuelve nombre, Dockerfile, contexto y
tag por imagen. **Ojo con el contexto de build**: `api` usa la **raiz** del repo
(necesita `src/` para des-picklear modelos), `ui` usa la carpeta `ui/`; el resto
usa la raiz.

```yaml
# =============================================================================
# tasks/ecr.yml  -  Build + push de las 5 imagenes a ECR
# =============================================================================
# Incluido por Taskfile.yml raiz con namespace "ecr:".
#
# USO TIPICO:
#   task ecr:build-all                                build + push de las 5
#   task ecr:build IMG=trainer                        UNA imagen, tag default
#   task ecr:build IMG=api TAG=v1.2.3                 UNA imagen, tag custom
#   task ecr:list                                     listar tags en ECR
#
# IMG = trainer | mlflow | reports | api | ui
# =============================================================================

version: "3"

vars:
  TAG_TRAINER: '{{.TAG_TRAINER | default "latest"}}'
  TAG_MLFLOW:  '{{.TAG_MLFLOW  | default "v3.12.0"}}'
  TAG_REPORTS: '{{.TAG_REPORTS | default "stable"}}'
  TAG_API:     '{{.TAG_API     | default "latest"}}'
  TAG_UI:      '{{.TAG_UI      | default "latest"}}'

tasks:

  # ═══ Login (token 12h, run: once) ═══════════════════════════════════════════

  login:
    desc: "docker login a ECR. Idempotente, token valido 12h"
    # run: once: si varias tasks dependen de login en una misma corrida,
    # solo se ejecuta una vez.
    run: once
    vars:
      ACCOUNT:
        sh: aws sts get-caller-identity --query Account --output text
    cmds:
      - aws ecr get-login-password --region {{.REGION}}
        | docker login --username AWS --password-stdin {{.ACCOUNT}}.dkr.ecr.{{.REGION}}.amazonaws.com

  # ═══ Build + push UNA imagen ════════════════════════════════════════════════

  build:
    desc: "Build + push UNA imagen. Vars: IMG=trainer|mlflow|reports|api|ui (REQ), TAG=<override>"
    requires:
      vars: [IMG]
    deps: [login]
    vars:
      ACCOUNT:
        sh: aws sts get-caller-identity --query Account --output text
      REGISTRY: '{{.ACCOUNT}}.dkr.ecr.{{.REGION}}.amazonaws.com'
      GIT_SHA:
        sh: git rev-parse --short=12 HEAD 2>/dev/null || echo unknown
      BUILD_DATE:
        sh: date -u +%Y-%m-%dT%H:%M:%SZ
      # Tabla IMG -> (image_name, dockerfile, context, default_tag). Cualquier IMG
      # fuera del set valido cae en el branch ERROR validado abajo.
      # CONTEXT: api usa contexto=raiz (necesita src/); ui usa contexto=ui/.
      IMAGE_NAME: '{{if eq .IMG "trainer"}}{{.PROJECT}}{{else if eq .IMG "mlflow"}}{{.PROJECT}}-mlflow{{else if eq .IMG "reports"}}{{.PROJECT}}-reports{{else if eq .IMG "api"}}{{.PROJECT}}-api{{else if eq .IMG "ui"}}{{.PROJECT}}-ui{{else}}ERROR{{end}}'
      DOCKERFILE: '{{if eq .IMG "trainer"}}Dockerfile{{else if eq .IMG "mlflow"}}docker/mlflow/Dockerfile{{else if eq .IMG "reports"}}docker/reports/Dockerfile{{else if eq .IMG "api"}}api/Dockerfile{{else if eq .IMG "ui"}}ui/Dockerfile{{else}}ERROR{{end}}'
      CONTEXT: '{{if eq .IMG "ui"}}ui{{else}}.{{end}}'
      RESOLVED_TAG: '{{if eq .IMG "trainer"}}{{.TAG | default .TAG_TRAINER}}{{else if eq .IMG "mlflow"}}{{.TAG | default .TAG_MLFLOW}}{{else if eq .IMG "reports"}}{{.TAG | default .TAG_REPORTS}}{{else if eq .IMG "api"}}{{.TAG | default .TAG_API}}{{else if eq .IMG "ui"}}{{.TAG | default .TAG_UI}}{{else}}ERROR{{end}}'
    cmds:
      - 'test "{{.IMAGE_NAME}}" != "ERROR" || { echo "ERROR IMG debe ser trainer|mlflow|reports|api|ui (recibido {{.IMG}})"; exit 1; }'
      - 'echo ">>> Build {{.IMAGE_NAME}}:{{.RESOLVED_TAG}}  (sha-{{.GIT_SHA}})"'
      # BUILD_DATE va como --label (metadata del config final), no como --build-arg:
      # asi no invalida la cache de capas ni vuelve no-reproducible el commit.
      - docker build
        --build-arg GIT_SHA={{.GIT_SHA}}
        --build-arg VERSION={{.RESOLVED_TAG}}
        --label org.opencontainers.image.created={{.BUILD_DATE}}
        -t {{.REGISTRY}}/{{.IMAGE_NAME}}:{{.RESOLVED_TAG}}
        -t {{.REGISTRY}}/{{.IMAGE_NAME}}:sha-{{.GIT_SHA}}
        -f {{.DOCKERFILE}} {{.CONTEXT}}
      # Push idempotente. En repos IMMUTABLE (p.ej. mlflow) solo pushea si el
      # tag NO existe -> re-correr el mismo commit es no-op en vez de error.
      # En repos MUTABLE siempre pushea (sobrescribe latest/stable).
      - |
        MUT=$(aws ecr describe-repositories --repository-names {{.IMAGE_NAME}} \
                --region {{.REGION}} --query 'repositories[0].imageTagMutability' --output text)
        push_tag() {
          tag="$1"
          if [ "$MUT" = "IMMUTABLE" ] && \
             aws ecr describe-images --repository-name {{.IMAGE_NAME}} \
                 --image-ids imageTag="$tag" --region {{.REGION}} >/dev/null 2>&1; then
            echo ">>> $tag ya existe en {{.IMAGE_NAME}} (IMMUTABLE) -- skip push"
          else
            docker push {{.REGISTRY}}/{{.IMAGE_NAME}}:"$tag"
          fi
        }
        push_tag "{{.RESOLVED_TAG}}"
        push_tag "sha-{{.GIT_SHA}}"

  # ═══ Build + push de las 3 ══════════════════════════════════════════════════

  build-all:
    desc: "Build + push de las 5 imagenes (trainer + mlflow + reports + api + ui)"
    deps: [login]
    vars:
      # Single source of truth del tag trainer = terraform.tfvars. Sin esto,
      # build-all pushea `latest` mientras la job-def queda pineada a otra tag
      # (p.ej. v0.2.0) -> CannotPullImageManifestError en el primer smoke.
      # Si el grep no encuentra nada, queda "" y `build` cae a TAG_TRAINER.
      TRAINER_TFVARS_TAG:
        sh: sed -nE 's/^[[:space:]]*trainer_image_tag[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' infra/envs/prod/terraform.tfvars 2>/dev/null | head -1
    cmds:
      - task: build
        vars: { IMG: trainer, TAG: '{{.TRAINER_TFVARS_TAG}}' }
      - task: build
        vars: { IMG: mlflow }
      - task: build
        vars: { IMG: reports }
      - task: build
        vars: { IMG: api }
      - task: build
        vars: { IMG: ui }

  # ═══ Inspeccion ═════════════════════════════════════════════════════════════

  list:
    desc: "Listar las 5 imagenes con tag default presente en cada repo ECR"
    silent: true
    cmds:
      - |
        for spec in "{{.PROJECT}}:{{.TAG_TRAINER}}" "{{.PROJECT}}-mlflow:{{.TAG_MLFLOW}}" "{{.PROJECT}}-reports:{{.TAG_REPORTS}}" "{{.PROJECT}}-api:{{.TAG_API}}" "{{.PROJECT}}-ui:{{.TAG_UI}}"; do
          repo="${spec%:*}"; tag="${spec##*:}"
          echo "=== $repo ($tag) ==="
          aws ecr list-images --repository-name "$repo" \
            --query "imageIds[?imageTag=='$tag']" --output table 2>/dev/null || true
          echo ""
        done
```

Cada imagen se pushea con **dos tags**: el movil (`latest`/`v3.12.0`/`stable`) que CI/CD persigue, y `sha-<git-sha>` que da rollback determinista (`task ecr:build IMG=trainer TAG=sha-<commit-anterior>`).

### 4.1.5 `tasks/batch.yml` — submit via Lambda dispatcher

```yaml
# =============================================================================
# tasks/batch.yml  -  AWS Batch (training jobs en la nube)
# =============================================================================
# Incluido por Taskfile.yml raiz con namespace "batch:".
#
# El submit pasa por el Lambda dispatcher (valida variety + hydrate de S3
# antes de encolar). Si necesitas bypass, llama `aws batch submit-job` directo.
#
# USO TIPICO:
#   task batch:train VARIETIES=POP                    una variedad, espera
#   task batch:train VARIETIES=POP,VENTURA            multiples en serie
#   task batch:train VARIETIES=POP TUNING=smoke       sanity check (~1 min)
#   task batch:train VARIETIES=POP WAIT=false         fire-and-forget
#   task batch:smoke                                  atajo: POP + smoke
#   task batch:eda   VARIETIES=POP                    EDA exploratorio (opcional, on-demand)
#   task batch:status                                 jobs activos en queues
#   task batch:cancel JOB_ID=<id>                     terminar un job
# =============================================================================

version: "3"

vars:
  DISPATCHER_FN: '{{.DISPATCHER_FN | default (printf "%s-dispatcher" .PROJECT)}}'
  JOBDEF:        '{{.JOBDEF        | default (printf "%s-trainer" .PROJECT)}}'
  QUEUE_SPOT:    '{{.QUEUE_SPOT    | default (printf "%s-job-queue-spot"     .PROJECT)}}'
  QUEUE_OD:      '{{.QUEUE_OD      | default (printf "%s-job-queue-ondemand" .PROJECT)}}'
  TUNING:        '{{.TUNING        | default "prod_xl"}}'
  WAIT:          '{{.WAIT          | default "true"}}'

tasks:

  train:
    desc: "Submit via Lambda dispatcher. Vars: VARIETIES=POP[,VENTURA] (REQ), TUNING, WAIT"
    requires:
      vars: [VARIETIES]
    cmds:
      - |
        set -e
        source tasks/lib/batch_wait.sh
        # Preflight: aborta en ~2s si la job-def apunta a una imagen ausente en
        # ECR, en vez de encolar y esperar ~3 min al CannotPullImageManifestError.
        assert_jobdef_image "{{.JOBDEF}}" "{{.REGION}}"
        PAYLOAD=$(jq -nc --arg v "{{.VARIETIES}}" --arg t "{{.TUNING}}" \
          '{varieties: $v, tuning: $t}')
        aws lambda invoke \
          --function-name "{{.DISPATCHER_FN}}" \
          --cli-binary-format raw-in-base64-out \
          --payload "$PAYLOAD" \
          /tmp/dispatcher-out.json \
          --query 'StatusCode' --output text
        cat /tmp/dispatcher-out.json
        JOB_ID=$(jq -r '.body.jobId // (.body|fromjson|.jobId)' /tmp/dispatcher-out.json 2>/dev/null \
                 || jq -r '.jobId' /tmp/dispatcher-out.json)
        echo ">>> Submitted  job=$JOB_ID  tuning={{.TUNING}}  varieties={{.VARIETIES}}"
        [ "{{.WAIT}}" != "true" ] && exit 0
        wait_job "$JOB_ID" "{{.VARIETIES}}"

  eda:
    desc: "EDA exploratorio en Batch (opcional, standalone, on-demand). Vars: VARIETIES=POP[,VENTURA] (REQ), WAIT"
    requires:
      vars: [VARIETIES]
    cmds:
      - |
        set -e
        source tasks/lib/batch_wait.sh
        assert_jobdef_image "{{.JOBDEF}}" "{{.REGION}}"
        # mode=eda -> el dispatcher arma command=["--eda","--varieties",...] (sin tuning).
        PAYLOAD=$(jq -nc --arg v "{{.VARIETIES}}" '{varieties: $v, mode: "eda"}')
        aws lambda invoke \
          --function-name "{{.DISPATCHER_FN}}" \
          --cli-binary-format raw-in-base64-out \
          --payload "$PAYLOAD" \
          /tmp/dispatcher-out.json \
          --query 'StatusCode' --output text
        cat /tmp/dispatcher-out.json
        JOB_ID=$(jq -r '.body.jobId // (.body|fromjson|.jobId)' /tmp/dispatcher-out.json 2>/dev/null \
                 || jq -r '.jobId' /tmp/dispatcher-out.json)
        echo ">>> Submitted EDA  job=$JOB_ID  varieties={{.VARIETIES}}"
        [ "{{.WAIT}}" != "true" ] && exit 0
        wait_job "$JOB_ID" "{{.VARIETIES}}"

  smoke:
    desc: "Sanity check end-to-end (~1 min). Equivalente a train VARIETIES=POP TUNING=smoke"
    cmds:
      - task: train
        vars: { VARIETIES: POP, TUNING: smoke }

  status:
    desc: "Jobs activos (SUBMITTED/PENDING/RUNNABLE/STARTING/RUNNING) en ambas queues"
    silent: true
    cmds:
      - |
        for queue in "{{.QUEUE_SPOT}}" "{{.QUEUE_OD}}"; do
          echo "=== $queue ==="
          for s in SUBMITTED PENDING RUNNABLE STARTING RUNNING; do
            aws batch list-jobs --job-queue "$queue" --job-status $s \
              --query 'jobSummaryList[].[jobId,jobName,status,createdAt]' --output table 2>/dev/null || true
          done
          echo ""
        done

  cancel:
    desc: "Terminar job RUNNING/PENDING. Vars: JOB_ID=<id> (REQ), REASON=<texto>"
    requires:
      vars: [JOB_ID]
    cmds:
      - aws batch terminate-job --job-id "{{.JOB_ID}}" --reason "{{.REASON | default \"cancelled via task\"}}"
```

Una sola via de submit (el Lambda dispatcher) — valida que la variety exista en `data/training/DB-HISTORICA.xlsx` antes de encolar y maneja hydrate desde S3. El path raw con `aws batch submit-job` queda disponible si alguna vez lo necesitas.

### 4.1.6 `tasks/ops.yml` — lifecycle cluster + MLflow registry (Dia 2)

```yaml
# =============================================================================
# tasks/ops.yml  -  Operaciones Dia 2: lifecycle del cluster + MLflow registry
# =============================================================================
# Incluido por Taskfile.yml raiz con namespace "ops:".
#
# Modulos VOLATILES (teardown los destruye, ~10-15 min para recrear):
#   scheduler, lambdas, monitoring, batch, reports, mlflow, cicd, consumer_iam
# Modulos PERMANENTES (NO se tocan en teardown):
#   network (VPC + NAT), storage (S3 + ECR), backend state
#
# USO TIPICO:
#   task ops:status                                   estado RDS + ECS + Batch
#   task ops:state-drift                              buckets S3 en AWS vs tfstate (pre-deploy)
#   task ops:up                                       encender idempotente (espera healthy)
#   task ops:down [COOLDOWN=600]                      apagar (drena Batch + stop RDS)
#   task ops:teardown                                 destroy volatiles
#   task ops:rebuild                                  re-apply + up
#   task ops:promote MODEL_NAME=rnd-forest-POP VERSION=3 [MAX_MAPE=20]
#   task ops:registry-list MODEL_NAME=rnd-forest-POP
# =============================================================================

version: "3"

vars:
  TF_DIR:       '{{.TF_DIR       | default "infra/envs/prod"}}'
  SCHEDULER_FN: '{{.SCHEDULER_FN | default (printf "%s-scheduler" .PROJECT)}}'
  QUEUE_SPOT:   '{{.QUEUE_SPOT   | default (printf "%s-job-queue-spot"     .PROJECT)}}'
  QUEUE_OD:     '{{.QUEUE_OD     | default (printf "%s-job-queue-ondemand" .PROJECT)}}'
  MAX_MAPE:     '{{.MAX_MAPE     | default "20"}}'
  # Orden reverso de apply: importante para destroy con dependencias.
  VOLATILE_MODULES: "module.scheduler module.lambdas module.monitoring module.batch module.reports module.api module.ui module.mlflow module.cicd module.consumer_iam"

tasks:

  # ═══ Estado ═════════════════════════════════════════════════════════════════

  status:
    desc: "Estado del cluster: RDS + ECS services + Batch jobs activos"
    silent: true
    cmds:
      - 'echo "=== RDS ==="'
      - aws rds describe-db-instances --db-instance-identifier {{.PROJECT}}-mlflow
        --query 'DBInstances[0].[DBInstanceStatus,DBInstanceClass,Endpoint.Address]'
        --output table 2>/dev/null || echo "  (RDS no existe o no accesible)"
      - 'echo ""'
      - 'echo "=== ECS Services ==="'
      - aws ecs describe-services --cluster {{.PROJECT}}-cluster
        --services mlflow reports api ui
        --query 'services[].[serviceName,desiredCount,runningCount,pendingCount]'
        --output table 2>/dev/null || echo "  (ECS no existe o servicios no creados)"
      - 'echo ""'
      - 'echo "=== Batch jobs activos ==="'
      - task: _batch-jobs
        vars: { ABORT_IF_RUNNING: "false" }

  # ═══ Helper unico: estado de Batch + assert opcional ════════════════════════

  _batch-jobs:
    internal: true
    silent: true
    vars:
      ABORT_IF_RUNNING: '{{.ABORT_IF_RUNNING | default "false"}}'
    cmds:
      - |
        total=0
        running=0
        for q in {{.QUEUE_SPOT}} {{.QUEUE_OD}}; do
          for s in SUBMITTED PENDING RUNNABLE STARTING RUNNING; do
            n=$(aws batch list-jobs --job-queue "$q" --job-status $s --query 'length(jobSummaryList)' --output text 2>/dev/null || echo 0)
            [ "$n" -gt 0 ] && echo "  queue=$q  status=$s  count=$n"
            total=$((total + n))
            [ "$s" = "RUNNING" ] && running=$((running + n))
          done
        done
        echo "  TOTAL activos $total"
        if [ "{{.ABORT_IF_RUNNING}}" = "true" ] && [ "$running" -gt 0 ]; then
          echo "ERROR $running job(s) RUNNING. Cancelar primero:"
          echo "       task batch:status              (ver detalle)"
          echo "       task batch:cancel JOB_ID=<id>  (cancelar)"
          exit 1
        fi

  # ═══ Drift detection (state vs realidad AWS) ═══════════════════════════════

  state-drift:
    desc: "Detectar drift de buckets S3: existen en AWS pero no en tfstate (causa BucketAlreadyExists en apply)"
    silent: true
    cmds:
      - |
        set -e
        echo ">>> Buckets S3 con prefijo {{.PROJECT}} en AWS:"
        aws_side=$(aws s3 ls | awk '/{{.PROJECT}}-/ {print $3}' | sort)
        echo "$aws_side" | sed 's/^/  /'
        echo ""
        echo ">>> Buckets S3 en tfstate ({{.TF_DIR}}):"
        tf_side=$(terraform -chdir={{.TF_DIR}} state list 2>/dev/null \
          | grep -E 'aws_s3_bucket\.[^.]+$' \
          | awk -F. '{print $NF}' \
          | while read r; do terraform -chdir={{.TF_DIR}} state show "module.storage.aws_s3_bucket.$r" 2>/dev/null \
              | awk -F\" '/^[[:space:]]+bucket[[:space:]]+=/ {print $2; exit}'; done \
          | sort)
        echo "$tf_side" | sed 's/^/  /'
        echo ""
        only_aws=$(comm -23 <(echo "$aws_side") <(echo "$tf_side"))
        # tfstate bucket vive fuera del state por diseno (lo crea infra/bootstrap.sh)
        only_aws=$(echo "$only_aws" | grep -v -- '-tfstate-' || true)
        if [ -n "$only_aws" ]; then
          echo "DRIFT: existen en AWS pero faltan en tfstate (causara BucketAlreadyExists):"
          echo "$only_aws" | sed 's/^/  - /'
          echo ""
          echo "Fix sugerido (ajustar el nombre del resource al que corresponda en modules/storage/main.tf):"
          echo "  terraform -chdir={{.TF_DIR}} import module.storage.aws_s3_bucket.<resource> <bucket-name>"
          exit 1
        fi
        echo "OK no drift detectado en buckets S3."

  # ═══ Lifecycle: up (idempotente) / down (con cooldown opcional) ═════════════

  up:
    desc: "Encender stack (idempotente: skip si MLflow ya UP, sino invoca scheduler.start y espera RDS+ALB healthy)"
    silent: true
    cmds:
      - |
        set -e
        source tasks/lib/wake.sh
        PROJECT={{.PROJECT}} TF_DIR={{.TF_DIR}} SCHEDULER_FN={{.SCHEDULER_FN}} wake_cluster

  down:
    desc: "Apagar stack (aborta si Batch RUNNING). Opcional COOLDOWN=N (segundos antes de apagar)"
    vars:
      COOLDOWN: '{{.COOLDOWN | default "0"}}'
    cmds:
      - 'echo ">>> Pre-check Batch jobs activos"'
      - task: _batch-jobs
        vars: { ABORT_IF_RUNNING: "true" }
      - |
        if [ "{{.COOLDOWN}}" -gt 0 ]; then
          echo ">>> Cool-down {{.COOLDOWN}}s antes de apagar..."
          sleep {{.COOLDOWN}}
        fi
      - 'echo ">>> Invocando scheduler.stop..."'
      # Tolerante a Lambda inexistente: en flujos de destroy / infra parcial,
      # el scheduler ya puede estar destruido. Skipear es seguro porque
      # `terraform destroy` igual tumba RDS/ECS a continuacion.
      - |
        if aws lambda get-function --function-name {{.SCHEDULER_FN}} >/dev/null 2>&1; then
          aws lambda invoke --function-name {{.SCHEDULER_FN}} \
            --payload '{"action":"stop"}' \
            --cli-binary-format raw-in-base64-out \
            /tmp/scheduler-stop.json
          cat /tmp/scheduler-stop.json && echo ""
        else
          echo "  Lambda {{.SCHEDULER_FN}} no existe -> skip (probablemente ya destruido)"
        fi

  # ═══ Teardown / Rebuild ═════════════════════════════════════════════════════

  teardown:
    desc: "Down + terraform destroy de modulos volatiles. Preserva storage + network"
    prompt: "Destruira los modulos volatiles. Storage (S3+ECR) y network (VPC) quedan. Continuar?"
    cmds:
      - task: down
      - 'echo ">>> Destroy modulos volatiles (orden reverso de apply)..."'
      - |
        for mod in {{.VOLATILE_MODULES}}; do
          echo ">>> terraform destroy -target=$mod"
          terraform -chdir={{.TF_DIR}} destroy -target=$mod -auto-approve || {
            echo "FAIL destroy de $mod fallo. Revisar manualmente."
            exit 1
          }
        done
      - 'echo "OK teardown completo. Para volver: task ops:rebuild"'

  rebuild:
    desc: "Re-apply de modulos volatiles + up"
    cmds:
      - 'echo ">>> Apply completo (modulos volatiles se re-crean, resto no-op)..."'
      - task: ":infra:apply"
      - task: up

  # ═══ MLflow Model Registry ═════════════════════════════════════════════════

  registry-list:
    desc: "Listar versiones de un modelo. Var: MODEL_NAME=rnd-forest-POP (REQ)"
    requires:
      vars: [MODEL_NAME]
    cmds:
      - |
        set -e
        source tasks/lib/mlflow_uri.sh
        URI=$(mlflow_uri {{.TF_DIR}})
        curl -s "$URI/api/2.0/mlflow/registered-models/get?name={{.MODEL_NAME}}" \
          | jq '.registered_model.latest_versions[] | {version, current_stage, run_id, creation_timestamp}'

  promote:
    desc: "Promover a Production con gate MAPE + A/B. Vars: MODEL_NAME (REQ), VERSION=N (REQ), MAX_MAPE=20"
    requires:
      vars: [MODEL_NAME, VERSION]
    cmds:
      - python scripts/promote_model.py {{.MODEL_NAME}} {{.VERSION}} --max-mape {{.MAX_MAPE}} --tf-dir {{.TF_DIR}}
```

**Decisiones clave** (resumidas — el codigo es la fuente):

- `ops:up` es **idempotente**: pre-check `/health` y solo invoca `scheduler.start` si esta DOWN. Lo usan tanto el operador manual como el flujo CI auto-train (el `wake.sh` escribe el estado previo a `/tmp/wake-status` para que el workflow decida si tiene que apagar al final).
- `ops:down` con `COOLDOWN=N` (default 0) cubre el "down ahora" manual y el "espera N seg y apaga" del CI post-train con una sola task.
- `ops:teardown` preserva `module.network` y `module.storage`. El NAT Gateway en network cuesta $32/mes encendido pero su create tarda 5+ min — si vas a teardown >1 vez/mes, conservarlo es net-positivo. Para hibernar largo: `task infra:destroy-target TARGET=module.network`.
- `ops:promote` delega en `scripts/promote_model.py` (Python con `MlflowClient`) — 3 gates: MAPE absoluto, A/B contra Production actual, transition con `archive_existing_versions=True`.

### 4.1.7 Helpers `tasks/lib/*.sh`

Bash compartido extraido del YAML para que sea testeable con `bash -n` y reusable:

**`tasks/lib/batch_wait.sh`** — preflight (`assert_jobdef_image`) + polling (`wait_job`):

```bash
# Helper bash compartido por tasks/batch.yml (polling de Batch jobs).
# Sourceado, no ejecutado. Requiere awscli configurado.

# assert_jobdef_image <job_definition_name> [region]
# Preflight: falla rapido si la imagen de la job-def ACTIVE no existe en ECR,
# en vez de esperar ~3 min a que Batch reporte
# `CannotPullImageManifestError: manifest unknown`.
# Causa raiz tipica: se bumpeo trainer_image_tag en terraform.tfvars + apply,
# pero nunca se corrio `task ecr:build IMG=trainer TAG=<tag>` (o al reves).
assert_jobdef_image() {
  local jobdef="$1" region="${2:-us-east-1}"
  local image repo tag
  image=$(aws batch describe-job-definitions --job-definition-name "$jobdef" \
            --status ACTIVE --region "$region" \
            --query 'reverse(sort_by(jobDefinitions,&revision))[0].containerProperties.image' \
            --output text 2>/dev/null)
  if [ -z "$image" ] || [ "$image" = "None" ]; then
    echo "  preflight: job-def '$jobdef' sin imagen resoluble -- skip check" >&2
    return 0
  fi
  repo="${image##*/}"; repo="${repo%%:*}"   # <registry>/<repo>:<tag> -> <repo>
  tag="${image##*:}"
  if aws ecr describe-images --repository-name "$repo" \
       --image-ids imageTag="$tag" --region "$region" >/dev/null 2>&1; then
    return 0
  fi
  cat >&2 <<EOF
ERROR la job-def '$jobdef' apunta a una imagen que NO existe en ECR:
        $image
      Construi + pushea esa tag ANTES de submitear:
        task ecr:build IMG=trainer TAG=$tag
      (o ajusta trainer_image_tag en infra/envs/prod/terraform.tfvars
       y corre  task infra:apply TARGET=module.batch)
EOF
  return 1
}

# wait_job <job_id> <label>
# Polling cada 30s hasta SUCCEEDED (return 0) o FAILED (return 1).
wait_job() {
  local job_id="$1" label="$2"
  while :; do
    local status
    status=$(aws batch describe-jobs --jobs "$job_id" --query 'jobs[0].status' --output text)
    echo "  $(date +%H:%M:%S)  $label  $status"
    case "$status" in
      SUCCEEDED) return 0 ;;
      FAILED)
        local reason
        reason=$(aws batch describe-jobs --jobs "$job_id" --query 'jobs[0].statusReason' --output text)
        echo "FAIL $label  reason=$reason"
        return 1
        ;;
      *) sleep 30 ;;
    esac
  done
}
```

**`tasks/lib/mlflow_uri.sh`** — resolver el ALB:

```bash
# Resolver la URI de MLflow (ALB) desde:
#   1. env MLFLOW_ALB_DNS (usado por GHA via vars.MLFLOW_ALB_DNS)
#   2. terraform output -raw alb_dns en TF_DIR (uso local)
# Sourceado, no ejecutado.

mlflow_uri() {
  local tf_dir="${1:-infra/envs/prod}"
  if [ -n "${MLFLOW_ALB_DNS:-}" ]; then
    echo "http://$MLFLOW_ALB_DNS"
    return 0
  fi
  local alb
  alb=$(terraform -chdir="$tf_dir" output -raw alb_dns 2>/dev/null)
  if [ -z "$alb" ]; then
    echo "ERROR no se pudo leer alb_dns (ni env MLFLOW_ALB_DNS ni terraform output)" >&2
    return 1
  fi
  echo "http://$alb"
}
```

Prioridad `MLFLOW_ALB_DNS` antes de `terraform output`: GitHub Actions inyecta la var desde `vars.MLFLOW_ALB_DNS` sin necesidad de correr `terraform init` (ahorra ~30 s y un permiso IAM al state). En local, el fallback funciona sin tocar nada.

**`tasks/lib/wake.sh`** — wake idempotente del cluster (47 lineas; extraido entero del YAML viejo para que el `ops:up` quede declarativo):

```bash
# Wake idempotente del cluster MLflow.
#   - Si MLflow ya responde /health -> noop, escribe true a STATUS_FILE.
#   - Si no, invoca scheduler.start, espera RDS available, espera ALB 200.
# Sourceado, no ejecutado.
#
# Vars de entorno:
#   PROJECT       (req)  nombre base del stack (ej. ml-training)
#   TF_DIR        (def)  infra/envs/prod
#   MLFLOW_ALB_DNS (opt) si esta seteada, salta terraform output
#   STATUS_FILE   (opt)  default /tmp/wake-status (true|false segun pre-check)

source "$(dirname "${BASH_SOURCE[0]}")/mlflow_uri.sh"

wake_cluster() {
  local project="${PROJECT:?PROJECT requerido}"
  local tf_dir="${TF_DIR:-infra/envs/prod}"
  local scheduler_fn="${SCHEDULER_FN:-${project}-scheduler}"
  local status_file="${STATUS_FILE:-/tmp/wake-status}"

  local uri
  uri=$(mlflow_uri "$tf_dir") || return 1
  local alb="${uri#http://}"

  echo ">>> Pre-check MLflow en $uri/health"
  if curl -fs -o /dev/null --max-time 5 "$uri/health"; then
    echo "WAS_UP=true (skip wake)"
    echo "true" > "$status_file"
    return 0
  fi
  echo "WAS_UP=false. Invocando $scheduler_fn (action=start)..."
  echo "false" > "$status_file"
  aws lambda invoke \
    --function-name "$scheduler_fn" \
    --cli-binary-format raw-in-base64-out \
    --payload '{"action":"start"}' \
    /tmp/wake-start.out >/dev/null
  cat /tmp/wake-start.out && echo ""

  echo ">>> Esperando RDS available (24x30s = 12 min max)..."
  local status=""
  for _ in $(seq 1 24); do
    status=$(aws rds describe-db-instances \
               --db-instance-identifier "${project}-mlflow" \
               --query 'DBInstances[0].DBInstanceStatus' --output text 2>/dev/null || echo "missing")
    echo "  $(date +%H:%M:%S)  RDS=$status"
    [ "$status" = "available" ] && break
    sleep 30
  done
  if [ "$status" != "available" ]; then
    echo "::error::RDS no available tras 12 min (estado=$status)"
    return 1
  fi

  echo ">>> Esperando MLflow ALB 200 (30x10s = 5 min max)..."
  local code="000"
  for _ in $(seq 1 30); do
    code=$(curl -fs -o /dev/null -w "%{http_code}" --max-time 5 "$uri/health" || echo "000")
    echo "  $(date +%H:%M:%S)  GET /health -> $code"
    [ "$code" = "200" ] && break
    sleep 10
  done
  if [ "$code" != "200" ]; then
    echo "::error::MLflow no respondio 200 tras 5 min (code=$code)"
    return 1
  fi
  echo "OK wake completo. MLflow UP en $uri"
}
```

**`tasks/lib/nuke.sh`** — helpers para `destroy` / `nuke`: vaciar buckets versionados, purgar repos ECR y borrar el OIDC provider. Tres funciones independientes que `Taskfile.yml` sourcea en los targets destructivos.

> 📂 **Pegar este bloque en**: `tasks/lib/nuke.sh`

```bash
# Helpers para destroy/nuke: vaciar buckets versionados, borrar repos ECR,
# borrar el OIDC provider. Sourceados, no ejecutados.

# empty_bucket <bucket> [delete]
#   Vacia versiones + delete markers. Si delete=true, ademas borra el bucket.
empty_bucket() {
  local bucket="$1" delete="${2:-false}"
  if ! aws s3api head-bucket --bucket "$bucket" 2>/dev/null; then
    echo "  $bucket no existe, skip"; return 0
  fi
  echo "  Vaciando $bucket (versiones + delete markers)..."
  aws s3api delete-objects --bucket "$bucket" \
    --delete "$(aws s3api list-object-versions --bucket "$bucket" \
      --query '{Objects: [Versions[].{Key:Key,VersionId:VersionId},DeleteMarkers[].{Key:Key,VersionId:VersionId}][]}' \
      --max-items 1000)" 2>/dev/null || echo "  (bucket ya vacio)"
  if [ "$delete" = "true" ]; then
    echo "  Borrando bucket $bucket..."
    aws s3 rb "s3://$bucket"
  fi
}

# purge_ecr <repo>
#   Borra TODAS las imagenes de un repo ECR (no borra el repo).
purge_ecr() {
  local repo="$1"
  if ! aws ecr describe-repositories --repository-names "$repo" >/dev/null 2>&1; then
    echo "  $repo no existe, skip"; return 0
  fi
  local ids
  ids=$(aws ecr list-images --repository-name "$repo" --query 'imageIds[*]' --output json)
  if [ "$ids" = "[]" ]; then
    echo "  $repo vacio"; return 0
  fi
  echo "  Borrando todas las imagenes de $repo..."
  aws ecr batch-delete-image --repository-name "$repo" --image-ids "$ids" >/dev/null
}

# delete_oidc
#   Borra el OIDC provider de GitHub Actions de la cuenta.
delete_oidc() {
  local arn
  arn=$(aws iam list-open-id-connect-providers \
    --query 'OpenIDConnectProviderList[?contains(Arn, `token.actions.githubusercontent.com`)].Arn' \
    --output text)
  if [ -z "$arn" ]; then
    echo "  OIDC provider no existe, skip"; return 0
  fi
  echo "  Borrando OIDC provider: $arn"
  aws iam delete-open-id-connect-provider --open-id-connect-provider-arn "$arn"
}
```

### 4.1.8 Root `Taskfile.yml` — `includes` + atajos high-level

Editar la raiz para:

1. Sumar los 4 namespaces AWS (`infra`, `ecr`, `batch`, `ops`) al `local:` que ya existia en `includes:`; reemplazar el comentario "Tramo I: solo importamos…" por uno generico.
2. Mantener las vars compartidas (`PROJECT`, `REGION`) — **quitar** `SUFFIX` del top-level (se vuelve lazy, ver callout).
3. Definir los atajos high-level (`deploy`, `smoke`, `wake`, `sleep`, `status`, `teardown`, `rebuild`, `destroy`, `nuke`) — encadenan tasks de los namespaces.
4. Expandir el `default` de Tramo I sección 4.6 con las secciones AWS (Bootstrap / Primer stand-up EN ORDEN / Operacion dia-2 / Teardown-Recovery-Destroy) — para que `task` (sin args) sea el menu unico de entrada local + AWS, mostrando la secuencia numerada 0→7 del stand-up (mapa del camino 4.1.10).

> **SUFFIX lazy (no top-level)** — la version Tramo I tenia `SUFFIX` en `vars:`
> raiz. Si se mantuviera asi, Task evaluaria `bash scripts/aws-suffix.sh` al
> **cargar** el Taskfile (incluso para `task` solo o `task --list`), lo que
> exige credenciales AWS configuradas y agrega latencia a cada invocacion.
> Solo `destroy` y `nuke` lo necesitan → se computa **por-tarea** (lazy) en
> esos dos bloques. Mismo patron que `tasks/local.yml` sección 4.6 documenta.

```yaml
version: "3"

dotenv: [ ".env" ]

# Includes namespaced por etapa. Cada tasks/X.yml documenta su uso en el header.
includes:
  infra:
    taskfile: ./tasks/infra.yml
    vars: { PROJECT: '{{.PROJECT}}', REGION: '{{.REGION}}' }
  ecr:
    taskfile: ./tasks/ecr.yml
    vars: { PROJECT: '{{.PROJECT}}', REGION: '{{.REGION}}' }
  batch:
    taskfile: ./tasks/batch.yml
    vars: { PROJECT: '{{.PROJECT}}', REGION: '{{.REGION}}' }
  ops:
    taskfile: ./tasks/ops.yml
    vars: { PROJECT: '{{.PROJECT}}', REGION: '{{.REGION}}' }
  local:
    taskfile: ./tasks/local.yml
    vars: { PROJECT: '{{.PROJECT}}', REGION: '{{.REGION}}' }

vars:
  # TUNING se define a nivel-tarea (default "prod_xl"), NO como var global:
  # un default global pisaria el del include batch: (ver tasks/batch.yml).
  VARIETIES: '{{.VARIETIES | default "POP"}}'
  PARALLEL:  '{{.PARALLEL  | default "1"}}'
  PROJECT:   '{{.PROJECT   | default "ml-training"}}'
  REGION:    '{{.AWS_DEFAULT_REGION | default "us-east-1"}}'
  HOST_UID:  { sh: id -u }
  HOST_GID:  { sh: id -g }
  # --user fija tu uid/gid del host en el container: los bind-mounts (./data,
  # ./reports, ./logs) salen con tu ownership, no como mluser (uid 1001) — evita
  # PermissionError. MPLCONFIGDIR=/tmp porque con --user el $HOME de la imagen no
  # es escribible por tu uid (matplotlib en eda/HTML). Mismo patron en `train`.
  DC_PY: docker compose run --rm --no-deps --user "{{.HOST_UID}}:{{.HOST_GID}}" -e MPLCONFIGDIR=/tmp --entrypoint python trainer
  # SUFFIX (sufijo del Account ID para bucket/repo names) NO se define aqui a
  # proposito: si fuera top-level, Task lo evaluaria al cargar y dispararia
  # `aws sts get-caller-identity` incluso para `task` o `task --list`. Solo
  # `destroy` y `nuke` lo necesitan -> se computa por-tarea (lazy).

tasks:

  # ... build / data:split / eda / train / down / _up / _ensure_dirs / _print_urls ...
  # (sin cambios, vienen de Tramo I sección 4.6)
  # `default` se EXPANDE — ver bloque dedicado mas abajo.

  lint:
    desc: "ruff check src/ main.py scripts/ (config en pyproject.toml)"
    cmds:
      - ruff check src/ main.py scripts/

  # ═══ Atajos high-level del stack AWS ════════════════════════════════════════

  deploy:
    desc: "AWS: stand-up completo (storage -> 5 imagenes -> resto) + imprime URLs"
    cmds:
      # Pre-check: si los buckets *-data-* / *-artifacts-* existen en AWS pero
      # no en tfstate (tipico tras nuke parcial o re-bootstrap), `apply
      # module.storage` falla con BucketAlreadyExists. Fail-fast con instrucciones.
      - task: ops:state-drift
      - 'echo ">>> Oleada A: apply module.storage (S3 + ECR)..."'
      - task: infra:apply
        vars: { TARGET: module.storage }
      - 'echo ">>> Oleada B: build + push 5 imagenes..."'
      - task: ecr:build-all
      - 'echo ">>> Oleada C: apply resto (network, mlflow, batch, ...)..."'
      - task: infra:apply
      - 'echo ""'
      - 'echo "Deploy completo. URLs publicas:"'
      - task: infra:urls

  smoke:
    desc: "AWS: deploy + smoke test (POP, tuning=smoke, ~1 min)"
    cmds:
      - task: deploy
      - task: batch:smoke

  wake:
    desc: "AWS: encender stack (idempotente)"
    cmds:
      - task: ops:up

  sleep:
    desc: "AWS: apagar stack"
    cmds:
      - task: ops:down

  status:
    desc: "AWS: outputs de Terraform + estado del cluster + URLs publicas"
    cmds:
      - 'echo "=== Terraform outputs ==="'
      - task: infra:output
      - 'echo ""'
      - 'echo "=== Cluster ==="'
      - task: ops:status
      - task: infra:urls

  urls:
    desc: "AWS: imprime las URLs publicas de produccion (derivadas del ALB)"
    cmds:
      - task: infra:urls

  teardown:
    desc: "AWS: down + destroy volatiles (preserva storage + network)"
    cmds:
      - task: ops:teardown

  rebuild:
    desc: "AWS: re-apply tras teardown + up"
    cmds:
      - task: ops:rebuild

  destroy:
    desc: "AWS DESTRUCTIVO: drena Batch, vacia S3/ECR, terraform destroy total"
    prompt: "Destruira envs/prod (S3 + ECR + RDS + ...). Irreversible. Continuar?"
    vars:
      SUFFIX:
        sh: bash scripts/aws-suffix.sh
    cmds:
      - task: ops:down
      - 'echo ">>> Vaciando buckets S3 versionados + purgando ECR..."'
      - |
        set -e
        source tasks/lib/nuke.sh
        empty_bucket "{{.PROJECT}}-data-{{.SUFFIX}}"
        empty_bucket "{{.PROJECT}}-artifacts-{{.SUFFIX}}"
        purge_ecr    "{{.PROJECT}}"
        purge_ecr    "{{.PROJECT}}-mlflow"
        purge_ecr    "{{.PROJECT}}-reports"
        purge_ecr    "{{.PROJECT}}-api"
        purge_ecr    "{{.PROJECT}}-ui"
      - 'echo ">>> terraform destroy total..."'
      - task: infra:destroy

  nuke:
    desc: "AWS IRREVERSIBLE: destroy + tfstate bucket + OIDC provider"
    prompt: "NUKE COMPLETO: borra state remoto + OIDC. Despues necesitas re-bootstrap. Continuar?"
    vars:
      SUFFIX:
        sh: bash scripts/aws-suffix.sh
    cmds:
      - task: destroy
      - 'echo ">>> Borrando bucket tfstate + OIDC..."'
      - |
        set -e
        source tasks/lib/nuke.sh
        # use_lockfile guarda el lock como objeto en el mismo bucket -> se va
        # con `empty_bucket` (no hay tabla DynamoDB que limpiar).
        empty_bucket "{{.PROJECT}}-tfstate-{{.SUFFIX}}" true
        delete_oidc
      - 'echo ""'
      - 'echo "NUKE COMPLETO. Para volver: task infra:bootstrap + task infra:bootstrap-oidc + task deploy"'
```

#### 4.1.8.1 Expandir el `default` (menu unico local + AWS)

El `default` de Tramo I sección 4.6 lista solo el pipeline local. En Tramo II se expande con cuatro secciones AWS para que `task` (sin args) sea **el unico comando** que el usuario nuevo necesita conocer:

```yaml
  default:
    desc: "Menu de comandos: local (docker compose) + AWS (Terraform/Batch/MLflow)"
    silent: true
    cmds:
      - |
        cat <<'EOF'

        ml_training — predice productividad de cosecha (KG/JR_H)
        ═════════════════════════════════════════════════════════

        ▸ LOCAL  (docker compose: postgres[mlflow+forecasts] + mlflow + reports + api + ui + trainer)
            task build              1a vez o tras cambiar codigo/Dockerfile (build + up stack completo)
            task up                 levanta db + mlflow + reports + api + ui (sin rebuild)
            task data:split         genera data/training/DB-HISTORICA.xlsx
            task data:upload        (opcional) sube el Excel acumulado a S3 (hydrate paritario con Batch)
            task eda                (opcional) EDA estadistico standalone
            task train              entrena + registra modelo + genera HTML en reports/
            task down               apaga servicios

        ▸ AWS — Bootstrap  (UNA VEZ por cuenta+region, antes del stand-up)
            task infra:bootstrap          backend tfstate (lock nativo S3)
            task infra:bootstrap-oidc     rol IAM para GitHub Actions
            task local:ensure-buckets     (opcional) buckets data + artifacts dev

        ▸ AWS — Primer stand-up  (correr EN ORDEN; ~45 min, ~10 de atencion activa)
            0  source scripts/prod.env       -> carga DATA_BUCKET, ACCOUNT_SUFFIX...
            1  task infra:validate           -> fmt + validate HCL (~10s)
            2  task infra:apply TARGET=module.storage
                                             -> Ola A: 2 S3 + 3 ECR vacios (~1m)
                                             si falla BucketAlreadyExists -> `task ops:state-drift`
                                             muestra el `terraform import` exacto (recovery
                                             tras nuke parcial o re-bootstrap del tfstate)
            3  aws s3 cp data/BD_HISTORICO_ACUMULADO.xlsx s3://$DATA_BUCKET/
                                             -> sube el Excel al bucket data
            4  task ecr:build-all            -> Ola B: build+push 5 imagenes (~12m)
            5  task infra:apply              -> Ola C: red+RDS+ALB+Fargate(mlflow/reports/api/ui)+Batch+lambda (~18m)
            6  task batch:smoke              -> 1 job Batch POP smoke (~12m)
            7  open http://$ALB/app/  +  http://$ALB/docs  -> UI + API arriba = fin del stand-up
            atajos:  task deploy = pasos 2+4+5   |   task smoke = deploy + paso 6

        ▸ AWS — Operacion dia-2
            task status             outputs terraform + estado del cluster + URLs
            task urls               URLs publicas de prod (UI/API/MLflow/registry/reports via ALB)
            task wake / task sleep  encender / apagar stack (idempotente)
            task batch:train VARIETIES=POP             entrenamiento en AWS Batch
            task batch:eda   VARIETIES=POP             (opcional) EDA exploratorio on-demand
            task ops:promote MODEL_NAME=... VERSION=N  promover a Production

        ▸ AWS — Teardown / Recovery / Destroy
            task teardown           destroy volatiles (preserva storage + network)
            task rebuild            re-apply tras teardown + up   (recovery, no destructivo)
            task ops:state-drift    detecta buckets S3 en AWS ausentes en tfstate
                                    (causa BucketAlreadyExists en apply; corre auto en deploy)
            task destroy            DESTRUCTIVO: S3 + ECR + RDS + todo lo demas
            task nuke               DESTRUCTIVO + borra tfstate + OIDC (irreversible)

        ▸ Variables  (override por CLI: VAR=valor)
            VARIETIES   POP (default) | POP,VENTURA,... | all
            TUNING      smoke ~1m | dev ~20m | prod ~2h | prod_xl ~4-6h (default)
            PARALLEL    1 (default) | N variedades en paralelo
            SEED        42 (default)

        ▸ URLs locales  (con servicios up)
            http://localhost:8501             UI Streamlit (dashboard gerencial)
            http://localhost:8000/docs        API FastAPI (Swagger)
            http://localhost:5000             MLflow UI (tracking + runs)
            http://localhost:8080/reports/    dashboards HTML por variedad
            http://localhost:8080/artifacts/  joblib + best_params

        Mas detalle:  task --list           (todas las tasks: incluye infra:*, ecr:*, batch:*, ops:*, local:*)
                      GUIA_MLOPS_AWS_V2.md  (guia paso a paso: local + AWS)
        EOF
```

> **Por qué cuatro secciones AWS y no una** — separa por ciclo de vida (bootstrap / primer stand-up / operación día-2 / destroy) para mostrar el **orden temporal**, no un listado plano. "Primer stand-up" embebe la secuencia 0→7 del mapa del camino (#4.1.10), así quien corre `task` ve qué ejecutar y en qué orden. "Teardown / Recovery / Destroy" agrupa los 4 comandos de ciclo de vida (los labels `DESTRUCTIVO` van por línea — rebuild **no** lo es).

### 4.1.9 Verificacion

```bash
# El default imprime el menu curado (local + AWS). Si los `includes:` cargan
# bien, no hay error; los namespaces (`infra:`, `ecr:`, `batch:`, `ops:`,
# `local:`) quedan disponibles para `task <ns>:<task>`.
task

# Validar sintaxis sin ejecutar nada (substitucion de vars + estructura)
task --dry deploy | head -5    # ver que el comando se renderiza con SUFFIX resuelto
task --dry nuke   | head -10   # ver que empty_bucket recibe el nombre completo
```

> **Gotcha #4.1.9**: si `task` falla con `open ./tasks/X.yml: no such file`, falta crear el archivo del `includes:` correspondiente — comentar el include hasta crear el yml. Commit: `feat(tasks): refactor AWS namespacing — 4 archivos + lib/`.

### 4.1.10 Mapa del camino: del plan a `/reports/POP/` visible

Tenés código, infra y taskfile listos. Acá arranca la ejecución real.

**Prerrequisitos** (los 5 con ✓ antes de seguir):

- Código del trainer (Tramo I, secciones 1-4): `task build` corre OK
- Módulos Terraform (Parte 3): `terraform validate` da Success en envs/prod
- Bootstrap backend (Parte 2): bucket `${PROJECT}-tfstate-${ACCOUNT_SUFFIX}` existe
- Taskfile (Parte 4, sección 4.1): `task` imprime el menú sin error
- Sesión AWS cargada (Capítulo 3, sección 3.5): `echo $ACCOUNT_SUFFIX` no vacío

```mermaid
flowchart TD
    P0["0. source scripts/prod.env<br/><i>vars del shell cargadas</i>"]
    P1["1. task infra:plan<br/><i>preview, no toca AWS</i>"]
    P2["2. task infra:apply TARGET=module.storage<br/><i>Ola A: 2 S3 + 3 ECR vacíos (~1 min)</i>"]
    P3["3. aws s3 cp data/BD_*.xlsx s3://...<br/><i>Excel en bucket data</i>"]
    P4["4. task ecr:build-all<br/><i>Ola B: 5 imágenes pusheadas (~12 min)</i>"]
    P5["5. task infra:apply<br/><i>Ola C: red+RDS+ALB+Fargate+Batch+λ (~17 min)</i>"]
    P6["6. task batch:smoke<br/><i>1 job Batch POP smoke (~12 min)</i>"]
    P7["7. open http://$ALB/reports/POP/<br/><b>✅ Gate Tramo II: dashboard visible</b>"]
    P8["8. (opcional) terraform plan -destroy<br/><i>dry-run de la bajada, no destruye</i>"]

    P0 --> P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7
    P7 -.-> P8

    style P7 fill:#d4edda,stroke:#155724
    style P8 fill:#f8f9fa,stroke:#6c757d,stroke-dasharray: 5 5
```

**Notas clave:**

- Los pasos 2 y 5 son ambos `task infra:apply`, pero el 2 usa `TARGET=module.storage` (solo S3+ECR) y el 5 es full. Esa separación es el corazón de Ola A → B → C.
- El paso 1 (`plan` sin TARGET) es informativo. Real apply parcial corre en paso 2.
- Paso 8 es opcional y NO destructivo: `plan -destroy` muestra el grafo sin ejecutarlo. Distinto a `task infra:destroy` (ese sí borra todo).
- Tiempo total primer stand-up: ~45 min wall-clock, ~10 min de atención activa.
- Si falla el paso N, NO saltes a N+1. Cada subsección tiene su bloque de recovery.

**Referencias por paso** (sección donde está el detalle):

- Paso 0 → Capítulo 3, sección 3.5
- Paso 1 → Parte 4, secciones 4.1.9 y 4.5
- Paso 1.5 → Parte 4, sección 4.2 (verificación sintáctica `task infra:validate`)
- Pasos 2-3 → Parte 4, sección 4.3
- Paso 4 → Parte 4, sección 4.4
- Paso 5 → Parte 4, sección 4.5
- Paso 6 → Parte 4, sección 4.6
- Paso 7 → Parte 4, sección 4.6.3
- Paso 8 → Parte 4, sección 4.8.3

> **Gotcha #4.1.10**: empezar #4.3 sin `source scripts/prod.env` en terminal nueva → las verificaciones `aws s3api ...` fallan con `$ACCOUNT_SUFFIX` y `$PROJECT` vacíos.

## 4.2 Verificación sintáctica previa al apply

Antes del primer `task infra:apply` (Ola A), una sola verificación: que el HCL parsea y formatea. Esto cuesta ~10 s y atrapa typos antes de un apply de varios minutos. La validación se hace **acá, no en Parte 3**, porque entre que terminás de pegar los módulos y arrancás el deploy pueden pasar días — re-validar justo antes del apply es lo único que importa.

```bash
# Prereq: `source scripts/prod.env` en esta terminal (sección 3.5).
task infra:validate
# Esperado: "Success! The configuration is valid."
```

> **Qué hace la task**: `_init` (descarga providers + backend S3) → `terraform fmt -check -recursive` → `terraform validate`. Si falla `fmt -check`, correr `terraform fmt -recursive infra/` para auto-arreglar. Si falla `validate`, el error indica archivo y línea — fijar y re-correr. **No avanzar a 4.3 hasta que diga "Success"**.

> **Gotcha #4.2**: `validate` cold (sin `.terraform/`) funciona porque la task lleva `deps: [_init]`. Si igual falla con "Module not installed", correr `task infra:plan` una vez para forzar el init.

---

## 4.3 Ola A — apply storage solo

Crea los 2 buckets S3 (con versioning + AES256 + public-access-block) + 5 repos ECR (trainer, mlflow, reports, api, ui). Tiempo: ~1 min.

```bash
# Prereq: una vez por terminal nueva, `source scripts/prod.env` (sección 3.5).
# La task `infra:apply` no lee vars del shell (PROJECT/REGION tienen default
# en Taskfile.yml; SUFFIX lo computa scripts/aws-suffix.sh lazy), pero las
# verificaciones `aws s3api ...` de mas abajo si necesitan $PROJECT y
# $ACCOUNT_SUFFIX en el shell.

task infra:apply TARGET=module.storage
```

> **Alternativa para sandbox local sin Terraform**: si solo necesitas los 2 buckets S3 (sin ECR, sin versioning, sin Terraform state — caso tipico de Tramo I), usa `task local:ensure-buckets` (definida en sección 4.6 del Tramo I). Crea los buckets con el mismo `ACCOUNT_SUFFIX` pero **sin versioning**, asi local y prod comparten nombres sin colisionar. Para el stand-up de prod (esta seccion), seguir con `task infra:apply` — el hardening (versioning + encryption) viene en el modulo storage.

### Verificacion Ola A

```bash
# Fuente de verdad post-apply: los outputs Terraform.
# El propio apply los imprime al terminar; volver a verlos con:
terraform -chdir=infra/envs/prod output
# Esperado: ecr_trainer_url / ecr_mlflow_url / ecr_reports_url / data_bucket /
# artifacts_bucket con valores no vacios.

# Unica invariante NO incluida en outputs (hardening): versioning ON.
source scripts/ensure-env.sh
for kind in data artifacts; do
  aws s3api get-bucket-versioning --bucket "${PROJECT}-${kind}-${ACCOUNT_SUFFIX}" \
      --query Status --output text   # Esperado: Enabled
done
```

### Subir el Excel inicial al bucket de data

Antes del primer training real, el bucket `data` necesita el Excel:

```bash
# Prereq: `source scripts/prod.env` (sección 3.5) ya exporta DATA_BUCKET
# (= ${PROJECT}-data-${ACCOUNT_SUFFIX}). Si abriste terminal nueva y solo
# tenés PROJECT/ACCOUNT_SUFFIX, `source scripts/ensure-env.sh` lo deriva.
source scripts/ensure-env.sh

# Asume que tenes data/BD_HISTORICO_ACUMULADO.xlsx en local (workflow normal)
aws s3 cp data/BD_HISTORICO_ACUMULADO.xlsx \
    "s3://${DATA_BUCKET}/BD_HISTORICO_ACUMULADO.xlsx"

# Verificar
aws s3 ls "s3://${DATA_BUCKET}/" --human-readable
```

> **En consola AWS veras** despues de Ola A:
> - S3 → Buckets → `ml-training-data-<suffix>` (con
>   `BD_HISTORICO_ACUMULADO.xlsx` adentro) y `ml-training-artifacts-<suffix>`
>   (vacio).
> - ECR → Repositories → 3 (`ml-training`, `ml-training-mlflow`,
>   `ml-training-reports`) los 3 vacios — el push viene en Ola B.

> **Gotcha #4.3**: si Ola A falla con `BucketAlreadyExists`, hay dos escenarios:
> 1. **Drift de tfstate** (el más común): los buckets `*-data-*` / `*-artifacts-*` existen en AWS pero no en tfstate — pasa tras un `task nuke` parcial (los buckets con versiones sobreviven porque `force_destroy=false`), tras re-bootstrapear el backend, o tras borrar el state a mano. **Diagnóstico + fix**: `task ops:state-drift` lista qué buckets faltan e imprime el `terraform import` exacto. Después re-correr `task infra:apply TARGET=module.storage` (los `versioning/SSE/PAB/lifecycle` se crean encima sin tocar los objetos). `task deploy` ya corre este check como pre-step. Tiempo: ~30 s.
> 2. **Colisión real de nombre**: el `ACCOUNT_SUFFIX` choca con otra cuenta o con un bucket recién borrado que aún retiene el nombre. Revisar #3.5 (cálculo del suffix) y re-correr `source scripts/prod.env`. Tiempo: ~2 min.

## 4.4 Ola B — build + push 5 imagenes a ECR

Las 5 imagenes son:

| Imagen | Dockerfile | Contexto | Tag | Para que |
|---|---|---|---|---|
| `ml-training` | `./Dockerfile` (raiz, ya existe) | raiz | `latest` + `sha-<git-sha>` | Trainer en AWS Batch |
| `ml-training-mlflow` | `./docker/mlflow/Dockerfile` (ya existe en local) | raiz | `v3.12.0` | MLflow server en Fargate |
| `ml-training-reports` | `./docker/reports/Dockerfile` (creado en 3.6.4) | raiz | `stable` | Nginx que sirve S3 |
| `ml-training-api` | `./api/Dockerfile` (#3.12.4) | **raiz** (necesita `src/`) | `latest` + `sha-<git-sha>` | API FastAPI en Fargate |
| `ml-training-ui` | `./ui/Dockerfile` (#3.12.8) | **`ui/`** | `latest` + `sha-<git-sha>` | UI Streamlit en Fargate |

### 4.4.1 Como funciona

La task `ecr:build-all` (definida en sección 4.1.4) encadena 5 invocaciones de
`ecr:build` con `IMG=trainer/mlflow/reports/api/ui`. Cada una hace
`docker build` con build args (`GIT_SHA`, `BUILD_DATE`, `VERSION`) y
pushea 2 tags: el solicitado (`latest`/`v3.12.0`/`stable`) y
`sha-<git-sha-corto>` para rollback determinista. El **contexto** de build lo
resuelve la task: `api` usa la raiz (para incluir `src/`), `ui` usa `ui/`.

### 4.4.2 Overrides via variables CLI

```bash
# Override del tag (e.g. bump version de MLflow)
task ecr:build IMG=mlflow TAG=v3.13.0

# Solo trainer (re-build despues de cambio de codigo)
task ecr:build IMG=trainer
```

### 4.4.3 Ejecutar

```bash
# Prereq: `source scripts/prod.env` (sección 3.5). La task `ecr:build-all` computa
# ACCOUNT_ID y la URI de ECR internamente, no las lee del shell.
task ecr:build-all
```

### Verificacion Ola B

```bash
# Push de imagenes NO esta en terraform outputs (Terraform no trackea estado ECR).
# Check directo de las 5 tags moviles esperadas:
for pair in "ml-training:latest" "ml-training-mlflow:v3.12.0" "ml-training-reports:stable" "ml-training-api:latest" "ml-training-ui:latest"; do
  repo="${pair%:*}"; tag="${pair#*:}"
  aws ecr list-images --repository-name "$repo" \
      --query "imageIds[?imageTag=='${tag}']" --output text
done
# Esperado: cada linea no vacia (imageDigest + tag).
```

> **En consola AWS veras** despues de Ola B:
> - ECR → Repositories → `ml-training` → Images: 2 tags
>   (`latest` + `sha-<12chars>`) con `imageSizeInBytes` >0 y
>   `imagePushedAt` reciente.
> - ECR → `ml-training-mlflow` → Images: 2 tags (`v3.12.0` + `sha-...`).
> - ECR → `ml-training-reports` → Images: 2 tags (`stable` + `sha-...`).
> - Cada imagen muestra el resultado del scan-on-push (vulnerabilities
>   findings: usualmente "No findings" en imagenes oficiales, algunos
>   MEDIUM/LOW en `ml-training` por las deps de Python).

> **Gotcha #4.4**: `docker buildx` falla en WSL con `no space left on device` → `docker system prune -a` libera capas viejas. Si `aws ecr get-login-password` da `403`, re-autenticar el perfil AWS. Tiempo: ~10 min primera vez, ~3 min con cache.

## 4.5 Ola C — apply full (en 4 sub-olas con checkpoint)

Ahora todo el resto (`network`, `mlflow`, `reports`, `batch`,
`monitoring`, `lambdas`, `scheduler`, `cicd`). Tiempo total: ~15-20
min. El que mas demora: RDS create (~8 min) + ALB warmup + Fargate
task launch (~3 min).

**Por que se parte en 4 sub-olas**: un `terraform apply` monolitico
falla "en silencio" — si el modulo 5 de 8 explota, te enteras 18 min
despues. Partiendo por capa de dependencia, cada checkpoint da
feedback en 2-5 min y el error es localizable. Tambien permite saltar
a la siguiente sub-ola sin re-planear las anteriores.

### 4.5.1 Sub-ola C1 — `network` (red base, ~1-2 min)

```bash
task infra:plan TARGET=module.network
task infra:apply TARGET=module.network
```

> **En consola AWS**: VPC console → Your VPCs → `ml-training-vpc`;
> Subnets → 2 (public + private); NAT Gateways → 1 (state=available);
> Security Groups → 6 (`ml-training-sg-alb`, `-sg-mlflow`, `-sg-batch`,
> `-sg-rds`, `-sg-api`, `-sg-ui`).

### 4.5.2 Sub-ola C2 — `mlflow` + `reports` + `api` + `ui` (RDS + ALB + 4 Fargate, ~12 min)

```bash
task infra:plan TARGET=module.mlflow
task infra:apply TARGET=module.mlflow      # ~8 min (RDS create domina)

task infra:plan TARGET=module.reports
task infra:apply TARGET=module.reports     # ~2 min

# App stack (Capa 4.5). api primero (la ui depende de api.internal_url).
task infra:apply TARGET=module.api         # ~2 min
task infra:apply TARGET=module.ui          # ~2 min

# Checkpoint: ALB responde + RDS available + app stack ruteado
export ALB="$(terraform -chdir=infra/envs/prod output -raw alb_dns)"
curl -sI "http://${ALB}/" | head -1          # MLflow   -> HTTP/1.1 200 OK
curl -sI "http://${ALB}/reports/" | head -1  # Reports  -> HTTP/1.1 200 OK
curl -sI "http://${ALB}/app/" | head -1      # UI       -> HTTP/1.1 200 OK
curl -s  "http://${ALB}/api/health" | head -c 200; echo  # API -> JSON {"status":...}
```

> **Nota**: en un stand-up de cero todavia no hay modelos `rnd-forest-*`
> registrados, asi que `/api/health` puede reportar `degraded` (sin modelos).
> Es esperado — la API los servira en cuanto entrenes la primera variedad
> (#4.6). El servicio igual queda `running`/healthy a nivel ECS (el health
> check del container es `/api/health`, que responde 200 aunque no haya modelos).

> **En consola AWS**: RDS → Databases → `ml-training-mlflow`
> (status=Available); ECS → Clusters → `ml-training-cluster` → Services
> (mlflow + reports + **api + ui**, runningCount=1 cada uno); EC2 → Load
> Balancers → `ml-training-alb` → Listener :80 → Rules (reglas nuevas:
> prio 70 `/app/*` → tg-ui; prio 88/89 `/api/*`+`/docs` → tg-api); EC2 →
> Target Groups → `ml-training-tg-api` + `ml-training-tg-ui` (healthy);
> Cloud Map → `api.ml-training.local`.

### 4.5.3 Sub-ola C3 — `batch` + `lambdas` (compute + orquestacion, ~3 min)

> **Nota de dependencias**: `module.lambdas` referencia `module.monitoring.sns_topic_arn` para el notifier. Al usar `terraform apply -target=module.lambdas`, Terraform arrastra `module.monitoring` automaticamente — asi que aunque el orden documentado es C3 (batch + lambdas) antes que C4 (monitoring), en la practica `monitoring` se crea durante C3 como dependencia transitiva. C4 entonces es no-op para `monitoring`.

```bash
task infra:apply TARGET=module.batch
task infra:apply TARGET=module.lambdas
```

> **En consola AWS**: Batch → Job queues (2: spot + ondemand, ambos
> VALID); Compute environments (2: ml-training-ce-spot, -ondemand);
> Job definitions → `ml-training-trainer`; Lambda → Functions (2:
> dispatcher + notifier).

### 4.5.4 Sub-ola C4 — `monitoring` + `scheduler` + `cicd` (~2 min)

```bash
task infra:apply TARGET=module.monitoring
task infra:apply TARGET=module.scheduler
task infra:apply TARGET=module.cicd
```

> **En consola AWS**: SNS → Topics → `ml-training-alerts`; CloudWatch
> → Alarms (**N + 2** donde N = `length(var.varieties)`: batch-failed +
> mape-<variety> × N + alb-5xx; el conteo escala automatico si agregas
> o quitas variedades); EventBridge → Rules (4: start, stop, rds-keepstop,
> batch-failed); Lambda → `ml-training-scheduler`; IAM → Roles
> (`gha-deploy`, `gha-train`).

### 4.5.5 Apply full alternativo (cuando ya pasaste por C1-C4 una vez)

Para re-applies idempotentes (despues de algun cambio menor), una vez
validado que todo arriba existe, podes usar:

```bash
task infra:plan
task infra:apply
```

> **Cuando usar el apply monolitico**: re-deploys post-stand-up. NUNCA
> en el primer stand-up — si algun modulo falla, debug es mucho mas
> caro.

### Recovery comun durante Ola C

| Sintoma | Causa probable | Fix |
|---|---|---|
| `RDSCreate` cuelga 15+ min | Subnet group sin AZs distintas o no hay capacity en la AZ | Re-apply (idempotente); si persiste, reduce a 1 AZ en module.network |
| `aws_ecs_service.mlflow: timeout waiting for steady state` | La imagen MLflow no esta en ECR o el comando rompe en startup | Re-corre 4.4.3; revisa `aws logs tail /ecs/ml-training/mlflow --follow` |
| `aws_lambda_function: source_code_hash mismatch` | Editaste el .py pero no re-zip-eo | `terraform apply` lo detecta y re-zipea (idempotente) |
| `permission denied: iam:CreateRole` | Tu profile AWS no tiene IAM permissions | `aws sts get-caller-identity` y revisa que sea admin/role-with-IAM |
| State lock acquire timeout | Otro `terraform apply` corriendo / state lock huerfano | `terraform force-unlock <LOCK_ID>` (mostrado en el error) |

### Verificacion Ola C

```bash
# Outputs cubren ALB DNS, ECR URLs, bucket names, queues.
terraform -chdir=infra/envs/prod output

# Smoke unico que outputs NO cubre: ALB sirve trafico Y RDS quedo available.
export ALB="$(terraform -chdir=infra/envs/prod output -raw alb_dns)"
curl -sf "http://${ALB}/" > /dev/null         && echo "MLflow OK"
curl -sf "http://${ALB}/reports/" > /dev/null && echo "reports OK"

aws rds describe-db-instances --db-instance-identifier ml-training-mlflow \
    --query 'DBInstances[0].DBInstanceStatus' --output text   # Esperado: available
```

Si los 3 chequeos pasan, la infra esta arriba. Lambdas + EventBridge rules no se
verifican aca: si el apply termino sin error, existen — `task infra:apply` aborta
en el primer recurso roto.

> **Gotchas #4.5**:
> - **C2 (RDS)**: subnet group requiere 2 AZs. Si la VPC solo tiene 1 subnet privada → `DBSubnetGroupDoesNotCoverEnoughAZs`. Agregar segunda AZ a `module.network`.
> - **C3 (CI/CD)**: si el OIDC provider de #2.5 no existe, `data "aws_iam_openid_connect_provider"` falla. Bootstrapear OIDC antes de `module.cicd`.
> - **Tiempo**: C1 ~2m + C2 ~10m (RDS domina) + C3 ~3m + C4 ~2m = ~17m total.

## 4.6 Smoke test — entrenar 1 variedad end-to-end

> [!TIP]
> Este es el **gate de aceptación del Tramo II**: si pasa, la infra AWS
> está bien armada y podés avanzar a Parte 5 (CloudWatch metric) y Parte 6
> (CI/CD). Si falla, **no parches workarounds** — volvé a las olas de sección 4.5
> y verificá los checkpoints. El costo del smoke es centavos (~10 min de
> EC2 Spot c6i.xlarge).

Esto verifica:

1. Lambda dispatcher recibe y valida payload (variety whitelist + hydrate S3).
2. Batch submit funciona (dispatcher → `submit-job`).
3. EC2 Spot arranca + corre el container.
4. El trainer hydrate-a la data desde S3.
5. Logs llegan a CloudWatch.
6. MLflow registra el run.
7. Outputs syncan a S3.
8. Dashboards visibles en `/reports/`.
9. Custom metric MAPE publicada (despues de Parte 5; en este smoke
   no se valida todavia).

### 4.6.1 Como funciona la task `batch:smoke`

`batch:smoke` es un atajo a `batch:train VARIETIES=POP TUNING=smoke`. El path:

0. **Preflight** `assert_jobdef_image` (en `tasks/lib/batch_wait.sh`): resuelve la imagen de la job-def `ml-training-trainer` ACTIVE y confirma que ese tag exista en ECR. Si falta, aborta en ~2s con el comando exacto a correr — antes de gastar Lambda invoke + ~3 min de Spot.
1. **Lambda invoke** al `ml-training-dispatcher` con payload `{varieties:"POP", tuning:"smoke"}`.
2. El dispatcher valida variety + S3 key, llama `aws batch submit-job` con `S3_DATA_KEY` como env override, y devuelve el `jobId`. (El hydrate real ocurre dentro del trainer en `main.py::_hydrate_data_from_s3`.)
3. **Polling** via `aws batch describe-jobs` cada 30s hasta que `status` sea `SUCCEEDED` (exit 0) o `FAILED` (exit 1). Funcion `wait_job` en `tasks/lib/batch_wait.sh`.

Una sola via de submit (el dispatcher) — el dev local con `task batch:train` y el CI con `task batch:train` ejecutan IDENTICO el camino.

### 4.6.2 Ejecutar

```bash
task batch:smoke
```

Tiempo total esperado: **10-15 min** desde invoke hasta `SUCCEEDED`.
Breakdown:
- Lambda invoke + submit: <5 s
- EC2 Spot provisioning: 2-5 min
- Container pull (primera vez ~3 GB): 3-5 min (cached despues)
- Trainer ejecucion `--tuning smoke`: 2-4 min (smoke = 5 iter Optuna)
- S3 sync + container shutdown: ~30 s

### 4.6.3 Verificacion post-smoke

```bash
source scripts/ensure-env.sh   # aborta si $PROJECT o $ACCOUNT_SUFFIX vacias

export ALB="$(terraform -chdir=infra/envs/prod output -raw alb_dns)"
export ARTIFACTS_BUCKET="${PROJECT}-artifacts-${ACCOUNT_SUFFIX}"

# 1) MLflow tiene el run
curl "http://${ALB}/api/2.0/mlflow/experiments/search" \
    -X POST -H "Content-Type: application/json" \
    -d '{}'
# Esperado: al menos un experimento llamado "POP" con runs.id

# 2) S3 tiene los artifacts
aws s3 ls "s3://${ARTIFACTS_BUCKET}/artifacts/" --recursive --human-readable | grep POP
# Esperado: final_pipeline_POP_*.joblib + run_summary_POP*.json

# 3) S3 tiene los reports
aws s3 ls "s3://${ARTIFACTS_BUCKET}/reports/" --recursive | grep POP
# Esperado: dashboard_POP.html

# 4) /reports/POP/ accesible via ALB
curl "http://${ALB}/reports/POP/"   # esperado: HTML del dashboard

# 5) Custom metric MAPE publicada (despues de Parte 5, no ahora)
# Para esta primera vuelta sin patch del trainer, NO esperar metricas
# en namespace "ml-training/Training" todavia.
```

Si (1) y (2) salen OK, **el smoke pasa**. (3) y (4) tambien deberian
salir OK porque `main.py:scripts.s3_sync.sync_to_s3` ya sube reports
si `S3_ARTIFACTS_BUCKET` esta seteado (ya esta, via job-def).

> **Gotcha #4.6**: si el job queda en `RUNNABLE` eternamente → #8.3.1 (típicamente Spot bid bajo, sin capacity en la AZ del CE, o role Batch sin `ec2:RunInstances`). Tiempo: ~8 min (spin-up Spot + pull 3 GB + Optuna 5 iter).

> **Gotcha #4.6b (imagen ausente)**: si el job pasa `RUNNABLE → STARTING → FAILED` con `reason=Task failed to start` y `container.reason=CannotPullImageManifestError: manifest unknown`, el tag que pinea la job-def NO existe en ECR (típico: bump de `trainer_image_tag` sin el `task ecr:build` correspondiente — ver Gotcha Parte 5). Desde esta versión el preflight `assert_jobdef_image` lo detecta antes de encolar. Verificá los tags reales con `aws ecr describe-images --repository-name ml-training --query 'imageDetails[].imageTags'` y construí el que falte: `task ecr:build IMG=trainer TAG=<tag>`.

## 4.7 Confirmar suscripcion SNS

SNS manda un email de confirmacion cuando creas la suscripcion (Parte
3.8). Tenes que clickear el link para activarla.

```bash
export TOPIC_ARN="$(terraform -chdir=infra/envs/prod output -raw sns_topic_arn)"

# Estado de la suscripcion
aws sns list-subscriptions-by-topic \
    --topic-arn "${TOPIC_ARN}" \
    --query 'Subscriptions[].[Endpoint,SubscriptionArn]' --output table
```

Si `SubscriptionArn` dice `PendingConfirmation`, revisa el mail
(`abantodca@gmail.com`) y clickea "Confirm subscription".

Test:

```bash
aws sns publish \
    --topic-arn "${TOPIC_ARN}" \
    --subject "TEST: ml-training alerts" \
    --message "Si recibis este mail, la suscripcion esta OK."
```

> **Gotcha #4.7**: el email de confirmación SNS puede caer en spam (remitente `no-reply@sns.amazonaws.com`). Si tampoco está ahí, re-disparar con `terraform apply -replace=module.monitoring.aws_sns_topic_subscription.email`. Validar luego con `aws sns list-subscriptions-by-topic --topic-arn "${TOPIC_ARN}"` (debe mostrar `SubscriptionArn` con ARN real, no `PendingConfirmation`).

## 4.8 Catalogo de tasks operativas

Las implementaciones viven en `tasks/*.yml` (definidas en sección 4.1.3-4.1.8). Esta seccion es referencia de uso.

### 4.8.1 Re-entrenar (`task batch:train`)

Submit via Lambda dispatcher (valida variety + S3 key) + polling hasta `SUCCEEDED`/`FAILED`. Multi-variedad en serie. El hydrate de S3 corre dentro del trainer (`main.py::_hydrate_data_from_s3`), no en el dispatcher.

```bash
task batch:train VARIETIES=POP                # espera hasta terminar
task batch:train VARIETIES=POP,JUPITER        # N variedades, una tras otra
task batch:train VARIETIES=POP WAIT=false     # fire-and-forget; el notifier mandara mail
task batch:train VARIETIES=POP TUNING=prod_xl # ~5-6h en On-Demand (evita kills Spot)
```

### 4.8.1.1 EDA exploratorio on-demand (`task batch:eda`)

El EDA es una necesidad **aparte y opcional** del entrenamiento: lo corres cuando querés inspeccionar la calidad/drift de los datos de una variedad, tantas veces como haga falta. No forma parte del pipeline de `batch:train` — es standalone y repetible.

Mismo camino que `batch:train` (dispatcher → Batch → hydrate de S3), pero el dispatcher recibe `mode=eda` y arma `command=["--eda", "--varieties", ...]`. El trainer, con `--eda`, corre `src.diagnostics.eda.run_eda` por variedad en vez de entrenar y sincroniza los HTML a S3 — quedan visibles en `http://$ALB/reports/EDA_<variety>_<ts>.html`.

```bash
task batch:eda VARIETIES=POP              # EDA de una variedad, espera a que termine
task batch:eda VARIETIES=POP,JUPITER      # varias en un solo job
task batch:eda VARIETIES=POP WAIT=false   # fire-and-forget
```

> **Equivalente local**: `task eda VARIETIES=POP` (corre el mismo `run_eda` en docker compose contra la data local). El de Batch es para correrlo contra la data ya hidratada en S3, sin levantar el stack local.

> **Nota**: el `mode` lo agrega el dispatcher (`infra/lambdas/dispatcher.py`); si cambiás esa Lambda, redeployá con `task infra:apply TARGET=module.lambdas` para que el código nuevo quede activo.

### 4.8.2 Lifecycle del cluster

```bash
# Encender / apagar (atajos high-level → ops:up / ops:down)
task wake                          # idempotente: skip si ya UP, espera RDS+ALB healthy
task sleep                         # aborta si hay Batch RUNNING
task ops:down COOLDOWN=600         # apaga tras 10 min (para CI post-train)

# Estado
task status                        # outputs Terraform + RDS + ECS + Batch
task ops:status                    # solo cluster (sin outputs Terraform)

# Teardown / rebuild (preserva storage + network)
task teardown                      # ops:down + destroy de modulos volatiles
task rebuild                       # re-apply (idempotente) + ops:up
```

`ops:up` es la misma task que usan los workflows CI auto-train. Hace pre-check `/health`, invoca `scheduler.start` si MLflow esta DOWN, y espera RDS available (12 min max) + ALB 200 (5 min max). El estado previo queda en `/tmp/wake-status` (`true|false`) para que el workflow decida si tiene que apagar al final.

### 4.8.3 Destroy total

```bash
task destroy   # vacia buckets + purga ECR + terraform destroy (backend tfstate queda)
task nuke      # ↑ + borra tfstate + OIDC  (IRREVERSIBLE: requiere re-bootstrap)
```

### 4.8.4 Resumen

| Operacion      | Comando                        |
|----------------|--------------------------------|
| Build todo     | `task deploy`                  |
| Smoke E2E      | `task smoke`                   |
| Encender       | `task wake`                    |
| Apagar         | `task sleep`                   |
| Estado         | `task status`                  |
| Teardown       | `task teardown`                |
| Rebuild        | `task rebuild`                 |
| Destroy / Nuke | `task destroy` / `task nuke`   |

`task --list` muestra el catalogo completo (`infra:*`, `ecr:*`, `batch:*`, `ops:*`).

---

> **Cierre Parte 4.** Infra arriba (ALB 200), smoke POP OK, SNS email
> confirmado. **Próximos pasos**: #5 (patch MAPE→CloudWatch) → #6 (3 workflows
> GHA) → #7 (promotion gate) → #8-12 (runbook/costos/hardening/troubleshooting).

---

# Parte 5 — Patch del trainer (emitir MAPE a CloudWatch)

> **STATUS: APLICADO** (auditoria 2026-05-18). El patch esta integrado en `src/orchestration/variety_runner.py` (import + llamada `emit_mape_metric(variety=variety, mape_value=champion.oof_mape)` despues del bloque mape_ok/gap_ok). `trainer_image_tag` bumpeado a `v0.2.0` en `infra/envs/prod/terraform.tfvars`. La proxima vez que rebuiledes con `task ecr:build IMG=trainer` la imagen llevara el patch.

Sin este patch, las alarmas de `module.monitoring` (Parte 3 sección 3.8) que escuchan el namespace `ml-training/Training` con dimension `variety` nunca disparan — el namespace esta vacio y un MAPE alto pasa silencioso.

## Mapa del camino — Parte 5

Parte 5 agrega 1 capa de código (emitir MAPE a CloudWatch) sobre un sistema que ya funciona. Saltarla no rompe nada — solo perdés la señal de alarma.

**Prerrequisitos:**

- Smoke de Parte 4 verde: `task batch:smoke` corrió SUCCEEDED.
- Alarmas creadas en Ola C4 (Parte 4, sección 4.5.4): `aws cloudwatch describe-alarms --alarm-name-prefix ml-training-mape-` lista N alarmas.
- Trainer corre en local: `docker compose run --rm --user "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp trainer --varieties POP --tuning smoke` termina exit 0 (el `--user` evita el `PermissionError` en los bind-mounts; en Batch no hace falta porque no hay bind-mount).

```mermaid
flowchart TD
    P1["1. Localizar insert point<br/><i>champion.oof_mape tras quality gate</i>"]
    P2["2. Crear src/utils/cloudwatch_metrics.py<br/><i>Gate por AWS_BATCH_JOB_ID (no-op local)</i>"]
    P3["3. Invocar emit_mape_metric() en variety_runner.py<br/><i>1 import + 1 línea</i>"]
    P4["4. Verify local<br/><i>docker compose run trainer → exit 0,<br/>log SIN 'CloudWatch MAPE=...'</i>"]
    P5["5. Re-build + push + apply + smoke<br/><i>task ecr:build TAG=v0.2.0 → bump tfvars →<br/>task infra:apply TARGET=module.batch → task batch:smoke</i>"]
    GATE["✅ Gate Parte 5<br/>CloudWatch namespace<br/>'ml-training/Training' con<br/>datapoints dim=variety"]

    P1 --> P2 --> P3 --> P4 --> P5 --> GATE

    style GATE fill:#d4edda,stroke:#155724
```

**Notas clave:**

- Pasos 2-3 son edición de código. NO toques infra hasta pasar paso 4 (verify local). Un patch mal-importado se ve en 2 segundos con `python3 -m py_compile`, no en 12 min de un Batch fallido.
- Gate por `AWS_BATCH_JOB_ID` es deliberado: docker compose siempre tiene `S3_ARTIFACTS_BUCKET`, gatear por bucket contaminaría el namespace prod con runs locales.
- El bump de tag (`v0.2.0`) NO es opcional. Si dejás `latest`, ECR sobrescribe el digest y perdés rollback rápido.
- Tiempo total: ~15 min (10 min son rebuild+push+apply+smoke, 5 min de edición).

**Referencias:** paso 1 → 5.1 · paso 2 → 5.2 · paso 3 → 5.3 · paso 4 → 5.4 · paso 5 → 5.5.

> **Gotcha Parte 5 (bump y build son DOS pasos; el orden importa)**: el tag trainer vive en dos sitios — `terraform.tfvars` (lo que la job-def referencia) y ECR (lo que existe para pullear). Desincronizarlos rompe de dos formas opuestas:
>
> - **Build sin bump**: corres `task ecr:build IMG=trainer TAG=v0.2.0` pero olvidás editar `terraform.tfvars` → la imagen está en ECR pero el job-def sigue en el tag viejo y CloudWatch sigue vacío.
> - **Bump sin build**: editás `terraform.tfvars = "v0.2.0"` + `task infra:apply` pero nunca construís ese tag → la job-def apunta a un manifest inexistente y el job **falla al arrancar** con `CannotPullImageManifestError: manifest unknown` (`reason=Task failed to start`, tras ~3 min en RUNNABLE/STARTING).
>
> **Mitigaciones ya integradas** (no hace falta acordarse a mano):
> - `task ecr:build-all` (y por ende `task deploy`) lee `trainer_image_tag` de `terraform.tfvars` y construye el trainer con ESE tag — el stand-up completo queda auto-consistente.
> - `task batch:train` corre `assert_jobdef_image` (en `tasks/lib/batch_wait.sh`) antes de encolar: si la job-def apunta a un tag ausente en ECR, aborta en ~2s con el comando exacto a correr, en vez de quemar ~3 min en un Batch fallido.
>
> Si bumpeás el tag a mano fuera de `build-all`, los DOS pasos siguen siendo tuyos: `task ecr:build IMG=trainer TAG=<nuevo>` **y** editar `terraform.tfvars` + `task infra:apply TARGET=module.batch`. Validá con `aws ecr describe-images --repository-name ml-training --query 'imageDetails[].imageTags'`. Commit: `feat(monitoring): emit MAPE custom metric a CloudWatch con dim=variety`.

## 5.1 Donde se inserta

El patch crea `src/utils/cloudwatch_metrics.py::emit_mape_metric()` y lo invoca al final de `train_variety()` en `src/orchestration/variety_runner.py`, justo despues del quality gate (donde el log dice "CAMPEON pasa quality gate" o "RECHAZADO por calidad operativa") y antes de generar el Excel/Dashboard.

**Por que aca y no en `runners.py`**: el `champion` existe como objeto `ModelResult` (dataclass del paso 05) solo dentro de `train_variety`. En `runners.py` solo se ve el dict agregado. **Por que `champion.oof_mape` y no `champion.full_mape`**: la alarma mide degradacion en datos no vistos (OOF) — `full_mape` es in-sample (optimista) y mete ruido cuando el modelo memoriza el train.

## 5.2 Crear `src/utils/cloudwatch_metrics.py`

```python
"""Emite custom metrics a CloudWatch.

Solo se activa cuando el trainer corre dentro de AWS Batch (detectado via
`AWS_BATCH_JOB_ID`, env var que el servicio Batch inyecta automaticamente y
NO existe en local). En docker compose local es no-op silencioso, evitando
contaminar el namespace prod con datos de smoke tests.
"""
from __future__ import annotations

import logging
import os
from typing import Final

log = logging.getLogger(__name__)

NAMESPACE: Final[str] = "ml-training/Training"


def emit_mape_metric(variety: str, mape_value: float) -> None:
    """Publica MAPE a CloudWatch con dimension `variety`.

    No falla el training si la publicacion falla (best-effort).
    """
    if not os.environ.get("AWS_BATCH_JOB_ID"):
        # Local (docker compose): skip silencioso.
        # AWS_BATCH_JOB_ID lo inyecta el servicio Batch automaticamente.
        return

    try:
        import boto3
    except ImportError:
        log.warning("boto3 no instalado, skip CloudWatch metric")
        return

    try:
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[{
                "MetricName": "MAPE",
                "Dimensions": [{"Name": "variety", "Value": variety}],
                "Value":      float(mape_value),
                "Unit":       "Percent",
            }],
        )
        log.info("CloudWatch MAPE=%.4f emitido (variety=%s)", mape_value, variety)
    except Exception as exc:
        log.warning("CloudWatch put_metric_data fallo: %s", exc)
```

> **Por qué gatear en `AWS_BATCH_JOB_ID` y no en `S3_ARTIFACTS_BUCKET`**: el compose local declara `S3_ARTIFACTS_BUCKET` como requerido (`:?`), así que en local SIEMPRE está seteada — gatear ahí haría que el smoke local publique a CloudWatch prod con las creds del bind-mount `~/.aws`. `AWS_BATCH_JOB_ID` solo existe dentro del container de Batch — gate seguro contra contaminación.

Verificar: `python3 -m py_compile src/utils/cloudwatch_metrics.py` no debe imprimir nada.

## 5.3 Invocar desde el runner

En `src/orchestration/variety_runner.py`:

```python
# 1) Import al inicio del modulo
from src.utils.cloudwatch_metrics import emit_mape_metric

# 2) Dentro de train_variety, despues del bloque `if not mape_ok / elif not gap_ok / else`
#    (linea ~104), antes de `losers = [...]`:
emit_mape_metric(variety=variety, mape_value=champion.oof_mape)
```

**Importante**: la llamada se hace **siempre**, incluso si el quality gate rechaza el modelo (`mape_ok=False`). Eso es deliberado — un MAPE alto rechazado es justamente el caso que la alarma `ml-training-mape-<variety>` quiere capturar. Si solo se emitiera cuando el gate aprueba, las alarmas nunca dispararian en los runs malos.

## 5.4 Verificar local que no rompe

```bash
# AWS_BATCH_JOB_ID solo existe en Batch -> en docker compose local, emit hace skip silencioso
# --user/MPLCONFIGDIR: evitan el PermissionError en los bind-mounts (logs/reports) al
# correr como mluser; en Batch no se usan (no hay bind-mount). En local podes correr
# `task train VARIETIES=POP TUNING=smoke`, que ya los aplica.
docker compose run --rm --user "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp trainer --varieties POP --tuning smoke
```

El log NO debe tener "CloudWatch MAPE=..." (es Batch-only). El training termina exitoso y NO se contamina el namespace prod de CloudWatch.

## 5.5 Re-build + push + verificar end-to-end

```bash
git add src/utils/cloudwatch_metrics.py src/orchestration/variety_runner.py
git commit -m "feat(monitoring): emit MAPE custom metric a CloudWatch con dim=variety"

# Bump version para que ECR retenga la anterior
task ecr:build IMG=trainer TAG=v0.2.0

# Propagar la tag a Batch
# Editar terraform.tfvars: trainer_image_tag = "v0.2.0"
task infra:apply TARGET=module.batch

# Re-correr smoke con el trainer parchado
task batch:smoke

# Confirmar metric publicada
aws cloudwatch list-metrics \
    --namespace "ml-training/Training" \
    --metric-name MAPE \
    --dimensions Name=variety,Value=POP \
    --query 'Metrics[]' --output table

# Y el ultimo datapoint
aws cloudwatch get-metric-statistics \
    --namespace "ml-training/Training" \
    --metric-name MAPE \
    --dimensions Name=variety,Value=POP \
    --start-time "$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ)" \
    --end-time   "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --period 60 \
    --statistics Maximum \
    --query 'Datapoints'
```

Si trae un valor, la alarma `ml-training-mape-pop` ya tiene datos y va a dispararse cuando supere `mape_alarm_threshold` (default 25%).

> **Gotcha #5.x (CloudWatch)**: si `Datapoints` vuelve vacío, confirmar que `terraform.tfvars` tenga el tag nuevo y `module.batch` se re-aplicó. Commit: `feat(runner): invocar emit_mape_metric post-quality-gate`.

> **Orden de pasos según el path**: la secuencia manual de arriba construye el tag *antes* de pinearlo en `terraform.tfvars` — orden seguro. Si en cambio usás el stand-up completo (`task deploy` → `task ecr:build-all`), `build-all` **lee** `trainer_image_tag` de `terraform.tfvars`, así que ahí el bump de tfvars va PRIMERO y el build lo sigue automáticamente. En ambos paths, `task batch:train` valida con `assert_jobdef_image` que el tag exista en ECR antes de encolar.

---

# Parte 6 — CI/CD con GitHub Actions

> **STATUS: NO APLICADO TODAVIA** (auditoria 2026-05-22). El directorio `.github/workflows/` **NO existe** en el repo. La infra IAM (`module.cicd` con roles `ml-training-gha-deploy` y `ml-training-gha-train`) SI esta lista — los workflows YAML que vienen abajo hay que crearlos a mano siguiendo 6.2/6.3/6.4. Hasta entonces, los disparadores (`on: push`, `workflow_dispatch`, etc.) no existen y el pipeline se opera 100% a mano via `task`.
>
> **Nota — enlaces aspiracionales**: los workflows `.github/workflows/{deploy,training,destroy}.yml` y los documentos `docs/adr/` (e.g. `ADR-008-ci-sin-tests-todavia.md`) que se referencian en esta Parte **aun no existen** en el repo. Son material por crear; los links son placeholders hasta que implementes Parte 6 y materialices los ADR.

## Mapa del camino — Parte 6

Parte 6 transfiere la operación del laptop al pipeline. Antes disparabas `task X` a mano; ahora GHA corre los mismos `task X` con OIDC.

**Prerrequisitos:**

- OIDC provider en AWS (Parte 2, sección 2.5): `aws iam list-open-id-connect-providers` lista `token.actions.githubusercontent.com`.
- Módulo `cicd` aplicado en Ola C4 (Parte 4, sección 4.5.4): roles `gha-deploy` y `gha-train` existen.
- Repo GitHub y vos como owner: `gh auth status` OK + `gh repo view --json viewerCanAdminister` da `true`.
- Tasks locales funcionan (Parte 4, sección 4.1): `task ecr:build IMG=trainer` corre OK.

```mermaid
flowchart TD
    P1["1. Modelo de trust<br/><i>2 roles IAM (deploy + train),<br/>trust = solo este repo</i>"]
    P2["2. Filosofía Task=SSOT<br/><i>matriz: qué vive en Terraform,<br/>Task, GHA</i>"]
    P3["3. gh variable set ×9<br/><i>AWS_REGION, role ARNs, ECR,<br/>ALB DNS, ALERT_EMAIL...<br/>Secrets = 0 (OIDC)</i>"]
    P4["4. Crear deploy.yml<br/><i>push main → lint + build + push +<br/>plan + apply (5 jobs)</i>"]
    P5["5. Crear training.yml<br/><i>workflow_dispatch + auto-train<br/>on push (action=train|promote)</i>"]
    P6["6. Crear destroy.yml<br/><i>3 modos: TEAR-DOWN / DESTROY / NUKE<br/>+ confirm textual + approval</i>"]
    P7["7. Branch protection<br/><i>main: require PR + deploy.yml verde</i>"]
    GATE["✅ Gate Parte 6<br/>Push trivial a main →<br/>deploy.yml verde → ALB 200"]

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> GATE

    style GATE fill:#d4edda,stroke:#155724
```

**Notas clave:**

- Los 3 workflows se complementan: `deploy.yml` mueve código+infra, `training.yml` dispara modelos, `destroy.yml` baja con auditoría. Falta uno y la operación tiene un hueco.
- Branch protection cierra el loop. Sin ella alguien puede pushear directo a main saltándose CI.
- No mezcles secrets y variables. TODO es `gh variable set` (no `gh secret set`) porque OIDC reemplaza access keys. Si te ves tentado a setear un AWS secret, parate — el role IAM necesita más permisos.
- Tiempo total: ~2-3h escribir los 3 workflows + debug primeros runs.

**Referencias:** paso 1 → 6.0 · paso 2 → 6.1 · paso 3 → 6.2 · paso 4 → 6.3 · paso 5 → 6.4 · paso 6 → 6.5 · paso 7 → 6.6.

> **Gotcha Parte 6**: error `Could not assume role` en el primer run → el `sub` del trust policy no matchea `repo:<org>/<repo>:*`. Revisar #3.11.2: `var.github_org` y `var.github_repo` deben matchear EXACTO el repo en GitHub. Commit: `feat(ci): consolidar 3 workflows GHA (deploy + training + destroy) con OIDC`.

## 6.0 Modelo de trust

En Parte 2 sección 2.5 creaste el OIDC provider; en Parte 3 sección 3.11 el modulo `cicd` creo 2 roles que confian en ese provider para tu `org/repo`:

- `ml-training-gha-deploy` — workflows que tocan infra (terraform apply, push ECR).
- `ml-training-gha-train` — workflows que solo invocan Lambda dispatcher.

ARNs en outputs:

```bash
terraform -chdir=infra/envs/prod output gha_deploy_role_arn
terraform -chdir=infra/envs/prod output gha_train_role_arn
```

## 6.1 Filosofia: Task = single source of truth

Toda la logica vive en `Taskfile.yml` + `tasks/*.yml`. Los workflows son thin wrappers: autentican via OIDC, setean `TF_VAR_*` y llaman `task X`.

```
Si `task X` corre OK en tu laptop → corre OK en CI.
Si rompe en CI → rompe en local con `task X`.
```

### 6.1.1 Matriz de decision: donde vive cada operacion

| Operacion | Herramienta | Por que |
|---|---|---|
| Crear VPC, RDS, Batch queue, Lambda, alarmas | **Terraform** (`infra/modules/*`) | Declarativo, idempotente, state |
| Build + push imagen a ECR | **Task** (`task ecr:build-all`) | Imperativo, encadena docker + ECR login + tag dual |
| Submit Batch job + polling hasta SUCCEEDED | **Task** (`task batch:train`) | Encadena `lambda invoke` + `jq` + `describe-jobs` con loop |
| Apply Terraform en oleadas + smoke | **Task** (`task deploy`) | Orquesta oleadas A+B+C |
| Wake / sleep RDS+Fargate | **Task** (`task wake`/`sleep`) | `ops:up` idempotente con polling RDS+ALB |
| Disparar entreno bajo demanda | **GHA workflow_dispatch** | UI con dropdown, accesible sin AWS CLI |
| Approval antes de deploy o promote | **GHA `environment: production`** | Solo GHA tiene approval gates |
| Cron L-V 08-12 PET encender/apagar | **EventBridge + Lambda** (`scheduler.py`) | Serverless, sin runner |
| Quality gate MAPE + A/B vs Production | **Task** (`task ops:promote`) llamado por GHA | Logica reutilizable local |
| Apagar todo con confirmacion textual | **GHA workflow** (`destroy.yml`) + **Task** | Confirmacion + approval son features GHA; logica en Task |

### 6.1.2 Anti-patrones (cuando NO meter en Taskfile)

| Anti-patron | Donde va en su lugar |
|---|---|
| Task que solo corre en CI (ej. `gha:upload-artifact`) | Step inline en el workflow |
| Task wrapper trivial de 1 linea (ej. `docker:build` = `docker build .`) | Eliminar |
| Logica que deberia ser Terraform (ej. `aws ec2 create-vpc`) | Modulo Terraform |

Regla: si un task no (a) encadena >=2 cosas, (b) usa vars computadas, o (c) agrega idempotencia/polling — no justifica el namespace.

## 6.2 Variables y secrets de GitHub

Settings → Secrets and variables → Actions. **Variables** (no secret):

| Nombre | Valor |
|---|---|
| `AWS_REGION` | `us-east-1` |
| `AWS_GHA_DEPLOY_ROLE_ARN` | `arn:aws:iam::<account>:role/ml-training-gha-deploy` |
| `AWS_GHA_TRAIN_ROLE_ARN` | `arn:aws:iam::<account>:role/ml-training-gha-train` |
| `ECR_TRAINER` | `<account>.dkr.ecr.us-east-1.amazonaws.com/ml-training` |
| `MLFLOW_ALB_DNS` | output de terraform `alb_dns` (sin `http://`) |
| `PROJECT` | `ml-training` |
| `ALERT_EMAIL` | email para SNS (`TF_VAR_alert_email`) |
| `CONSUMER_ORG` | org del repo consumer (`TF_VAR_consumer_org`) |
| `CONSUMER_REPO` | nombre del repo consumer (`TF_VAR_consumer_repo`) |

**Secrets**: ninguno (OIDC remplaza access keys).

```bash
# Prereqs: `source scripts/prod.env` (sección 3.5) para $ACCOUNT_ID y terraform state
# inicializado en infra/envs/prod (para los `terraform output -raw`).
gh variable set AWS_REGION              -b "us-east-1"
gh variable set AWS_GHA_DEPLOY_ROLE_ARN -b "$(terraform -chdir=infra/envs/prod output -raw gha_deploy_role_arn)"
gh variable set AWS_GHA_TRAIN_ROLE_ARN  -b "$(terraform -chdir=infra/envs/prod output -raw gha_train_role_arn)"
gh variable set ECR_TRAINER             -b "${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/ml-training"
gh variable set MLFLOW_ALB_DNS          -b "$(terraform -chdir=infra/envs/prod output -raw alb_dns)"
gh variable set PROJECT                 -b "ml-training"
gh variable set ALERT_EMAIL             -b "abantodca@gmail.com"
gh variable set CONSUMER_ORG            -b "abantodca"
gh variable set CONSUMER_REPO           -b "ml_serving"
```

`ALERT_EMAIL`, `CONSUMER_ORG`, `CONSUMER_REPO` deben estar **antes** del primer `infra-apply`, sino Terraform falla con `variable required value not provided`.

## 6.3 `deploy.yml` — lint + build + plan + apply

Trigger: push a `main`, PR a `main`, `workflow_dispatch`. Cinco jobs:

| Job | Disparo | Que hace |
|---|---|---|
| `changes` | siempre | `dorny/paths-filter@v3` setea outputs `infra` (toco `infra/**`) y `trainer` (toco `src/**`, `main.py`, `Dockerfile`, `requirements.txt`). |
| `lint` | siempre | `task lint && task infra:validate`. Sin `test` — ver [ADR-008](docs/adr/ADR-008-ci-sin-tests-todavia.md). |
| `build-and-push` | push main **AND** `trainer == 'true'` | Asume `gha-deploy`, `task ecr:build IMG=trainer`. |
| `terraform-plan` | PR **AND** `infra == 'true'` | Asume `gha-deploy`, `task infra:plan` comentado en el PR via `github-script`. |
| `infra-apply` | push main **AND** `infra == 'true'` **AND** lint OK | `environment: production` (approval). `task deploy` con `TF_VAR_trainer_image_tag=sha-<sha>`. |

> **3 decisiones no obvias del workflow**:
> - **`if: always()` en `infra-apply`**: el job depende de `build-and-push`, que puede haberse skipped (PR sin cambios al trainer). Sin `always()`, GHA marca `infra-apply` como skipped. Con `always()` + check explicito `(success || skipped)`, el apply corre aunque solo cambie infra.
> - **`cancel-in-progress` condicional**: en PRs queremos cancelar runs viejos (ahorra CI minutes); en push a main NO (un apply a medias deja state inconsistente).
> - **Build directo, no via `task ecr:build-all`**: solo el trainer se rebuildea por commit; mlflow + reports se actualizan via bump manual de TAG (cambios infrecuentes).

### 6.3.1 Archivo completo `.github/workflows/deploy.yml`

```yaml
name: Deploy

# Estrategia thin: cada job autentica via OIDC y delega en `task X`.
# Jobs y disparos:
#   lint            -> SIEMPRE (push, PR, manual). Sin job test (ver ADR-008).
#   build-and-push  -> solo push a main (publica trainer en ECR con sha + latest)
#   terraform-plan  -> solo PR a main con cambios en infra/** (comenta plan en PR)
#   infra-apply     -> solo push a main (orquesta task deploy con approval)

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch: {}

permissions:
  id-token: write
  contents: read
  pull-requests: write

concurrency:
  group: deploy-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

jobs:
  changes:
    runs-on: ubuntu-latest
    outputs:
      infra:   ${{ steps.filter.outputs.infra }}
      trainer: ${{ steps.filter.outputs.trainer }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            infra:
              - 'infra/**'
              - '.github/workflows/deploy.yml'
            trainer:
              - 'src/**'
              - 'scripts/**'
              - 'main.py'
              - 'Dockerfile'
              - 'requirements.txt'
              - 'requirements-dev.txt'
              - 'pyproject.toml'

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.13', cache: 'pip' }
      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: 1.6.6 }
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - name: Install Python deps
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      # Single source of truth: las mismas tasks que un dev corre local.
      - run: task lint
      - run: task infra:validate

  build-and-push:
    needs: [lint, changes]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main' && needs.changes.outputs.trainer == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_GHA_DEPLOY_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - run: task ecr:build IMG=trainer
        env:
          AWS_DEFAULT_REGION: ${{ vars.AWS_REGION }}
          PROJECT: ${{ vars.PROJECT }}
      - run: echo "::notice title=Pushed::${{ vars.ECR_TRAINER }}:sha-$(git rev-parse --short=12 HEAD)"

  terraform-plan:
    needs: [lint, changes]
    if: github.event_name == 'pull_request' && needs.changes.outputs.infra == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_GHA_DEPLOY_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: 1.6.6 }
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - name: task infra:plan (tee a tfplan.txt)
        id: plan
        env:
          AWS_DEFAULT_REGION: ${{ vars.AWS_REGION }}
          PROJECT: ${{ vars.PROJECT }}
          TF_VAR_alert_email:   ${{ vars.ALERT_EMAIL }}
          TF_VAR_github_org:    ${{ github.repository_owner }}
          TF_VAR_github_repo:   ${{ github.event.repository.name }}
          TF_VAR_consumer_org:  ${{ vars.CONSUMER_ORG }}
          TF_VAR_consumer_repo: ${{ vars.CONSUMER_REPO }}
        run: |
          set -o pipefail
          task infra:plan 2>&1 | tee tfplan.txt
        continue-on-error: true
      - uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const planOutput = fs.readFileSync('tfplan.txt', 'utf8').slice(-60000);
            const body = `### Terraform plan\n\n\`\`\`\n${planOutput}\n\`\`\`\n*exit code: ${{ steps.plan.outcome }}*`;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });
      - if: steps.plan.outcome == 'failure'
        run: exit 1

  infra-apply:
    needs: [lint, changes, build-and-push]
    if: |
      always() &&
      github.event_name == 'push' && github.ref == 'refs/heads/main' &&
      needs.lint.result == 'success' &&
      (needs.build-and-push.result == 'success' || needs.build-and-push.result == 'skipped') &&
      needs.changes.outputs.infra == 'true'
    runs-on: ubuntu-latest
    environment: production
    concurrency:
      group: tf-apply-${{ github.ref }}
      cancel-in-progress: false
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_GHA_DEPLOY_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: 1.6.6 }
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - name: task deploy (oleadas A+B+C)
        env:
          AWS_DEFAULT_REGION:       ${{ vars.AWS_REGION }}
          PROJECT:                  ${{ vars.PROJECT }}
          TF_VAR_alert_email:       ${{ vars.ALERT_EMAIL }}
          TF_VAR_github_org:        ${{ github.repository_owner }}
          TF_VAR_github_repo:       ${{ github.event.repository.name }}
          TF_VAR_consumer_org:      ${{ vars.CONSUMER_ORG }}
          TF_VAR_consumer_repo:     ${{ vars.CONSUMER_REPO }}
          TF_VAR_trainer_image_tag: sha-${{ github.sha }}
        run: task deploy
      - id: out
        working-directory: infra/envs/prod
        run: |
          echo "alb_dns=$(terraform output -raw alb_dns)"                       >> $GITHUB_OUTPUT
          echo "tracking_uri=$(terraform output -raw tracking_uri)"             >> $GITHUB_OUTPUT
          echo "job_queue_spot=$(terraform output -raw job_queue_spot)"         >> $GITHUB_OUTPUT
          echo "dispatcher=$(terraform output -raw dispatcher_function_name)"   >> $GITHUB_OUTPUT
      - run: |
          {
            echo "## Despliegue aplicado"
            echo ""
            echo "| Recurso | Valor |"
            echo "|---|---|"
            echo "| MLflow UI | ${{ steps.out.outputs.tracking_uri }} |"
            echo "| Reports | http://${{ steps.out.outputs.alb_dns }}/reports/ |"
            echo "| Artifacts | http://${{ steps.out.outputs.alb_dns }}/artifacts/ |"
            echo "| Batch queue Spot | ${{ steps.out.outputs.job_queue_spot }} |"
            echo "| Lambda dispatcher | ${{ steps.out.outputs.dispatcher }} |"
            echo ""
            echo "**Trainer image tag:** \`sha-${{ github.sha }}\`"
          } | tee -a "$GITHUB_STEP_SUMMARY"
```

> **Extension App Stack (API + UI)** — al materializar este `deploy.yml`, sumar
> el `api/` + `ui/` (ya soportado por `task ecr:build IMG=api|ui` y `ecr:build-all`,
> que hoy construye **5 imagenes**). Cambios minimos sobre el YAML de arriba:
> 1. **`changes` filter**: agregar `api: ['api/**', 'src/**']` (la imagen api COPIA
>    el `src/` raiz) y `ui: ['ui/**']`.
> 2. **`build-and-push`**: agregar pasos condicionados — `task ecr:build IMG=api`
>    (si `changes.api`) y `task ecr:build IMG=ui` (si `changes.ui`).
> 3. **`infra-apply`**: ampliar el gate a `(infra || trainer || api || ui)` y
>    exportar, junto a `TF_VAR_trainer_image_tag`, los tags del app stack con el
>    **mismo sha12** que usa `ecr:build` (evita el mismatch 40 vs 12 chars):
>    ```bash
>    SHA12=$(git rev-parse --short=12 HEAD)
>    export TF_VAR_trainer_image_tag="sha-$SHA12"
>    export TF_VAR_api_image_tag="sha-$SHA12"
>    export TF_VAR_ui_image_tag="sha-$SHA12"
>    task deploy   # Ola A storage -> Ola B build 5 imgs -> Ola C apply (mlflow/reports/api/ui/batch/...)
>    ```
> 4. **Outputs del summary**: agregar `ui_url` y `api_docs_url` (nuevos outputs de
>    `infra/envs/prod/outputs.tf`). El rol `gha-deploy` ya tiene `ecr:*`, asi que
>    el push a los repos `*-api` y `*-ui` no requiere cambios IAM.

> **Configuracion previa en GitHub** (UNA sola vez): Settings → Environments → New environment → `production` → Required reviewers → agregate. Sin esto, `infra-apply` arranca sin approval.

> **Gotcha #6.3 (deploy)**: `infra-apply` con `AccessDenied for sts:AssumeRoleWithWebIdentity` → trust policy del rol `gha-deploy` no permite `repo:<org>/<repo>:*`. Test rápido: push trivial a `main` que no toque infra ni trainer (solo `lint` corre, resto skipped). Commit: `feat(ci): deploy workflow consolidado`.

## 6.4 `training.yml` — train + auto-train + promote

Trigger: `workflow_dispatch` (UI manual) **o** `workflow_run` cuando `Deploy` completa con success. Cinco jobs:

| Job | Disparo | Que hace |
|---|---|---|
| `detect` | siempre | Decide modo. Si `workflow_dispatch`: usa `inputs.action`. Si `workflow_run`: `git diff` entre el SHA y el ultimo `Deploy success` anterior (via `gh run list`) — si toco `src/`, `main.py`, `Dockerfile`, `scripts/`, `requirements.txt`: `mode=train`; sino `skip`. |
| `wake-services` | `mode == 'train'` | `task ops:up` (idempotente). Outputs `mlflow_was_up=true/false`. |
| `train` | `mode == 'train'` | `task batch:train VARIETIES=... TUNING=... WAIT=true`. |
| `cool-down-and-stop` | `mode == 'train'` **AND** `mlflow_was_up == 'false'` | `task ops:down COOLDOWN=600`. **Solo apaga si nosotros lo levantamos**. |
| `promote` | `mode == 'promote'` | `environment: production` (approval). `task ops:promote MODEL_NAME=... VERSION=... MAX_MAPE=...`. |

> **Decisiones no obvias**:
> - **`permissions: actions: read`**: el job `detect` corre `gh run list --workflow Deploy --status success` para encontrar el BASE_SHA correcto (cubre push de N commits y squash merges; `HEAD~1` fallaba).
> - **Cool-down condicional**: si un humano ya prendio MLflow para mirar dashboards y dispara un train manual, no queremos apagarlo al terminar.
> - **`train` NO hace `aws batch submit-job` directo**: seguridad. El rol `gha-train` solo tiene `lambda:InvokeFunction` sobre `dispatcher`. La validacion de payload (varieties whitelist, tuning whitelist) vive en Lambda.

### 6.4.1 Archivo completo `.github/workflows/training.yml`

```yaml
name: Training

# Workflow consolidado. Tres modos via input `action`:
#   action=train    -> entrena variedades elegidas (con wake/cool-down si MLflow esta apagado)
#   action=promote  -> promueve un modelo a Production (3 gates: MAPE absoluto + A/B + approval)
# Tambien auto-trigger:
#   - workflow_run "Deploy" success -> auto-train si push toco trainer

on:
  workflow_dispatch:
    inputs:
      action:     { type: choice, options: [train, promote], default: train, required: true }
      varieties:  { type: string, default: 'all', description: '[train] CSV (POP,JUPITER) o "all"' }
      tuning:     { type: choice, options: [smoke, dev, prod, prod_xl], default: prod }
      model_name: { type: string, description: '[promote] ej: rnd-forest-POP' }
      version:    { type: string, description: '[promote] version a promover' }
      max_mape:   { type: string, default: '20', description: '[promote] umbral MAPE %' }

  workflow_run:
    workflows: ["Deploy"]
    types: [completed]
    branches: [main]

permissions:
  id-token: write
  contents: read
  actions:  read  # gh run list (detect.BASE_SHA en auto-train)

concurrency:
  group: training-${{ github.event.inputs.action || 'auto' }}
  cancel-in-progress: false

jobs:
  detect:
    runs-on: ubuntu-latest
    if: github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'success'
    outputs:
      mode:      ${{ steps.decide.outputs.mode }}
      varieties: ${{ steps.decide.outputs.varieties }}
      tuning:    ${{ steps.decide.outputs.tuning }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          ref: ${{ github.event.workflow_run.head_sha || github.sha }}
      - id: decide
        env: { GH_TOKEN: "${{ github.token }}" }
        run: |
          set -e
          if [[ "${{ github.event_name }}" == "workflow_dispatch" ]]; then
            echo "mode=${{ inputs.action }}"                  >> $GITHUB_OUTPUT
            echo "varieties=${{ inputs.varieties || 'all' }}" >> $GITHUB_OUTPUT
            echo "tuning=${{ inputs.tuning || 'prod' }}"      >> $GITHUB_OUTPUT
            exit 0
          fi

          # workflow_run: BASE_SHA = ultimo Deploy success anterior
          # (cubre push multi-commit y squash merges)
          HEAD_SHA="${{ github.event.workflow_run.head_sha }}"
          PREV_SHA=$(gh run list --workflow Deploy --branch main --status success --limit 20 --json headSha \
            --jq "[.[] | select(.headSha != \"$HEAD_SHA\")] | .[0].headSha" 2>/dev/null || echo "")

          if [ -n "$PREV_SHA" ] && [ "$PREV_SHA" != "null" ]; then
            BASE_SHA="$PREV_SHA"
          else
            BASE_SHA="${HEAD_SHA}^"
          fi

          CHANGED=$(git diff --name-only "$BASE_SHA" "$HEAD_SHA" 2>/dev/null || echo "")
          if echo "$CHANGED" | grep -qE '^(src/|main\.py|Dockerfile|requirements\.txt|scripts/)'; then
            echo "mode=train"      >> $GITHUB_OUTPUT
            echo "varieties=all"   >> $GITHUB_OUTPUT
            echo "tuning=prod"     >> $GITHUB_OUTPUT
          else
            echo "mode=skip"       >> $GITHUB_OUTPUT
            echo "::notice::Push no toco trainer, skip auto-train"
          fi

  wake-services:
    needs: detect
    if: needs.detect.outputs.mode == 'train'
    runs-on: ubuntu-latest
    outputs:
      mlflow_was_up: ${{ steps.wake.outputs.was_up }}
    steps:
      - uses: actions/checkout@v4
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_GHA_TRAIN_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      # Same logica que `task ops:up` local. STATUS_FILE comunica was_up al cool-down.
      - id: wake
        env:
          MLFLOW_ALB_DNS: ${{ vars.MLFLOW_ALB_DNS }}
          PROJECT: ${{ vars.PROJECT }}
          STATUS_FILE: /tmp/wake-status
        run: |
          task ops:up
          echo "was_up=$(cat $STATUS_FILE)" >> $GITHUB_OUTPUT

  train:
    needs: [detect, wake-services]
    if: needs.detect.outputs.mode == 'train'
    runs-on: ubuntu-latest
    timeout-minutes: 480
    steps:
      - uses: actions/checkout@v4
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_GHA_TRAIN_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - run: task batch:train VARIETIES="${{ needs.detect.outputs.varieties }}" TUNING="${{ needs.detect.outputs.tuning }}" WAIT=true
        env:
          AWS_DEFAULT_REGION: ${{ vars.AWS_REGION }}
          PROJECT: ${{ vars.PROJECT }}

  cool-down-and-stop:
    needs: [detect, wake-services, train]
    if: |
      always() && needs.detect.outputs.mode == 'train' &&
      needs.wake-services.result == 'success' &&
      needs.wake-services.outputs.mlflow_was_up == 'false'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_GHA_TRAIN_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - run: task ops:down COOLDOWN=600
        env:
          AWS_DEFAULT_REGION: ${{ vars.AWS_REGION }}
          PROJECT: ${{ vars.PROJECT }}

  promote:
    needs: detect
    if: needs.detect.outputs.mode == 'promote'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_GHA_TRAIN_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - run: task ops:promote MODEL_NAME="${{ inputs.model_name }}" VERSION="${{ inputs.version }}" MAX_MAPE="${{ inputs.max_mape }}"
        env:
          MLFLOW_ALB_DNS: ${{ vars.MLFLOW_ALB_DNS }}
```

Uso desde la UI: Actions → Training → Run workflow. Para entrenar: `action=train`, `varieties=POP`, `tuning=smoke`. Para promover: `action=promote`, `model_name=rnd-forest-POP`, `version=5`. Espera "Waiting for review" y aproba.

> **Gotcha #6.4 (training)**: sin `permissions: actions: read`, `gh run list` falla con 403 silencioso y cae al fallback `${HEAD_SHA}^` que solo cubre single-commit pushes. Test: `gh workflow run training.yml -f action=train -f varieties=POP -f tuning=smoke` → `detect` debe resolver `mode=train`.

## 6.5 `destroy.yml` — 3 modos (TEAR-DOWN / DESTROY / NUKE)

Trigger: **solo `workflow_dispatch`**.

| Modo | Que destruye | Que preserva | Reversible con |
|---|---|---|---|
| **TEAR-DOWN** | Modulos volatiles (mlflow, reports, batch, lambdas, monitoring, scheduler, cicd, consumer_iam) | S3 + ECR + network + tfstate + OIDC | `task rebuild` (~20 min, modelos intactos). Costo restante: ~$8/mes (S3). |
| **DESTROY** | TODOS los modulos administrados (incluye storage). Vacia buckets versionados + purga ECR antes del `terraform destroy`. | tfstate bucket + OIDC provider | Re-crear via `task deploy`. Costo: $0/mes. |
| **NUKE** | DESTROY + tfstate bucket + OIDC provider. Cuenta limpia. | Nada | Re-bootstrap desde cero (Parte 2). |

### 6.5.1 Doble salvaguarda

```
┌─ Salvaguarda 1: input textual ────────────────────────────┐
│  El campo "confirmar" requiere escribir literalmente:     │
│                                                           │
│     DESTRUIR-ML-TRAINING                                  │
│                                                           │
│  El `if:` matchea string-exacto. Cualquier variacion      │
│  (lowercase, abreviado, espacios) -> conclusion=skipped.  │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─ Salvaguarda 2: environment approval ─────────────────────┐
│  `environment: production` requiere que un reviewer       │
│  clickee "Approve and deploy" antes de que arranque.      │
│  Ultima ventana para cancelar si elegiste mal el modo.    │
└───────────────────────────────────────────────────────────┘
```

### 6.5.2 Archivo completo `.github/workflows/destroy.yml`

```yaml
name: Destroy

# Tres modos via input `modo`:
#   TEAR-DOWN -> destroy volatiles. Preserva S3 + ECR + network + tfstate + OIDC.
#                Reversible con `task rebuild`. Costo restante: ~$8/mes (S3).
#   DESTROY   -> terraform destroy de TODOS los modulos (incluye storage). Vacia
#                buckets versionados + purga ECR antes. Preserva tfstate + OIDC.
#   NUKE      -> DESTROY + borra tfstate bucket + OIDC. Cuenta limpia.
#
# Doble salvaguarda: input textual exacto + environment approval.

on:
  workflow_dispatch:
    inputs:
      confirmar:
        description: 'Escribe exactamente: DESTRUIR-ML-TRAINING'
        required: true
        type: string
      modo:
        description: 'TEAR-DOWN (preserva storage) | DESTROY (borra storage, preserva tfstate+OIDC) | NUKE (borra TODO)'
        required: true
        default: TEAR-DOWN
        type: choice
        options: [TEAR-DOWN, DESTROY, NUKE]

permissions:
  id-token: write
  contents: read

concurrency:
  group: destroy-${{ github.ref }}
  cancel-in-progress: false

jobs:
  destroy:
    if: ${{ inputs.confirmar == 'DESTRUIR-ML-TRAINING' }}
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_GHA_DEPLOY_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: 1.6.6 }
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }

      # Capturar Lambdas ANTES del destroy (log groups los crea AWS, no TF, asi que sobreviven)
      - id: lambdas
        run: |
          FNS=$(aws lambda list-functions \
            --query 'Functions[?starts_with(FunctionName, `ml-training-`)].FunctionName' \
            --output text || true)
          echo "fns=${FNS}" >> $GITHUB_OUTPUT

      - name: TEAR-DOWN (volatiles, preserva storage + network)
        if: ${{ inputs.modo == 'TEAR-DOWN' }}
        env:
          AWS_DEFAULT_REGION:   ${{ vars.AWS_REGION }}
          PROJECT:              ${{ vars.PROJECT }}
          TF_VAR_alert_email:   ${{ vars.ALERT_EMAIL }}
          TF_VAR_github_org:    ${{ github.repository_owner }}
          TF_VAR_github_repo:   ${{ github.event.repository.name }}
          TF_VAR_consumer_org:  ${{ vars.CONSUMER_ORG }}
          TF_VAR_consumer_repo: ${{ vars.CONSUMER_REPO }}
        run: task --yes teardown

      - name: DESTROY (todos los modulos, preserva tfstate + OIDC)
        if: ${{ inputs.modo == 'DESTROY' }}
        env:
          AWS_DEFAULT_REGION:   ${{ vars.AWS_REGION }}
          PROJECT:              ${{ vars.PROJECT }}
          TF_VAR_alert_email:   ${{ vars.ALERT_EMAIL }}
          TF_VAR_github_org:    ${{ github.repository_owner }}
          TF_VAR_github_repo:   ${{ github.event.repository.name }}
          TF_VAR_consumer_org:  ${{ vars.CONSUMER_ORG }}
          TF_VAR_consumer_repo: ${{ vars.CONSUMER_REPO }}
        run: task --yes destroy

      - name: NUKE (TODO + tfstate + OIDC)
        if: ${{ inputs.modo == 'NUKE' }}
        env:
          AWS_DEFAULT_REGION:   ${{ vars.AWS_REGION }}
          PROJECT:              ${{ vars.PROJECT }}
          TF_VAR_alert_email:   ${{ vars.ALERT_EMAIL }}
          TF_VAR_github_org:    ${{ github.repository_owner }}
          TF_VAR_github_repo:   ${{ github.event.repository.name }}
          TF_VAR_consumer_org:  ${{ vars.CONSUMER_ORG }}
          TF_VAR_consumer_repo: ${{ vars.CONSUMER_REPO }}
        run: task --yes nuke

      - name: Limpiar log groups Lambda (cleanup post-destroy)
        if: ${{ always() && inputs.modo != 'TEAR-DOWN' && steps.lambdas.outputs.fns != '' }}
        run: |
          for FN in ${{ steps.lambdas.outputs.fns }}; do
            LG="/aws/lambda/${FN}"
            if aws logs describe-log-groups --log-group-name-prefix "$LG" \
                 --query "logGroups[?logGroupName=='$LG'].logGroupName" --output text \
                 | grep -q "$LG"; then
              aws logs delete-log-group --log-group-name "$LG"
              echo "Borrado: $LG"
            fi
          done

      - name: Verificacion post-operacion
        if: ${{ always() }}
        run: |
          echo "=== EC2 con tag Project=ml-training ==="
          aws ec2 describe-instances \
            --filters "Name=tag:Project,Values=ml-training" \
                      "Name=instance-state-name,Values=running,pending,stopping,stopped" \
            --query 'Reservations[].Instances[].InstanceId' --output text || true
          echo ""
          echo "=== RDS ==="
          aws rds describe-db-instances \
            --query "DBInstances[?contains(DBInstanceIdentifier,'ml-training')].DBInstanceIdentifier" \
            --output text || true
          echo ""
          echo "=== Fargate services ==="
          aws ecs list-services --cluster ml-training-cluster \
            --query 'serviceArns' --output text 2>/dev/null || echo "(cluster destruido)"
          echo ""
          echo "=== ALB ==="
          aws elbv2 describe-load-balancers \
            --query "LoadBalancers[?contains(LoadBalancerName,'ml-training')].LoadBalancerName" \
            --output text || true
          echo ""
          echo "=== ECR ==="
          aws ecr describe-repositories \
            --query "repositories[?contains(repositoryName,'ml-training')].repositoryName" \
            --output text 2>/dev/null || echo "(sin repos)"
          echo ""
          echo "=== S3 buckets ==="
          aws s3 ls | grep ml-training || echo "(sin buckets)"
          echo ""
          echo "=== OIDC provider ==="
          aws iam list-open-id-connect-providers \
            --query "OpenIDConnectProviderList[?contains(Arn, 'token.actions.githubusercontent.com')].Arn" \
            --output text || echo "(sin OIDC provider)"
```

> **Errores tipicos del destroy**:
>
> | Sintoma | Causa | Solucion |
> |---|---|---|
> | `BucketNotEmpty` | Bucket S3 con versioning + objetos no vaciados | `task destroy` ya purga con `--all-versions`. Si persiste, re-correr. |
> | `RepositoryHasImages` (ECR) | ECR repo con imagenes y `force_delete=false` | `task destroy` ya borra antes. Verificar `force_delete=true` en `module.storage`. |
> | `DependencyViolation: Network interface in use` | Lambdas con ENIs activos (AWS los libera lento) | Esperar 15 min y re-correr. |
> | Workflow status=skipped sin razon | `confirmar` no matcheo exacto | Texto debe ser `DESTRUIR-ML-TRAINING` (case-sensitive, sin espacios). |

> **Checkpoint**: probar la salvaguarda textual sin destruir nada:
> ```bash
> # Texto incorrecto -> debe skippear
> gh workflow run destroy.yml -f confirmar=destruir -f modo=TEAR-DOWN
> gh run list --workflow=destroy.yml --limit 1   # esperado: skipped
> ```
> NUNCA pruebes DESTROY o NUKE la primera vez salvo que ya hayas terminado el proyecto.

## 6.6 Branch protection

```bash
gh api "repos/${GITHUB_OWNER}/ml_training/branches/main/protection" -X PUT --input - <<EOF
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["Deploy / lint"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1
  },
  "restrictions": null
}
EOF
```

El context `Deploy / lint` corresponde al par (`name: Deploy`, job id `lint`) del workflow consolidado. Cuando agregues `tests/`, actualizar a `["Deploy / lint", "Deploy / test"]` (ver ADR-008).

---

# Parte 7 — Promotion gate (extendido)

> [!IMPORTANT]
> Esta es la **única vía oficial** de promover un modelo a `Production`.
> No hay `mlflow.set_model_version_alias("Production")` manual permitido —
> todo pasa por `task ops:promote` (que valida MAPE absoluto) y
> opcionalmente por el workflow `training.yml action=promote` (que añade
> A/B contra Production actual + approval humano via GitHub Environment).
> Saltarse el gate rompe trazabilidad y el rollback automático.

La Parte 6 sección 6.4 (`training.yml action=promote`) ya implementa el gate basico (MAPE umbral + A/B + approval).
Esta Parte 7 documenta el ciclo completo del modelo y cuando se promueve.

## Mapa del camino — Parte 7

Parte 7 define cuándo un modelo deja de ser experimental y entra a Production. No agregás infra ni código — usás el workflow que ya creaste en Parte 6, sección 6.4 con `action=promote`.

**Prerrequisitos:**

- `training.yml` operativo (Parte 6, sección 6.4): `gh workflow list` muestra "Training".
- Al menos 1 run en MLflow Registry: `mlflow search registered-models --filter "name='rnd-forest-POP'"` lista ≥1 versión.
- GitHub Environment "production" (Parte 6, sección 6.3): Settings → Environments → production con required reviewers.

```mermaid
flowchart TD
    PUSH([push a main])
    DEPLOY["deploy.yml<br/>build :sha + apply"]
    TRAIN["training.yml action=train<br/>→ run en MLflow stage=None"]
    REVIEW{{"Review manual<br/>data scientist mira<br/>/reports/&lt;variety&gt;/"}}
    STAGING["transition None → Staging<br/><i>UI MLflow o API</i>"]
    PROMOTE["training.yml action=promote"]
    G1{"Gate 1<br/>MAPE &lt; max_mape?"}
    G2{"Gate 2<br/>mejora vs Production?"}
    G3{"Gate 3<br/>approval Environment"}
    PROD["transition Staging → Production<br/><i>auto-archive de versión vieja</i>"]
    ABORT[abort]
    ROLLBACK["ROLLBACK<br/>mlflow transition-stage<br/>--version &lt;vieja&gt;<br/>--stage Production"]

    PUSH --> DEPLOY --> TRAIN --> REVIEW --> STAGING --> PROMOTE --> G1
    G1 -->|sí| G2
    G2 -->|sí| G3
    G3 -->|aprobado| PROD
    G1 -->|no| ABORT
    G2 -->|no| ABORT
    G3 -->|rechazado| ABORT
    PROD -.problema en producción.-> ROLLBACK

    style PROD fill:#d4edda,stroke:#155724
    style ABORT fill:#f8d7da,stroke:#721c24
    style ROLLBACK fill:#fff3cd,stroke:#856404
```

**Notas clave:**

- Los 3 gates son AND, no OR. Cualquier gate que falle aborta el promote.
- El `None → Staging` lo hace un humano mirando el dashboard. NO hay automatización ahí a propósito — el dashboard de residuos muestra bias que MAPE no captura.
- El `Staging → Production` es semi-automático: gates 1-2 son código (`task ops:promote`); gate 3 (approval) es humano. Audit log queda en GHA.
- Rollback NO requiere re-training. La versión vieja sigue en Registry (archived, no deleted). Una llamada `transition-stage --version <vieja>` la restaura.

**Referencias:** ciclo de vida → 7.1 · gates → 7.2 · approval humano → 7.3 · rollback → 7.4 · workflow `action=promote` → Parte 6, sección 6.4.

> **Gotcha Parte 7**: olvidar "transition a Staging" antes de `action=promote`. El gate filtra por `stage=Staging` — si el modelo está en `None`, el workflow dice "no candidate found".

## 7.1 Ciclo de vida de un modelo

```
[push a main]
       │
       ▼
deploy.yml: lint + build + push :sha-abc123 a ECR + plan/apply Terraform
       │
       ▼
[manual trigger training.yml (action=train) en GitHub UI]
       │
       ▼
Lambda dispatcher -> Batch -> trainer corre con :sha-abc123
       │
       ▼
trainer loguea run a MLflow + custom metric MAPE a CloudWatch
       │
       ▼
trainer registra modelo en Registry stage "None"
       │
       ▼
[review manual: mirar dashboard /reports/<variety>/]
       │
       ▼
[transition a Staging via UI MLflow o API]
       │
       ▼
[manual trigger training.yml (action=promote) en GitHub UI]
       │
       ├─> Gate 1: MAPE < max_mape?  [si NO -> abort]
       ├─> Gate 2: mejora vs Production actual? [si NO -> abort]
       ├─> Gate 3: approval humano (GitHub Environment)
       │
       ▼
transition_model_version_stage(Production)
       │
       ▼
[modelo en Production, archive_existing_versions = true automatico]
```

## 7.2 Gates por nivel

| Stage | Gate | Quien |
|---|---|---|
| `None` -> `Staging` | Visual review del dashboard `/reports/<variety>/` (residuos, feature importance, comparacion XGB vs LGB) | Data scientist |
| `Staging` -> `Production` | Quality gate (MAPE < umbral) + A/B vs Production actual + approval | Workflow `training.yml` (action=promote) + revisor humano |
| `Production` -> `Archived` | Auto al promover una nueva version (archive_existing_versions = true) | MLflow |

## 7.3 Por que el approval humano (GitHub Environment)

Aun con gates automaticos:
- MAPE puede ser engañoso si la distribucion de los predichos cambio
  (modelo "barato" que predice todo igual gana en MAPE pero falla en
  recall extremo).
- A/B en metric absoluta no captura compliance / domain expert
  judgement.
- El approval crea audit log en GitHub (quien aprobo + cuando).

## 7.4 Rollback de un Production

Si la version promovida tiene problemas:

```bash
# Listar versiones del modelo
mlflow search registered-models --filter "name='rnd-forest-POP'"

# Transition la version vieja de vuelta a Production
mlflow models transition-stage 
    --model-name "rnd-forest-POP" 
    --version 3 
    --stage Production 
    --archive-existing
```

O via UI: MLflow → Models → `rnd-forest-POP` → seleccionar version
buena → "Transition to" → Production.

---

> **Cierre Partes 5-7.** Sistema production-grade funcional: trainer parchado
> emitiendo MAPE por variedad a CloudWatch, 3 workflows GHA consolidados
> (`deploy.yml` / `training.yml` / `destroy.yml` con OIDC + confirmación
> textual), promotion ciclo A/B documentado. **Próximos pasos**: #8 (runbook)
> → #9 (costos/modos) → #11 (troubleshooting) →
> #12 (apéndices).

---

# Parte 8 — Runbook operativo extendido

> **Por qué esta parte**: si el sistema está en producción y vos no estás,
> alguien tiene que operarlo sin leerse las 5 oleadas. Este es ese manual:
> comandos copy-paste con el "por qué" al lado, para entender qué se hace y no
> ejecutar a ciegas.

## Mapa del camino — Parte 8

Parte 8 es referencia operativa: no se lee de corrido, vas a la subsección que matchea tu situación.

**Prerrequisitos:**

- Sistema en producción (Partes 4-7 completas): `curl $ALB/` da 200.
- Acceso AWS y `gh` CLI: `aws sts get-caller-identity` y `gh auth status` ambos OK.
- Suscripción SNS confirmada (Parte 4, sección 4.7): correo "Subscription confirmed" recibido.

```mermaid
flowchart TD
    SIT([Tu situación])

    subgraph Diario[Diario / Frecuente]
        D1[Re-entrenar 1 variedad → 8.1.1]
        D2[Re-entrenar TODAS recovery → 8.1.2]
        D3[Spot vs On-Demand → 8.1.3]
        D4[Rollback imagen trainer → 8.1.4]
    end

    subgraph Semanal[Semanal / Mensual]
        S1[Tear-down vacaciones → 8.2.1]
        S2[Rebuild post-vacaciones → 8.2.2]
        S3[Subir Excel nuevo → 8.2.3]
    end

    subgraph Incidentes[Incidentes — algo rompió]
        I1[Job RUNNABLE eterno → 8.3.1]
        I2[MLflow 403 Invalid Host → 8.3.2]
        I3[RDS too many connections → 8.3.3]
        I4[Spot interrupt mid-training → 8.3.4]
        I5[Terraform state lock → 8.3.5]
        I6[S3 sync 403 → 8.3.6]
        I7[Job arrancó, MLflow apagado → 8.3.7]
        I8[Cold-start RDS lento → 8.3.8]
    end

    subgraph Lifecycle[Lifecycle del cluster]
        L1[Shutdown limpio SIGTERM → 8.4]
        L2[TEAR-DOWN reversible → 8.5]
        L3[REBUILD post-teardown → 8.6]
        L4[DESTROY irreversible → 8.7]
    end

    SIT --> Diario
    SIT --> Semanal
    SIT --> Incidentes
    SIT --> Lifecycle

    style L4 fill:#f8d7da,stroke:#721c24
    style L2 fill:#fff3cd,stroke:#856404
```

**Notas clave:**

- El orden NO es secuencial, es taxonómico. Solo corrés lo que tu situación pide.
- **TEAR-DOWN ≠ DESTROY.** Tear-down es reversible (`task rebuild`); destroy es irreversible. Confundirlos = pérdida de Model Registry.
- `task sleep` es el atajo de bajo costo: si solo querés ahorrar 1-2 noches (no semanas), apaga RDS+Fargate sin borrar nada. Rebuild desde sleep = 5 min vs 30 min desde tear-down.
- Frecuencia esperada: 8.1 semanal · 8.2 mensual · 8.3 cuando algo rompe (idealmente nunca) · 8.5-8.7 raro (vacaciones, cierre).

> **Gotcha Parte 8**: confundir `task teardown` con `task destroy`. Antes de correr cualquiera, leer el WARNING de #8.5 / #8.7 — hay confirmación textual pero leer-y-tipear sin entender lleva a pérdida de datos. Confirmar también suscripciones SNS (sino no llega aviso de `FAILED`).

## 8.1 Manual diario / mas frecuente

### 8.1.1 Re-entrenar una variedad

**Por que se hace**: data nueva subida al bucket (`aws s3 cp` del Excel
acumulado), o pediste un re-train porque cambiaste hiperparametros.

```bash
# Opcion A — via Task (preferido para humanos, polling + exit-code visible)
task batch:train VARIETIES=POP TUNING=prod

# Opcion B — via GitHub Actions UI (preferido si lo dispara alguien sin AWS CLI)
# Actions -> Train -> Run workflow -> POP / prod / wait=true

# Opcion C — via AWS CLI directo (preferido en scripts ad-hoc)
aws lambda invoke 
    --function-name ml-training-dispatcher 
    --cli-binary-format raw-in-base64-out 
    --payload '{"varieties":"POP","tuning":"prod"}' 
    /tmp/out.json
```

**Que pasa por debajo**: Lambda valida el payload (variedad esta en
allowlist, tuning es uno de los 4 validos) → boto3 `submit_job` con
queue Spot (o On-Demand si tuning=prod_xl) → Batch wakea un EC2 c6i.2xlarge
→ pull image → entrenamiento corre → champion log a MLflow → sync a S3
→ container muere → EC2 termina. Todo dura 30-60 min en prod.

### 8.1.2 Re-entrenar TODAS las variedades en un dia (recovery)

**Por que se hace**: rollback de data, hubo un bug en `cli.py` que
hizo que los runs del ultimo mes no se loggearan, o necesitas refresh
total.

```bash
# Loop: una variedad a la vez, espera completion antes de la siguiente
varieties=(POP JUPITER VENTURA SEKOYA ALLISON STELLA)
for v in "${varieties[@]}"; do
    echo "==> Retrain $v"
    task batch:train VARIETIES=$v TUNING=prod || {
        echo "WARN: $v fallo, continuando con el resto"
    }
done
```

**Por que secuencial y no paralelo**: las 6 en paralelo serian
6 × c6i.2xlarge ≈ 48 vCPUs simultaneos. Con `spot_max_vcpus=16`
default, Batch encolaria igual pero te llevarias el cap. Si queres
paralelo de verdad, subir `spot_max_vcpus=48` ANTES (via terraform.tfvars).

### 8.1.3 Spot vs On-Demand por preset (cuando elegir cual)

| Preset | Tiempo estimado | Probabilidad de Spot interrupt | Recomendacion |
|---|---|---|---|
| `smoke`  | 2-4 min  | <1% | Spot SIEMPRE |
| `dev`    | 10-20 min | <2% | Spot SIEMPRE |
| `prod`   | 30-60 min | ~5% | Spot (retry=2 cubre el caso) |
| `prod_xl`| 4-6 h    | 15-30% | **On-Demand** (forzado por dispatcher: tuning=prod_xl → queue ondemand) |

**Por que la regla**: la probabilidad de interrupcion crece con el
tiempo en Spot. Para jobs de 6h, el retry-cost (perder 5h de
computo) supera al 70% de ahorro. La logica esta en `dispatcher.py`:

```python
queue = JOB_QUEUE_ONDEMAND if tuning == "prod_xl" else JOB_QUEUE_SPOT
```

### 8.1.4 Rollback de imagen del trainer

**Por que se hace**: pushaste una version que tiene un bug y queres
volver a la anterior sin re-build.

```bash
# Listar tags del trainer en ECR
aws ecr list-images --repository-name ml-training \
    --query 'imageIds[?imageTag != null].[imageTag]' --output table

# Re-tag la version anterior como :latest
export PREV_SHA="sha-abcdef123456"   # buscar la version anterior buena
export REGION="$AWS_DEFAULT_REGION"
export ACCOUNT="$ACCOUNT_ID"
REG="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

# Pull la imagen vieja
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REG"
docker pull "${REG}/ml-training:${PREV_SHA}"
docker tag  "${REG}/ml-training:${PREV_SHA}" \
            "${REG}/ml-training:latest"
docker push "${REG}/ml-training:latest"
```

**Por que NO actualizar el job-def via Terraform en vez**: porque
Batch arranca el container con la tag puntual configurada en el job-def.
Si en `terraform.tfvars` decis `trainer_image_tag = "sha-abcdef"` y
aplicas, hace lo mismo pero te deja un audit log en el state remoto —
preferible para rollbacks de produccion (en ese caso usar la Opcion
Terraform de abajo).

```bash
# Opcion Terraform (mas auditable)
terraform -chdir=infra/envs/prod apply -target=module.batch -var=trainer_image_tag=sha-abcdef123456 -auto-approve
```

## 8.2 Manual semanal / mensual

### 8.2.1 Bajar todo para ahorrar (tear-down)

**Por que se hace**: fin de mes, vacaciones, pausa del proyecto.
Conocido como "scale to near-zero".

```bash
task teardown
# Confirmar con "y" al prompt de Task
```

**Por que el orden importa** (ver Parte 4.8.3):
1. **scheduler primero**: si esta arriba, va a re-encender RDS+Fargate
   en el proximo cron y anular el tear-down.
2. **reports + mlflow despues**: son Fargate consumidores; bajan
   primero para que el ALB no tenga targets unhealthy.
3. **batch**: drena Spot CE a 0.
4. **network**: NAT GW es el ultimo en irse — `$32/mes` ahorro
   inmediato.

### 8.2.2 Volver a levantar (rebuild)

```bash
task rebuild
```

**Por que cambia el ALB DNS**: el `aws_lb.main` se recrea con un nombre
distinto en su DNS. Si tenias bookmarks, actualizalos. Si quisieras
un DNS estable, TLS + Route 53 lo resuelve (hardening, futuro).

### 8.2.3 Subir data nueva

**Por que se hace**: cada mes (o cada cierto periodo) llega un Excel
nuevo con los datos acumulados.

```bash
export BUCKET="${PROJECT}-data-${ACCOUNT_SUFFIX}"

# Subir nuevo Excel (versiones se guardan automaticamente por
# `aws_s3_bucket_versioning` Enabled en modulo storage)
aws s3 cp data/BD_HISTORICO_ACUMULADO.xlsx "s3://${BUCKET}/BD_HISTORICO_ACUMULADO.xlsx"

# Verificar version mas reciente
aws s3api list-object-versions --bucket "$BUCKET" --prefix BD_HISTORICO_ACUMULADO.xlsx \
    --query 'Versions[0].[VersionId,LastModified,Size]' --output table

# Lanzar re-train de todas las variedades con la data nueva
for v in POP JUPITER VENTURA SEKOYA ALLISON STELLA; do
    task batch:train VARIETIES=$v TUNING=prod WAIT=false
done
```

**Por que `wait=false` aca**: 6 jobs encolados, no querras esperar
cada uno secuencial. Los jobs corren en paralelo segun
`spot_max_vcpus`. El monitoreo es via SNS (notifier publica si alguno
FAILED) o `aws batch list-jobs`.

## 8.3 Manual de incidentes

### 8.3.1 Job se quedo en RUNNABLE eternamente

**Sintoma**: `aws batch describe-jobs --jobs <id>` muestra `status =
RUNNABLE` por mas de 10 min sin pasar a `STARTING`.

**Por que pasa**: no hay capacity Spot en `us-east-1a` para
`c6i.2xlarge`, o tu quota de vCPUs esta llena.

**Que mirar**:

```bash
# 1) Estado del CE
aws batch describe-compute-environments 
    --compute-environments ml-training-ce-spot 
    --query 'computeEnvironments[0].status' --output text
# Esperado: VALID. Si dice INVALID, ver statusReason.

# 2) Quota EC2
aws service-quotas get-service-quota 
    --service-code ec2 
    --quota-code L-1216C47A 
    --query 'Quota.Value'

# 3) Estado del Spot fleet implicito (via instancias)
aws ec2 describe-spot-instance-requests 
    --filters Name=state,Values=open,active 
    --query 'SpotInstanceRequests[].[InstanceType,State,Status.Code]' --output table
```

**Fix**:
- Si quota llena: pedir aumento (Capítulo 3.4).
- Si no hay capacity Spot: cancelar el job y resubmit con
  `tuning=prod_xl` (lo manda a la queue On-Demand). O esperar.

### 8.3.2 MLflow 403 "Invalid Host header"

**Por que pasa**: MLflow 3.x rechaza requests cuyo `Host:` header no
esta en `--allowed-hosts`. El V2 ya pone `--allowed-hosts '*'` (3.5.2)
pero si lo cambiaste a una lista, y el ALB DNS no esta — boom.

**Fix**: editar `modules/mlflow/ecs.tf:container_definitions.command`
para incluir el nuevo DNS, `terraform apply -target=module.mlflow`.

### 8.3.3 RDS "too many connections"

**Por que pasa**: `db.t4g.small` (default actual) tiene maximo ~200
conexiones. Ademas del trainer de Batch (cada variedad en paralelo abre
~5, worker pool), ahora la **API** mantiene su pool a la base `forecasts`
(mismo RDS). Si lanzas muchas variedades en paralelo con la API tambien
sirviendo, podes acercarte al techo.

**Fix temporal**: cancelar jobs activos en Batch hasta que el conteo baje.
**Fix permanente**: bajar el `pool_size`/`max_overflow` de la API (SQLAlchemy)
y/o subir `rds_instance_class` a `db.t4g.medium` (4 GB, ~400 conexiones) en
`terraform.tfvars`. (Con `db.t4g.micro`, 1 GB, el techo era ~85 — por eso el
default del stack completo es `small`.)

### 8.3.4 Spot interrupt mid-training

**Sintoma**: job en FAILED con `statusReason = "Host EC2 ..."`.
**Por que pasa**: AWS necesito tu c6i.2xlarge para otro customer.
**Que pasa automaticamente**: `retry_strategy.attempts = 2` + filtro
`Host EC2*` (Parte 3.7.3). El job se re-encola en otra instancia.
**Que hace falta a mano**: nada si pasa una vez. Si pasa
sistematicamente (3+ jobs FAILED por Spot en un dia), considerar:

- Cambiar a `tuning=prod_xl` para esa variedad (queue OD).
- Ver `Best practices > Capacity` en consola Batch — quizas la AZ tiene
  presion. Cambiar `instance_type` a alternativa (`c6a.2xlarge`, `m6i.2xlarge`).

### 8.3.5 `task infra:apply` falla con state lock

**Sintoma**: `Error acquiring the state lock` con un `LockID`.

**Por que pasa**: otra invocacion de `terraform apply` esta corriendo
(o crasheo sin liberar lock). Con `use_lockfile=true` (Parte 2.1) el lock
es un objeto `envs/prod/terraform.tfstate.tflock` en el bucket de tfstate.

**Fix**:

```bash
# Ver el lock huerfano en S3 (deberia haber UN objeto .tflock si hay lock)
aws s3 ls "s3://${PROJECT}-tfstate-${ACCOUNT_SUFFIX}/envs/prod/" | grep tflock

# Opcional: descargar el objeto y leer su JSON (contiene LockID, Who, Operation)
aws s3 cp "s3://${PROJECT}-tfstate-${ACCOUNT_SUFFIX}/envs/prod/terraform.tfstate.tflock" -

# Si el ID corresponde a un proceso que ya murio (laptop crasheada),
# forzar unlock:
task infra:force-unlock LOCK_ID=<LOCK_ID>
```

### 8.3.6 S3 sync del trainer falla con 403

**Por que pasa**: el job-role no tiene `s3:PutObject` sobre
`artifacts_bucket`, o el bucket esta en otra region.

**Que mirar**:

```bash
# Inline policy del job role
aws iam list-role-policies --role-name ml-training-job-role
aws iam get-role-policy --role-name ml-training-job-role --policy-name <name>
```

**Fix**: en Parte 3.7.2 el inline policy `job_s3` ya cubre PutObject
sobre el bucket. Si falla, verificar que el bucket creado matchea
`var.artifacts_bucket`.

### 8.3.7 Job arranca pero MLflow esta apagado (fuera de ventana)

**Por que pasa**: lanzaste un train fuera de L-V 08-12 PET. El cron
de stop apago MLflow (Fargate desired_count=0). El trainer intenta
conectar a `tracking_uri` y obtiene timeout.

**Fix manual** (sin parche de auto-wake integrado):

```bash
task wake
# Esperar 5-8 min hasta que ALB responde 200
task batch:train VARIETIES=POP
```

> **Solucion permanente**: implementar auto-train on push con
> wake + cool-down). Ese workflow invoca Lambda scheduler antes del
> train y apaga 10 min despues si MLflow estaba abajo. La ampliacion
> de permisos del role `gha-train` para invocar el scheduler queda pendiente en este repo.

### 8.3.8 Cold-start de RDS lento el primer request

**Por que pasa**: RDS post-`start_db_instance` tarda ~5 min en estar
disponible. El primer query desde MLflow puede demorar 10-20s extra
por warm-up de buffers.

**Que NO hacer**: no agregues timeout corto en el container — vas a
matar conexiones legitimas. **Que hacer**: el healthcheck del task
def tiene `startPeriod = 60`; aumentalo si tu RDS warmup es
consistentemente mas lento.

## 8.4 Shutdown limpio DENTRO del job de training

**Por que importa**: si Batch te interrumpe (Spot) o vos cancelas el
job, el contenedor recibe `SIGTERM`. El trainer tiene 30s para
limpiar antes de `SIGKILL`. Si en ese momento estabas en medio de
`mlflow.log_model`, el modelo queda corrupto o el run en estado
`RUNNING` para siempre.

El Dockerfile ya tiene `tini` y `STOPSIGNAL SIGTERM` (3.0.5 contracts).
`tini` reenvia el SIGTERM al child Python.

**Que falta en el codigo Python**: un handler de SIGTERM. Patch
opcional:

```python
# main.py — al inicio de main()
import signal

def _graceful_exit(signum, _frame):
    import logging
    logging.getLogger().warning("SIGTERM recibido — abortando run en limpio")
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.set_tag("mlflow.runStatus", "KILLED")
            mlflow.end_run(status="KILLED")
    except Exception:
        pass
    import sys
    sys.exit(143)   # 128 + 15 (SIGTERM)

signal.signal(signal.SIGTERM, _graceful_exit)
```

**Por que NO matar el archivo con `pkill` o `kill -9`**: SIGKILL no
es interceptable. Cualquier estado a medio escribir queda corrupto.

## 8.5 TEAR-DOWN — apagar todo preservando state + datos

> [!WARNING]
> TEAR-DOWN es **reversible** (con `task rebuild` levantás todo de
> nuevo en ~20-30 min), pero te lleva un `terraform destroy` de 5-6
> módulos. Si sólo querés ahorrar 1-2 noches, usá `task sleep`
> (apaga RDS + Fargate, costo cae a ~$15/mes mientras esté apagado) en
> vez de teardown — el rebuild a partir de sleep es 5 min, no 30.

Cuando lo uso: vacaciones de 1+ semanas, fin del mes y queres bajar el
gasto, evento de costo inesperado, pausar el proyecto.

**Que SE PRESERVA** (no se borra):

- S3 `ml-training-tfstate-XXXXXX` (Terraform state + lockfile efimero)
- S3 `ml-training-data-XXXXXX` (Excels de input)
- S3 `ml-training-artifacts-XXXXXX` (modelos serializados + reportes)
- ECR `ml-training`, `ml-training-mlflow`, `ml-training-reports` (todas las tags)
- IAM roles (gha-deploy, batch-execution, lambda-exec, ...)
- OIDC provider de GitHub
- SNS topic + suscripcion email
- EventBridge rules (vacias mientras esten apagadas)

**Que SE APAGA / BORRA temporalmente**:

- ECS Fargate services (MLflow + Reports): `desired_count = 0`
- RDS instance: **stopped** (snapshot automatico antes; se reactiva al
  arrancar)
- Batch compute environments: `desired_vcpus = 0` (no hay EC2 corriendo)
- ALB + listener: borrados (se recrean en rebuild — el DNS cambia)
- NAT Gateway: borrado ($32/mes ahorro)
- Subnets/VPC: se preservan o borran segun el flag (default: preservar)

**Costo despues del tear-down**: ~$35/mes (NAT GW $32 permanente + S3 ~$3). El tear-down preserva `module.network`; para eliminar el NAT GW usar `task destroy` o destruir `module.network` aparte.

### Comando `task teardown`

```bash
task teardown
# Pide confirmacion: Task pide "y" para proceder.
# Pasos internos (tasks/ops.yml :: teardown):
#  1. ops:down -> aborta si hay Batch jobs RUNNING; invoca la Lambda
#     ml-training-scheduler (action=stop) para apagar RDS + Fargate.
#  2. Loop `terraform destroy -target=$mod` sobre los modulos VOLATILES, en orden:
#     scheduler -> lambdas -> monitoring -> batch -> reports -> api -> ui -> mlflow -> cicd -> consumer_iam
#  network y storage son PERMANENTES: NO se destruyen (el NAT GW sigue vivo).
```

### Periodo de gracia de RDS

RDS auto-arranca despues de **7 dias** de estar stopped (limitacion AWS).
Si vas a estar fuera mas de 7 dias, dos opciones:

- **Opcion A (recomendada)**: el scheduler Lambda `ml-training-rds-keepstop`
  (Parte 3.11) detecta que RDS arranco solo y lo vuelve a parar
  automaticamente. Cron: cada 6h chequea state, si RUNNING y fuera de
  ventana lo para.
- **Opcion B**: snapshot manual + delete instance. Para rebuild, restore
  from snapshot (~10 min). Solo si vas a estar fuera 1+ mes.

## 8.6 REBUILD — volver despues de tear-down

Cuando lo uso: vuelvo de vacaciones, retomo el proyecto, necesito la UI
de MLflow para mirar runs viejos.

**Precondicion**: la cuenta tiene los recursos preservados del tear-down
(buckets S3, ECR, IAM, OIDC).

### Comando `task rebuild`

```bash
task rebuild
# Pasos internos (tasks/ops.yml :: rebuild):
#  1. terraform apply COMPLETO (sin -target): los modulos volatiles se re-crean,
#     el resto es no-op.
#  2. ops:up -> invoca la Lambda scheduler (action=start): arranca RDS (~5 min
#     cold start) + servicios Fargate de forma secuencial y espera ALB 200.
#  El ALB DNS nuevo cambia respecto al stand-up original.
```

**Tiempo**: 20-30 min, dominado por RDS cold start (5 min) + Fargate
task launch (~3 min) + ALB target registration (~2 min).

**Lo unico que cambia respecto al stand-up original**: el DNS del ALB.
Si tenías bookmark, actualizalo. (Si agregaste un dominio (TLS, hardening)
custom via Route53, el dominio sigue igual; sólo el record A apunta al
nuevo ALB.)

## 8.7 DESTROY — eliminar TODO de la cuenta AWS

> [!CAUTION]
> **Operación irreversible**. `task destroy` borra buckets S3 (incluido
> `tfstate`), repos ECR (todas las tags), la instancia RDS (Model Registry
> completo), los IAM roles y el OIDC provider. Pide confirmación textual
> **dos veces** (dos prompts y/n de Task). Antes de correrlo: hacer los 3
> backups de la sub-sección siguiente (export Registry → JSON, snapshot
> manual RDS, export tfstate) — sin ellos no hay forma de recuperar.

Cuando lo uso: cierre del proyecto, migracion a otra cuenta, hard reset
para empezar de cero.

> **Nota importante**: esta seccion solo aplica si ya estuviste operando
> el sistema por un tiempo y queres salir. En el stand-up inicial (Parte
> 1.1) no hay nada que respaldar — los backups de abajo presuponen que
> tenes modelos registrados, una RDS poblada y un Terraform state con
> recursos. Si no tenes nada de eso, salta directo al comando.

**ATENCION**: esto borra:

- TODOS los buckets S3 (incluido tfstate — perdes el historial de cambios
  de Terraform; los modelos en `s3://artifacts/`; los Excels en `s3://data/`)
- TODOS los repos ECR (perdes todas las tags / versiones de imagenes)
- RDS instance + snapshots automaticos (perdes el Model Registry entero —
  todas las versiones, transitions, tags)
- IAM roles + OIDC provider (proximo deploy desde GHA va a fallar hasta
  recrear)
- VPC + NAT + ALB + Fargate + Batch + Lambdas + EventBridge + SNS +
  CloudWatch alarms + log groups

**ANTES de destruir, hacer 3 backups manuales**:

```bash
# Pre-requisito: tener un bucket de archive FUERA del proyecto (otra cuenta
# o, como minimo, otro nombre que NO sea destruido por este `task destroy`).
# Crearlo a mano si no existe (este bucket vive aparte del state):
export ARCHIVE_BUCKET="${PROJECT}-archive-${ACCOUNT_SUFFIX}"
if ! aws s3api head-bucket --bucket "${ARCHIVE_BUCKET}" 2>/dev/null; then
  aws s3api create-bucket --bucket "${ARCHIVE_BUCKET}" --region "${AWS_DEFAULT_REGION}"
  aws s3api put-bucket-versioning --bucket "${ARCHIVE_BUCKET}" \
    --versioning-configuration Status=Enabled
fi

# (1) Export del Model Registry a JSON (corre con MLflow encendido)
export MLFLOW_TRACKING_URI="http://<ALB-DNS>/"
mlflow models search > artifacts/model-registry-export.json
aws s3 cp artifacts/model-registry-export.json \
  "s3://${ARCHIVE_BUCKET}/ml-training-$(date +%Y-%m-%d)/"

# (2) Snapshot manual de RDS (queda independiente del instance)
aws rds create-db-snapshot \
  --db-instance-identifier ml-training-mlflow \
  --db-snapshot-identifier "ml-training-mlflow-final-$(date +%Y-%m-%d)"

# (3) Export del Terraform state como ultimo backup
cd infra/envs/prod
terraform state pull > /tmp/tfstate-final-backup.json
aws s3 cp /tmp/tfstate-final-backup.json \
  "s3://${ARCHIVE_BUCKET}/ml-training-$(date +%Y-%m-%d)/"
cd ../../..
```

> **Por que un bucket separado**: `task destroy` borra todos los buckets
> del proyecto (`ml-training-data-*`, `ml-training-artifacts-*`,
> `ml-training-tfstate-*`). Si pusieras los backups ahi, se borrarian
> en el mismo apply. El `${PROJECT}-archive-${ACCOUNT_SUFFIX}` queda
> intacto porque Terraform no lo conoce.

### Comando `task destroy`

```bash
task destroy
# Pide confirmacion: Task pide "y" en dos prompts.
# Pasos internos (Taskfile :: destroy):
#  1. ops:down (apaga scheduler/RDS/Fargate; aborta si hay jobs RUNNING).
#  2. Vaciar buckets versionados data + artifacts (incluye versions + delete markers).
#  3. purge_ecr de los 5 repos (ml-training, -mlflow, -reports, -api, -ui).
#  4. infra:destroy -> terraform destroy total de envs/prod (incluye S3 + RDS).
# NOTA: el bucket tfstate y el OIDC provider NO los borra `task destroy`;
#       para eso esta `task nuke` (destroy + empty tfstate + delete_oidc).
```

**Tiempo**: 30-45 min, dominado por el vaciado de buckets versionados
(cada modelo es ~10 MB con N versions).

---

# Parte 9 — Costos detallados

> **Por qué esta parte**: AWS no muestra un total previsto antes de gastar; sin
> entender el desglose te llevás sorpresa en la factura. Acá está el número
> realista por modo de operación y los `dials` para bajarlo.

## Mapa del camino — Parte 9

Parte 9 es referencia: tablas de costos para razonar "cuánto cuesta cada modo" y "cuánto bajo apagando X". Llegás a la tabla que responde tu pregunta.

**Prerrequisitos:**

- Infra desplegada o conocida (sabés cuántas variedades, Multi-AZ on/off, etc.).
- AWS Cost Explorer habilitado: Billing → Cost Explorer → Enabled (una vez por cuenta).
- Tags `Project` aplicados (Parte 3, sección 3.2): `aws ce get-cost-and-usage --group-by Type=TAG,Key=Project` muestra `ml-training`.

```mermaid
flowchart TD
    Q([Tu pregunta de costos])

    subgraph Base[Presupuesto base]
        B1["Operar normal L-V 08-12<br/>≈ <b>$75/mes</b><br/>→ 9.1"]
    end

    subgraph Alt[Escenarios alternativos]
        A1["Hibernado (tear-down) ≈ $35/m<br/>(NAT GW permanente)<br/>→ 9.2"]
        A2["24/7 ≈ $195/m<br/>→ 9.2"]
        A3["No-NAT (VPC endpoints) ≈ $36/m<br/>→ 9.2"]
        A4["Multi-AZ RDS +$13/m<br/>→ 9.2"]
        A5["TLS + custom domain +$1/m<br/>→ 9.2"]
    end

    subgraph Delta[Delta por modo lifecycle]
        D1["Matriz 13 items × 3 modos<br/>(STAND-UP/TEAR-DOWN/DESTROY)<br/>→ 9.3"]
    end

    subgraph Futuro[Optimizaciones futuras]
        F1["VPC endpoints (post 60d)<br/>S3 Intelligent-Tiering<br/>ECR scan policies<br/>Fargate Spot MLflow<br/>→ 9.4"]
    end

    Q --> Base
    Q --> Alt
    Q --> Delta
    Q --> Futuro

    style B1 fill:#d1ecf1,stroke:#0c5460
```

**Notas clave:**

- Los números son estimados con asunciones específicas (10 trainings/mes, 5 GB S3, 80h/mes scheduler). Tus números reales se desvían según variedades, frecuencia de re-train y tráfico ALB+NAT.
- El item que más pesa: NAT GW $32/mes 24×7. Si te molesta, ir a 9.4.1 (VPC endpoints, hardening futuro). Aplica si tráfico NAT > 80 GB/mes.
- Los "(futuro)" en 9.4 son intencionales: cada optimización cuesta horas de ingeniería. Re-leer a los 60 días de operación real.

> **Gotcha Parte 9**: confundir "operando normal" con "24/7". El default lockeado es scheduler L-V 08-12 = 80h/mes de Fargate; pasar a 24/7 (scheduler OFF) duplica el número. Validar con `aws events describe-rule --name ml-training-start` y comparar con Cost Explorer (±20% de #9.1).

## 9.1 Escenario lockeado: scheduler L-V 08-12 PET — ~$75/mes

Asume: 80 horas/mes del stack encendido (4h × 5 dias × ~4 semanas; el scheduler
prende/apaga MLflow + Reports + **API + UI** juntos), 10 trainings/mes promedio
(1h cada uno), 5 GB de S3 storage total, ~3.5 GB de ECR images (5 repos).

| Item | Calculo | Mensual (USD) |
|---|---|---|
| S3 (5 GB Standard) | 5 × $0.023 | $0.12 |
| S3 (versiones no-current con lifecycle 90d) | ~10 GB | $0.23 |
| ECR (~3.5 GB, 5 repos) | 3.5 × $0.10 | $0.35 |
| RDS db.t4g.small (80h/mes; hostea MLflow + forecasts) | 80 × $0.032 | $2.56 |
| RDS storage (20 GB gp3) | 20 × $0.115 | $2.30 |
| Fargate MLflow (2 vCPU, 4 GB, 80h) | 80 × ($0.04048 × 2 + $0.004445 × 4) | $7.90 |
| Fargate Reports (0.5 vCPU, 1 GB, 80h) | 80 × ($0.04048 × 0.5 + $0.004445 × 1) | $1.97 |
| Fargate API (1 vCPU, 2 GB, 80h) | 80 × ($0.04048 × 1 + $0.004445 × 2) | $3.95 |
| Fargate UI (0.5 vCPU, 1 GB, 80h) | 80 × ($0.04048 × 0.5 + $0.004445 × 1) | $1.97 |
| ALB (24/7) | 720h × $0.0225 | $16.20 |
| ALB LCU | <0.5 LCU promedio | $0.50 |
| NAT Gateway (24/7) | 720h × $0.045 | $32.40 |
| NAT egress data | 10 GB | $0.45 |
| EC2 Spot c6i.2xlarge (10 jobs × 1h) | 10 × $0.102 | $1.02 |
| Lambdas (negligible) | ~1000 invocs/mes | $0.10 |
| EventBridge | ~120 events/mes | $0.10 |
| SNS | ~10 publishes/mes | $0.01 |
| CloudWatch Logs (14d retention, 1 GB) | 1 × $0.50 | $0.50 |
| CloudWatch Custom Metrics (N MAPE + 3 base; N=6 ejemplo) | (N+3) × $0.30 | $2.70 |
| CloudWatch PutMetricData API | ~10k calls/mes | $0.01 |
| Data transfer (ALB out a Internet) | 5 GB | $0.45 |
| **Total** | | **~$75** |

> **Notas del calculo**:
> - **Custom Metrics $0.30/serie**: cobro por metrica unica (combinacion
>   namespace + name + dimensiones). MAPE con `dim=variety` genera **N
>   series** (una por variedad en `var.varieties`) + 3 metricas base de
>   Batch/ALB = **N+3 series**. El ejemplo usa N=6 (default actual) → 9
>   series × $0.30 = $2.70. Si cambias `varieties`, este item escala
>   lineal: cada variedad nueva = +$0.30/mes. Si tu trafico de
>   PutMetricData superara 1M calls/mes, sumar $0.01/1000 calls
>   (despreciable aca).
> - **Data transfer ALB out 5 GB**: trafico de la UI MLflow + reports + el
>   dashboard Streamlit (`/app/`) + el Swagger de la API (`/docs`) hacia tu
>   browser desde Internet. NO se cuenta el egress AWS→AWS (Batch→S3, ECS→RDS,
>   UI→API por service discovery) porque va por la VPC interna sin costo. El
>   item "NAT egress 10 GB" de arriba ya cubre lo que SALE de la VPC.
> - **App stack (API + UI) = ~+$6/mes** sobre el escenario training-only: las
>   dos tasks Fargate ($3.95 + $1.97) prendidas las mismas 80h del scheduler,
>   mas el delta de RDS micro→small (~+$1.1) por hostear tambien `forecasts`.
>   Si activas `api_preload_models=true` o subis `api_memory` a 4096, la linea
>   "Fargate API" escala proporcional (4 GB × 80h × $0.004445 ≈ +$1.4/mes).

**Por que el numero crecio de ~$68 a ~$75**: la version training-only daba ~$68
(sin API ni UI). El App stack de la Capa 4.5 suma ~$6 (2 Fargate + RDS small).
$75 es realista con el monorepo completo (training + serving) del V2.

## 9.2 Comparativa con escenarios alternativos

| Escenario | Cambio vs default | Costo total/mes | Cuando elegirlo |
|---|---|---|---|
| **Hibernado** | tear-down (sección 8.5; NAT GW queda) | ~$35 | Pausa de 1+ semana |
| **Default (lockeado)** | Scheduler L-V 08-12 PET | **~$75** | Operacion normal |
| **24/7** | Scheduler OFF, stack (MLflow+Reports+API+UI) + RDS siempre on | ~$195 | Equipo distribuido multi-timezone |
| **No-NAT** | VPC endpoints en vez de NAT GW (hardening, futuro) | ~$36 | Trafico NAT < 10 GB/mes |
| **Multi-AZ RDS** | RDS Multi-AZ (hardening, futuro) | +$13 sobre default | Compliance / SLA estricto |
| **TLS + Custom Domain** | Route 53 zone + ACM cert (hardening, futuro) | +$1 sobre default | Exposicion publica |

## 9.3 Matriz de costos por modo de lifecycle

Tabla resumen que cruza los 4 modos (STAND-UP / TEAR-DOWN / DESTROY) con
los recursos: util para razonar "cuanto bajo apagando X" antes de
ejecutar `task teardown` o `task destroy` (los modos en si estan
documentados en secciones 8.5 a 8.7).

| Recurso | STAND-UP (operando) | TEAR-DOWN (hibernado) | DESTROY (vacio) |
|---|---|---|---|
| S3 (todos los buckets, ~5 GB) | $0.12 | $0.12 | $0 |
| ECR (5 repos, ~3.5 GB) | $0.35 | $0.35 | $0 |
| RDS db.t4g.small (L-V 08-12 PET ≈ 80h/mes) | $2.56 | $0 (stopped) | $0 |
| RDS allocated storage (20 GB gp3) | $2.30 | $2.30 (storage solo) | $0 |
| ECS Fargate MLflow (2 vCPU, 4 GB, 80h/mes) | $7.90 | $0 | $0 |
| ECS Fargate Reports (0.5 vCPU, 1 GB, 80h/mes) | $1.97 | $0 | $0 |
| ECS Fargate API (1 vCPU, 2 GB, 80h/mes) | $3.95 | $0 | $0 |
| ECS Fargate UI (0.5 vCPU, 1 GB, 80h/mes) | $1.97 | $0 | $0 |
| ALB | $16 | $0 (borrado) | $0 |
| NAT Gateway (24/7) | $32 | $0 (borrado) | $0 |
| Batch EC2 (Spot c6i.2xlarge, ~10 jobs/mes × 1h) | $3 | $0 | $0 |
| Lambdas (negligible) | $0.10 | $0.10 | $0 |
| EventBridge / SNS / CloudWatch | $0.30 | $0.30 | $0 |
| Data transfer (NAT egress + ALB) | $5 | $0 | $0 |
| **Total mensual** | **~$75** | **~$3** | **$0** |

> La suma directa de esta tabla da ~$72; los ~$75 reales (que matchean
> sección 9.1) incluyen items consolidados aca: ALB LCU + CloudWatch Custom
> Metrics (9 series MAPE+base × $0.30) + S3 lifecycle de versiones. La columna
> hibernado NO cambia con el App stack: el scheduler escala API+UI a 0 igual
> que MLflow/Reports. Tabla pensada como **delta entre modos**, no como suma
> auditable — para esa ver sección 9.1.

Si queres bajar mas el modo operando, ver sección 9.4 (S3 lifecycle +
Intelligent Tiering, ECR scan policies). Los VPC endpoints eliminan el
NAT GW (~$32 menos, hardening futuro).

## 9.4 Optimizaciones adicionales ((futuro))

**Por que se llaman "futuras" en vez de aplicarlas dia 1**: cada una
tiene un costo de ingenieria o un trade-off. Aplicarlas dia 1 te frena
sin que aporten valor hasta que la operacion tenga datos.

### 9.4.1 VPC endpoints en vez de NAT (hardening, futuro)

Cambia $32/mes NAT GW por $7/mes en 4 endpoints (S3, ECR, CloudWatch,
STS). Net: $25/mes menos. **Por que no se aplica dia 1**: agrega
~100 lineas de Terraform y rompe si te falta un endpoint para algun
servicio que el trainer use indirectamente. Aplicarlo despues, cuando
tenes lista de servicios consumidos en CloudTrail.

### 9.4.2 S3 Intelligent-Tiering

Auto-tier `artifacts/` despues de 90d a Standard-IA (-50% storage).
**Por que no dia 1**: tu volumen S3 es ~5 GB. Ahorro real: <$0.5/mes.
No vale la pena hasta que pases los 100 GB.

### 9.4.3 ECR scan policies

Borrar imagenes con vulnerabilidades CVSS > 7. **Por que no dia 1**:
genera ruido (la imagen base puede tener CVEs que upstream parchea
en semanas). Aplicar despues de la primera vuelta.

### 9.4.4 Fargate Spot para MLflow

50-70% off Fargate, pero interrupcion = MLflow caido. **Por que no
dia 1**: MLflow es path-critical para CI/CD; downtime mid-day rompe
tu workflow. Usar solo en envs/dev.

---

# Parte 11 — Troubleshooting (catalogo)

> **Por qué esta parte**: errores que YA pasaron (en V1 o durante el desarrollo
> del V2), cada uno con síntoma, causa y fix concreto. Si tu error no está acá,
> `aws logs tail /aws/batch/ml-training --follow` es siempre el primer paso.

## Mapa del camino — Parte 11

Parte 11 es 1 tabla larga (17 entradas). Este mapa agrupa por fase del flujo: buscás por contexto, vas directo a la fila.

**Antes de mirar la tabla — diagnóstico genérico** (3 pasos, 1 minuto):

1. `aws sts get-caller-identity` — ¿estás autenticado en la cuenta correcta?
2. `echo "$AWS_REGION $PROJECT $ACCOUNT_SUFFIX"` — ¿tenés las env vars de Capítulo 3, sección 3.5?
3. `aws logs tail /aws/batch/ml-training --follow` (o el log group relevante) — ¿qué dice CloudWatch?

Si los 3 dan info esperada y aún así falla, ir al catálogo:

```mermaid
flowchart TD
    ERR([Algo rompió])

    subgraph TF[Terraform / infra apply]
        T1["#1 init: failed credentials"]
        T1b["#1b init: Missing region (terminal nueva)"]
        T2["#2 apply cuelga en RDS"]
        T3["#3 Lambda 'cannot be assumed' (race AWS)"]
        T10["#10 destroy network DependencyViolation"]
        T12["#12 secret cannot be deleted"]
    end

    subgraph BT[Batch / training]
        B4["#4 Job SUBMITTED 5+ min"]
        B5["#5 Trainer out of memory"]
        B15["#15 dispatcher 500 'variedades no permitidas'"]
        B17["#17 dispatcher jobId=null (IAM SubmitJob/TagResource)"]
        B83["cross-ref Parte 8: 8.3.1, 8.3.4"]
    end

    subgraph ML[MLflow / RDS / reports]
        M6["#6 mlflow API request failed"]
        M7["#7 /reports/POP/ da 404"]
        M13["#13 RDS arrancó solo tras 7d"]
        M83["cross-ref Parte 8: 8.3.2, 8.3.3, 8.3.7"]
    end

    subgraph CI[CI/CD / ECR / OIDC]
        C8["#8 GHA Could not assume role"]
        C11["#11 ECR token expired"]
        C14["#14 Workflow Unable to locate credentials"]
    end

    subgraph OBS[Observabilidad / costos]
        O9["#9 Alarma mape-pop no dispara"]
        O16["#16 NAT GW caro"]
    end

    ERR --> TF
    ERR --> BT
    ERR --> ML
    ERR --> CI
    ERR --> OBS
```

**Notas clave:**

- Buscás por fase, no por síntoma literal. "Rompe en CI" → CI/CD. "Rompe al hacer apply" → Terraform.
- Cross-refs a Parte 8, sección 8.3: hay overlap entre incidentes operativos (Parte 8) y errores catalogados (Parte 11). Parte 8 tiene más contexto del "por qué"; Parte 11 es lookup rápido. Si matchea ambos, leer Parte 8 primero.
- Si tu error no está en ningún bloque: los logs son la verdad. `aws logs tail /aws/batch/ml-training --follow` (training), `/aws/lambda/ml-training-dispatcher` (Lambdas), `/ecs/ml-training/mlflow` (MLflow).
- Tabla viva: V1 → V2 pasó de 8 a 17 filas. Cada error nuevo que tarda > 30 min en resolverse merece fila nueva.

> **Gotcha Parte 11**: asumir que el error es de la fase donde aparece. Ej: "ECR push 403" puede ser (a) token expirado (fila #11) o (b) IAM del role `gha-deploy` sin `ecr:PutImage` (#3.11.2). Mismo síntoma, fix distinto — cuando dudes, mirá IAM antes de re-loguear. Pre-check: si `aws sts get-caller-identity` falla, no tiene sentido buscar en la tabla. Commit (al agregar fila): `docs: troubleshooting #N — <síntoma corto>`.

| # | Sintoma | Causa | Fix |
|---|---|---|---|
| 1 | `terraform init` falla con `failed to retrieve credentials` | `AWS_PROFILE` no exportado o profile inexistente | `aws sts get-caller-identity` para verificar; `$AWS_PROFILE = "..."`. |
| 1b | `terraform init` falla con `Missing region value` (o `bucket/key/use_lockfile` vacios) | Terminal nueva sin re-exportar sección 3.5; tipico al volver al proyecto despues de cerrar WSL | `echo "$AWS_REGION $PROJECT $ACCOUNT_SUFFIX"` — si alguno esta vacio, re-correr el bloque export de sección 3.5 antes del init. |
| 2 | `terraform apply` cuelga en `aws_db_instance.mlflow: Still creating...` por 15+ min | RDS create normal toma 8-12 min; si > 15 min hay un problema | Re-intentar; chequear subnet group AZs distintas; ver eventos en AWS console RDS |
| 3 | `aws_lambda_function: InvalidParameterValueException: The role defined for the function cannot be assumed by Lambda` | El role recien creado tarda en propagarse | `sleep 10 && terraform apply` (race-condition AWS) |
| 4 | Batch job en `SUBMITTED` por 5+ min | El CE no escalo todavia | Normal; espera. Si > 15 min, revisar quotas (8.3.1) |
| 5 | Trainer en log: `Error 28: out of memory` | OOM al cargar el Excel grande con todas las variedades | Bajar `parallel_varieties` a 1 (default), o subir job-def `memory` |
| 6 | Trainer en log: `mlflow.exceptions.MlflowException: API request failed` | MLflow apagado / cold-start | Esperar 5 min; verificar `aws_ecs_service.mlflow` desired=1 |
| 7 | Dashboard `/reports/POP/` da 404 | Sync de S3 a nginx no corrio | Esperar 60s (loop sync); o entrar al container: `aws ecs execute-command ...` |
| 8 | GitHub Actions falla con `Could not assume role` | Trust policy del role no incluye el sub `repo:org/repo:*` | Re-aplicar `module.cicd` con `github_org` + `github_repo` correctos |
| 9 | Alarma `ml-training-mape-pop` no dispara aunque MAPE es alto | El trainer no publica a CloudWatch | Verificar Parte 5 patch aplicado + IAM `cloudwatch:PutMetricData` |
| 10 | `terraform destroy` en `module.network` falla con `DependencyViolation` | Algo en otra capa todavia usa la VPC (ALB, ENI huerfana) | `terraform destroy` modulo por modulo en orden inverso |
| 11 | ECR push falla con `denied: Your authorization token has expired` | Token tiene 12h de validez | `aws ecr get-login-password ... | docker login ...` de nuevo |
| 12 | `aws_secretsmanager_secret`: `cannot be deleted before the recovery window` | AWS deja 7 dias minimum para recovery | Usar `aws secretsmanager delete-secret --force-delete-without-recovery` |
| 13 | RDS arranco solo despues de 7 dias stopped | Hard limit AWS — auto-arranque post-7d | Scheduler keepstop (3.10.2) lo re-para cada 6h |
| 14 | Workflow `training.yml` falla con `Unable to locate credentials` | Permissions `id-token: write` faltante en YAML | Agregar `permissions: { id-token: write, contents: read }` al job |
| 15 | Lambda dispatcher 500: `variedades no permitidas: ['xyz']` | Variety no esta en `varieties_allowed` del terraform | Agregar a `var.varieties_allowed` y `terraform apply -target=module.lambdas` |
| 16 | NAT GW cuesta mas de lo esperado | Trafico NAT alto (mucho egress S3 cross-region o ECR pulls grandes) | Activar VPC endpoints (sección 9.4.1) |
| 17 | `task batch:smoke` se cuelga mostrando `POP None`; al invocar la Lambda, `StatusCode=200` pero `body.jobId=null` y `errorMessage: AccessDeniedException ... batch:SubmitJob` (o `batch:TagResource`) | El rol `ml-training-dispatcher` no autoriza el SubmitJob: (a) la Lambda pasa la job-def por NAME, que autoriza contra el ARN **sin** `:revision` y no matchea el patron `...:*`; (b) `submit_job` pasa `tags={...}`, lo cual exige `batch:TagResource` aparte. El `200` es de la *invocacion*, no del resultado — leer `body.errorMessage`, no el StatusCode | Agregar al inline policy (#3.9) el ARN de job-def **sin** revision (junto al `:*`) y la accion `batch:TagResource`. Aplicar: `task infra:apply TARGET='module.lambdas.aws_iam_role_policy.dispatcher'`. Verificar con `aws lambda invoke ... | jq '.body.jobId'` |

---

# Parte 12 — Apendices

## Mapa del camino — Parte 12

Parte 12 es 4 apéndices independientes. Este mapa te dice cuál abrir según lo que necesitás.

```mermaid
flowchart TD
    Q([Tu pregunta])

    Q --> QA{"¿No conozco<br/>un término?"}
    QA -->|AWS/DevOps| A1["Apéndice A.1<br/>ALB, SLR, OIDC, NAT GW..."]
    QA -->|Estadística/EDA| A2["Apéndice A.2<br/>BP, DW, ADF, VIF, MI, PSI"]

    Q --> QB{"¿Defender decisión<br/>ante no-técnico?"}
    QB --> B1["Apéndice B.1<br/>Por qué MLOps"]
    QB --> B2["Apéndice B.2<br/>Por qué Terraform"]
    QB --> B3["Apéndice B.3<br/>Por qué Task"]
    QB --> B4["Apéndice B.4<br/>Por qué GHA + OIDC"]

    Q --> QD{"¿Qué archivos<br/>crea la guía?"}
    QD --> D["Apéndice C<br/>Árbol de ~80 archivos<br/>(checklist post-stand-up)"]

    style D fill:#d1ecf1,stroke:#0c5460
```

**Notas clave:**

- Apéndice A (Glosario) es lookup rápido, no lectura corrida.
- Apéndice B (Conceptos) es para audiencia no-técnica o juniors. Si sos experto MLOps, B está por completitud — pasalo a quien pregunta el "por qué".
- Apéndice C es el mejor checklist post-stand-up: después de Partes 2-4, verificar que cada archivo del árbol existe.

## Apendice A — Glosario (referencia rapida)

### A.1 AWS / DevOps

| Termino | Que es en 1 frase |
|---|---|
| **ALB** | Application Load Balancer (L7) de AWS. Aca expone MLflow + Reports en :80. |
| **AWS Batch** | Servicio que corre jobs ephemeros en EC2. Autoescala 0↔N segun cola. |
| **CE (Compute Environment)** | En Batch, define las EC2 disponibles (tipo, Spot/OD, min/max vCPUs). |
| **ECR** | Elastic Container Registry. Registry Docker privado de AWS. |
| **ECS Fargate** | Elastic Container Service modo serverless. No manejas EC2 subyacentes. |
| **EventBridge** | Bus de eventos de AWS. Cron, eventos de servicios, custom. |
| **IaC** | Infrastructure as Code (Terraform aca). |
| **MLflow** | Tracking + Registry de modelos ML. |
| **NAT GW** | NAT Gateway: permite que subnets privadas salgan a Internet. $32/mes. |
| **OIDC** | OpenID Connect. GitHub Actions lo usa para asumir IAM roles sin secrets. |
| **RDS** | Relational Database Service. Postgres managed (backend de MLflow). |
| **SLR (Service Linked Role)** | IAM role que AWS crea solo para que un servicio funcione. |
| **Spot** | EC2 70% mas barato pero interrumpible con 2 min de aviso. |
| **State (Terraform)** | JSON con mapping HCL ↔ recursos reales. Vive en S3; lock nativo S3 (use_lockfile, sin DynamoDB). |
| **STS** | Security Token Service. Emite credenciales temporales (asume role). |

### A.2 Estadistica / EDA (usado en `task eda`)

| Termino | Test / metrica que mide | Como leerlo |
|---|---|---|
| **BP** | **Breusch–Pagan** test de heterocedasticidad sobre residuos | `p < 0.05` → varianza no constante; modelo lineal asume lo contrario |
| **DW** | **Durbin–Watson** statistic de autocorrelacion en residuos | Valores cerca de `2` → sin autocorrelacion; `<1.5` o `>2.5` → señal autoregresiva sin modelar |
| **ADF** | **Augmented Dickey–Fuller** test de raiz unitaria (estacionariedad) | `p < 0.05` → serie estacionaria; relevante si features tienen tendencia temporal |
| **VIF** | **Variance Inflation Factor** entre features | `VIF > 10` → multicolinealidad severa; candidato a drop / regularizacion |
| **MI** | **Mutual Information** feature-target | Alto MI con bajo coef linear → no-linealidad capturable por trees (XGB/LGB) |
| **PSI** | **Population Stability Index** entre distribuciones train/test | `PSI > 0.25` → drift severo; el split de validacion no representa training |

> **Cuando se ejecuta cada uno**: el `task eda VARIETIES=POP` los corre
> todos como diagnostico **previo al training**. Si BP o DW disparan,
> la heuristica del proyecto es seguir entrenando — los modelos
> tree-based no asumen homocedasticidad ni residuos i.i.d. —, pero
> registrar el flag en el run de MLflow como tag (`eda_bp_flag=true`).
> PSI alto si justifica abortar: el modelo aprendido sobre training no
> va a generalizar al test.

## Apendice B — Conceptos fundamentales (lectura opcional)

> **Skip si sos experto MLOps.** Este apendice arranca desde nivel
> 101 ("por que MLOps y no un script en CRON") y existe por
> completitud — para un lector con experiencia previa en CI/CD,
> tracking servers, model registries y promotion gates, el contenido
> es redundante. Volver aca solo si necesitas defender la eleccion
> de tooling ante alguien no tecnico.

### B.1 Por que MLOps y no "el script de Python que corre en CRON"

3 problemas que MLOps resuelve:
1. **Reproducibilidad**: tu modelo no se reproduce desde una notebook
   sin Docker + git SHA + MLflow tag.
2. **Auditabilidad**: cuando un cliente reclama una prediccion mala
   de hace 3 meses, sabes que version del modelo predijo y con que
   data.
3. **Observabilidad**: MAPE silencioso es bug silencioso. Alarmas
   automaticas convierten degradacion en pager.

### B.2 Por que Terraform y no CDK / Pulumi

- **Terraform**: declarativo, HCL, ecosistema gigante, multi-cloud.
  Estandar de industria.
- **CDK**: TypeScript/Python imperativo que GENERA CloudFormation.
  AWS-only.
- **Pulumi**: TypeScript/Python imperativo nativo multi-cloud.

Terraform gana en V2 porque:
- Tu equipo es mas probable que conozca HCL que CDK.
- El state remoto en S3 + DDB es 50 lineas, no CDK Pipelines de 500.
- La modularidad HCL es mas explicita que CDK constructs.

Si en el futuro queres Pulumi (mejor IDE, types), la migracion del
state es soportada (`pulumi import`).

### B.3 Por que Task si Terraform ya orquesta

Terraform es declarativo: te dice _que_ infra existe, no _en que orden_
hacer cosas que dependen entre si. Ejemplos donde Task gana:

- "Antes de bajar Fargate, drena Batch" — orden + condicion (pre-check
  con exit 1 si hay jobs RUNNING, en `ops:down`).
- "Push ECR, despues `terraform apply -target=module.batch`" — multi-tool
  encadenado (`task deploy` hace ambos en oleadas A+B+C).
- "Si RDS esta stopped, start; espera available; despues apply" — flujo
  con polling (`ops:up` invoca scheduler.start y chequea ALB hasta 200 con timeout).

**Vs Ansible (V1 deprecated)**: ambos resuelven lo mismo, pero Task es
single-binary 10 MB vs Ansible ~200 MB con Python+pipx, sintaxis
YAML+POSIX shell mas legible que YAML+Jinja+modulos `ansible.builtin.X`,
y corre con un solo binario en Linux/WSL/macOS sin overhead de runtime
Python. (En este proyecto Task se instala dentro de WSL Ubuntu — ver
Capítulo 3.1 — para mantener un unico entorno bash a lo largo de toda la
guia.) Para un stack Docker + AWS managed services (sin
hosts EC2), Ansible es overkill.

**Vs bash/Makefile**: Task da una sola lista descubrible (`task --list`),
namespacing con `includes:`, cache de builds via `sources:` con hash
(no solo timestamps como Make), `prompt:` para destructivos, variables
con scope, y manejo de errores estructurado por task. Bash sirve para
1-2 scripts; Task para 30+ tasks organizadas por dominio.

### B.4 Por que GitHub Actions con OIDC

OIDC remplaza el caso comun de access keys de larga duracion:

- AWS access keys leakean (PRs maliciosos, dumps de logs accidentales).
- GitHub PAT leakean en clonados.
- OIDC = credenciales de 60 min firmadas por GitHub para tu sub
  (repo:org/repo:ref). Si tu repo se hackea, el atacante NO obtiene
  AWS keys.

## Apendice C — Mapa de archivos creados por la guia

Al terminar las 5 oleadas, tu repo tiene **agregados** (todo lo que
ya estaba se preserva):

```
ml_training/
├── infra/                                       # NUEVO
│   ├── bootstrap.sh                             # Parte 2.2
│   ├── bootstrap-oidc.sh                        # Parte 2.5
│   ├── envs/prod/                               # Parte 3.2 (5 archivos .tf + terraform.tfvars gitignored)
│   ├── modules/
│   │   ├── _shared/                             # 5 trust policies JSON/tftpl + README (dedupe IAM)
│   │   ├── network/                             # variables + main + security_groups + outputs (4)
│   │   ├── storage/                             # variables + main + outputs (3)
│   │   ├── mlflow/                              # split: variables + main + alb + ecs + iam + rds + outputs (7)
│   │   ├── reports/                             # variables + main + outputs (3)
│   │   ├── api/                                 # Capa 4.5 — variables + main + outputs (3)
│   │   ├── ui/                                  # Capa 4.5 — variables + main + outputs (3)
│   │   ├── batch/                               # split: variables + main + iam + outputs (4)
│   │   ├── monitoring/                          # variables + main + outputs (3)
│   │   ├── lambdas/                             # split: variables + dispatcher + notifier + outputs (4; sin main.tf)
│   │   ├── scheduler/                           # variables + main + outputs (3)
│   │   ├── cicd/                                # variables + main + outputs (3)
│   │   └── consumer-iam/                        # Patch 13.5 OPCIONAL/LEGACY (variables + main + outputs) — ver #3.11.5
│   └── lambdas/                                 # Codigo Python de las Lambdas (3 .py)
│       ├── dispatcher.py                        # Parte 3.9.5
│       ├── notifier.py                          # Parte 3.9.6
│       └── scheduler.py                         # Parte 3.10.4 (WORKDAYS_CRON + wake secuencial + app stack api/ui)
├── tasks/                                       # NUEVO (orquestacion AWS + helpers)
│   ├── infra.yml                                # Parte 4.1.3   terraform + bootstrap
│   ├── ecr.yml                                  # Parte 4.1.4   build + push 5 imagenes
│   ├── batch.yml                                # Parte 4.1.5   submit via Lambda dispatcher + smoke
│   ├── ops.yml                                  # Parte 4.1.6   lifecycle cluster + MLflow registry (Dia 2)
│   ├── local.yml                                # sección 4.6 Tramo I  helpers dev local (ensure-buckets)
│   └── lib/                                     # Parte 4.1.7   helpers bash sourceables
│       ├── batch_wait.sh                        #               polling de Batch jobs
│       ├── wake.sh                              #               wake idempotente del cluster
│       ├── mlflow_uri.sh                        #               resolver ALB DNS
│       └── nuke.sh                              #               helpers para destroy/nuke
├── Taskfile.yml                                 # MODIFICADO (includes + atajos high-level: deploy/wake/sleep/teardown/destroy/nuke/status)
├── docker/
│   ├── mlflow/Dockerfile                        # Parte 4.5.1 (Tramo I local)
│   ├── nginx-reports.conf                       # Parte 4.5.2 (Tramo I local)
│   └── reports/                                 # Parte 3.6 (Tramo II ECS: Dockerfile + nginx.conf + entrypoint.sh)
├── docker-compose.yml                           # Parte 4.5.3 (con override-friendly MLFLOW_TRACKING_URI — Patch 13.8)
├── docker-compose.override.yml.example          # Patch 13.8 — template para apuntar trainer local a MLflow prod
├── api/                                         # App stack Capa 4.5: FastAPI (api/app/ + api/Dockerfile #3.12.4 + requirements) — del Tramo I/monorepo
├── ui/                                          # App stack Capa 4.5: Streamlit (ui/app/ + ui/Dockerfile #3.12.8 + requirements) — del Tramo I/monorepo
├── .github/workflows/                           # PENDIENTE — no existe aun (ver STATUS en Parte 6)
│   ├── deploy.yml                               # Parte 6.3 — por crear
│   ├── training.yml                             # Parte 6.4 — por crear
│   └── destroy.yml                              # Parte 6.5 — por crear
├── src/                                         # MODIFICADO
│   ├── orchestration/variety_runner.py          # Parte 5.3 (emit_mape_metric tras quality gate)
│   └── utils/cloudwatch_metrics.py              # Parte 5.2 (archivo nuevo)
├── main.py                                      # OBLIGATORIO (entrypoint del trainer); signal handler en sección 8.4
├── GUIA_MLOPS_AWS_V2.md                         # esta guia
└── (resto del proyecto: README.md, pyproject.toml, requirements*.txt, .dockerignore,
    .env.example, .gitignore, .editorconfig, notebooks/, scripts/, .claude/, etc.
    NOTA: docs/adr/ NO existe aun — los ADR referenciados en la guia (e.g. ADR-008)
    son aspiracionales / por crear.)
```

Total agregado: ~80 archivos nuevos, ~3 modificados.
Total LOC agregadas: ~3500 (HCL + Python + YAML + Markdown).

**Notas sobre las customizaciones aplicadas** en este repo:
- `infra/modules/consumer-iam/` ya cableado en `envs/prod/main.tf` como
  Capa 10 (consumer-iam).
- `.github/workflows/auto-train-on-push.yml` **NO existe** como archivo
  separado: la funcionalidad esta absorbida en `training.yml` (jobs
  `detect` + `wake-services` + `train` + `cool-down-and-stop`; ver sección 6.4).
- Modificaciones a `infra/envs/prod/{main,variables,outputs}.tf` y
  `terraform.tfvars` para registrar el modulo `consumer_iam`.

---

> **Cierre Tramo II — guía completa.**
>
> Para usar esta guía desde el día 1:
> 1. Validar prereqs en **Capítulo 3**, luego ejecutar **Capítulo 4** (local).
> 2. Si nunca aplicaste nada en AWS: empezar en **Parte 2** (bootstrap).
> 3. Seguir lineal hasta **Parte 4.6** (smoke test): infra operativa y un
>    job de Batch que entrena POP end-to-end.
> 4. Después **Partes 5-7** para CI/CD + promotion (no son indispensables
>    el día 1 pero conviene activarlos en semana 2).
> 5. **Parte 8** se usa como manual cuando algo falla; **Parte 11** es
>    la primera consulta cuando ves un error.
>
> Mantenimiento de esta guia: cada vez que cambies un modulo
> Terraform o un playbook, actualizar la seccion correspondiente +
> registrar el cambio en el `CHANGELOG.md` o `git log`.
