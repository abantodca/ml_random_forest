# Guía experta — Entorno local (Docker Compose, MLflow, Taskfile)

> Tramo I del stack: levantar **todo el sistema en tu máquina** —trainer, API, UI, MLflow y
> Postgres— con `docker compose`, y entrenar una variedad de punta a punta. Es el entorno donde se
> valida el **mismo contrato de imagen** que después se publica en ECR y se ejecuta en AWS Batch.
> El smoke local reduce el riesgo, pero no garantiza por sí solo que AWS pase: IAM, red, cuotas,
> arquitectura de CPU y servicios administrados se validan otra vez en el smoke productivo.
>
> La regla que gobierna este tramo: **local y producción se diferencian por configuración, nunca por
> código**. El backend de MLflow es Postgres + S3 en los dos lados
> ([ADR-001](adr/ADR-001-mlflow-backend-postgres-s3.md)), los buckets son S3 **real** también en local
> ([ADR-003](adr/ADR-003-s3-real-sin-localstack.md)), y la imagen del trainer es la misma que corre en
> Batch. Lo único que cambia entre un tramo y el otro es a dónde apunta `MLFLOW_TRACKING_URI` y de
> dónde sale el dataset.
>
> Cuando esto funcione, seguí con [`docs/02-produccion-aws.md`](02-produccion-aws.md). El *por qué* de
> cada pieza —diagramas, componentes, costo, qué NO se usa— está en
> [`docs/03-arquitectura.md`](03-arquitectura.md).

**Cómo leer los bloques de comandos.** Salvo que se indique lo contrario, todo corre **desde la raíz
del repo, en tu máquina**. Donde importa, los bloques llevan marca: `[HOST]` es tu shell (WSL/Linux),
`[CONTENEDOR]` es dentro de un contenedor del compose (se llega con `docker compose exec`). La
distinción importa más de lo que parece: `task` orquesta desde el host pero casi todo el trabajo
ocurre adentro, y un `python main.py` corrido en el host —sin las dependencias pineadas de la
imagen— produce resultados que no son los que vas a ver en producción.

**Qué NO cubre este tramo.** Nada de AWS más allá de los dos buckets sandbox: ni Terraform, ni Batch,
ni ECS, ni CI/CD. Todo eso es [Tramo II](02-produccion-aws.md). La suite automatizada todavía es
deuda conocida; mientras no exista, este entorno es un baseline operativo, no una certificación de
producción. Lint, smoke runs y gates estadísticos no reemplazan pruebas unitarias, de integración y
de contratos. El mínimo exigible se define en [2.A](#2a-baseline-de-calidad-mlops).

## Convenciones de las dos guías

Valen igual para este documento y para [`02-produccion-aws.md`](02-produccion-aws.md).

- **Orden de lectura**: local primero, AWS después. **No avances al Tramo II sin que el smoke local
  del Capítulo 4 termine en verde.**
- **Punto de partida real**: la única asunción es que el repo tiene `src/`, `main.py`, `scripts/` y
  `requirements.txt`. Todo lo demás —`Dockerfile`, compose, `Taskfile`, `.env`— se construye en el
  Capítulo 4.
- **Capa de orquestación**: `Taskfile.yml` delega en cinco taskfiles con namespace y cinco helpers
  de shell bajo `tasks/`, que **no están versionados**. `tasks/local.yml` se crea en [#4.6.2](#462-taskslocalyml-canónico-único);
  el resto —`infra`, `ecr`, `batch`, `ops` y `lib/*.sh`— en
  [`02-produccion-aws.md` #4.1.3-4.1.7](02-produccion-aws.md). Sin esa capa **ningún** comando
  `task` funciona, ni siquiera `task --list`: `go-task` resuelve `includes:` antes de despachar.
- **Convención de comandos**: todos los bloques `bash` se ejecutan desde la raíz del repo. En
  Windows, exclusivamente desde **WSL Ubuntu** (no Git Bash, no PowerShell).
- **Una sola receta de imagen**: local y CI usan el mismo `Dockerfile`. Una reconstrucción no es
  necesariamente el mismo binario byte a byte, porque tags base y repositorios pueden cambiar. En
  producción se despliega la imagen construida por CI mediante tag SHA inmutable —idealmente por
  digest— y ese digest se registra en MLflow.
- **Convención de verificación**: cada bloque importante cierra con un comando o tabla que valida el
  estado. **Si falla, parar y resolver antes de seguir.**
- **Re-ejecutabilidad**: toda sección que produce archivos o recursos AWS es idempotente (correr N
  veces da el mismo estado). El callout `🔄 Al re-ejecutar` aparece solo cuando hay un pitfall
  específico que no esté ya en el cuerpo de la sección.
- **Convención de avisos**:
  - `> **Nota** — …` aclara el porqué de una decisión, dentro de un paso.
  - `> **Warning** — …` señala riesgos reales: pérdida de datos, costo inesperado, operación
    irreversible.
  - Para transiciones críticas entre Partes/Capítulos se usan callouts GitHub-flavored, que
    GitHub.com y la mayoría de previews renderizan como cajas coloreadas: `> [!NOTE]` (información
    complementaria), `> [!TIP]` (atajo o mejor práctica), `> [!IMPORTANT]` (paso que **no se puede
    saltear**), `> [!WARNING]` (riesgo de pérdida de tiempo o datos), `> [!CAUTION]` (operación
    **irreversible**: destroy, nuke, force-push).

---


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
MLflow Model Registry: nueva versión con tag validation_status=pending
  │ workflow_dispatch training.yml (action=promote)
  ▼
Quality gate (calidad absoluta + comparación contra alias @champion)
  │ approval humano en GitHub Environments
  ▼
MLflow Model Registry: alias @champion reasignado de forma auditable
```

> **Sobre el Model Registry en Tramo Local.** El diagrama es el flujo
> **productivo** (Tramo II). En local, `task train` loggea runs al MLflow del
> compose y sube el `.joblib` a S3 sandbox, **pero no registra el modelo en el
> Registry** — deliberado: versionar cada iteración local solo agrega ruido.
> En Tramo II el trainer registra el campeón y lo marca con
> `validation_status=pending`; el quality gate de Parte 7 reasigna el alias
> `@champion` cuando la versión pasa. Los stages `None`, `Staging` y
> `Production` están deprecados en MLflow y no forman parte del flujo nuevo.
> Para verlo en local sin esperar,
> apuntar el trainer al MLflow productivo vía `docker-compose.override.yml.example`.

### 1.5 Contrato del run MLflow

Cualquier run que el trainer escriba — local en tu laptop o productivo
en Batch — cumple un **contrato fijo**, alineado con Cap 2. Quien
revise el run después (vos, un colega, el quality gate de Parte 7) lo
puede auditar end-to-end sin inspeccionar el código del trainer:

**Tags de trazabilidad** (set en `mlflow.set_tags({...})` por `run_metadata.py` + `single_run.py`):

| Tag | Valor | Por qué importa |
|---|---|---|
| `git_commit` | SHA completo del `HEAD` al lanzar el run | Une el modelo con una revisión inequívoca. |
| `git_dirty` | `true` / `false` (`git diff --quiet HEAD`) | Si `true`, el run **no es promovible**: hay código sin commit. |
| `dataset_sha256` | SHA-256 completo (64 hex) de `data/training/DB-HISTORICA.xlsx` | Distingue dos runs con mismo `git_commit` pero data distinta. |
| `dataset_n_rows` | filas totales agregadas sobre todas las hojas | Detecta truncamientos accidentales del Excel. |
| `dataset_s3_version_id` | VersionId de S3 o `local` | Permite recuperar exactamente el objeto usado aunque se sobrescriba la key. |
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

> **Verificación**: el #4.10 check #7 confirma que estos siete tags están
> presentes después de `task train`. Si alguno falta, el run no
> califica para `task ops:promote` (Parte 7) — el gate lo
> rechazará.

### 1.6 Costo objetivo

| Configuración | Costo mensual aproximado |
|---|---|
| **Tramo Local (laptop + 2 buckets sandbox)** | **~$0.05** (sólo storage S3) |
| Tramo II — Ciclo Mi+Ju 08-16 PET + teardown semanal (default, incluye API + UI) | ~$29 |
| Tramo II — Mismo scheduler pero SIN teardown (ALB+NAT 24/7) | ~$71 |
| Tramo II — Sin scheduler (24/7) | ~$195 |
| Tramo II — Hibernado (tear-down: NAT liberado en teardown vía `enable_nat=false`, solo storage) | ~$1 |

Detalle en Parte 9 (costos por servicio y por modo de lifecycle).
Son estimaciones de referencia: recalcular región, fecha, horas reales, IPv4,
NAT/GB, ALB LCU, snapshots, logs y métricas antes de aprobar presupuesto.

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
| RDS | Postgres 15, `db.t4g.small`, single-AZ | ~$23/mes 24/7, ~$2.21 con el ciclo Mi+Ju (69h encendido); hostea MLflow + la base `forecasts` de la API. | Multi-AZ (hardening, futuro). |
| Auto on/off de **servicios** | Scheduler EventBridge **Mi+Ju 08-16 PET** wake/sleep de RDS + Fargate (MLflow + Reports + API + UI); chequeo de Batch RUNNING antes de apagar. | UI 8 h/día, 2 días/semana. **Esto es scheduling de servicios, no de jobs de training** (ver fila "Trigger training"). El scheduler NO toca ALB ni NAT: esos los libera el `teardown` semanal, que es de donde sale el grueso del ahorro. | 24/7 si hay equipo distribuido. |
| Ciclo de vida semanal | `task rebuild` el miércoles, `task teardown` el jueves. **Nunca `task destroy`** para esto: vacía los buckets S3 donde viven modelos y reports. | El stack solo existe ~36h/semana; ALB+NAT dejan de facturar el resto. Es el 90% del ahorro vs el modelo anterior. | Dejarlo levantado si el uso pasa a ser diario. |
| TLS / acceso público | El perfil económico original usa ALB HTTP; se clasifica como **laboratorio controlado**, no producción expuesta. | Un SG abierto a Internet no autentica ni cifra. La implementación mostrada tampoco activa basic auth. | HTTPS + autenticación son obligatorios antes de exponer MLflow, API, UI, reports o artifacts a Internet. |
| Egress privado | NAT GW single-AZ ($32/mes 24/7, ~$6.25 con el ciclo semanal) | Setup simple si tráfico <10 GB/mes. | VPC endpoints **solo por hardening, ya no por costo**: con el NAT a ~139h/mes los endpoints interface salen más caros (`02-produccion-aws.md` #9.4.1). |
| Trigger training (de los **jobs**, no de los servicios) | (a) GHA `training.yml` workflow_dispatch (wake-train-sleep); (b) `aws lambda invoke ml-training-dispatcher`. **Sin cron de jobs**, sin S3 PutObject trigger — el scheduler de la fila anterior solo wake/sleep-ea servicios, nunca dispara entrenamientos. | Click desde GitHub UI eligiendo variedad. Training off-window wake-ea servicios y los apaga al terminar. | EventBridge cron diario de training / S3 trigger (Parte 7.5). |
| Modelos entrenables | **XGBoost + LightGBM** sobre `KG/JR_H`, con `TransformedTargetRegressor` (`log1p` + cap-p99). Champion automático. | Lo que vive en `src/step_04_train/registry.py`. | Stacking (eliminado, no existe). |
| Variedades válidas | **Dinámicas**: hojas del Excel `BD_HISTORICO_ACUMULADO.xlsx`. | Source of truth = el Excel. `list_varieties()` enumera `pd.ExcelFile(path).sheet_names`. La variable Terraform `varieties_allowed` es un allow-list defensivo del Lambda dispatcher, no la definición. | Agregar variedad = agregar hoja + `aws s3 cp` + opcional ampliar `varieties_allowed`. |
| Auth CI/CD | OIDC (sin access keys de larga duración) | Auditable en CloudTrail, sin rotación manual, blast-radius limitado al repo. | Keys sólo en CI legacy. |
| Promotion | Quality gate + comparación contra `@champion` + approval en GitHub Environments. El alias se reasigna solo después del approval. | Un MAPE menor no garantiza una mejora estable ni suficiente para negocio. | Auto-promote únicamente con política, observabilidad y rollback ensayado. |
| Reproducibilidad | `SEED=42` propagado a `np.random`, `random.seed`, `xgb.random_state`, `lgb.seed` y `TPESampler(seed=...)`. `TimeSeriesSplit` no acepta `random_state`: sus cortes son deterministas por orden temporal. El seed se registra como tag. | Reduce variación, pero no promete igualdad bit a bit entre CPU, librerías o paralelismo distintos. | Digest de imagen fijado + tolerancias numéricas documentadas. |
| Lineage del dataset | Tags `dataset_sha256` completo, `dataset_n_rows`, `dataset_s3_key` y `dataset_s3_version_id`. | El hash detecta cambios; key + VersionId permiten recuperar el objeto exacto. | DVC / lakeFS cuando haya varios productores o datasets grandes. |
| Contrato del run MLflow | Cada run loggea **tags obligatorios** (`git_commit`, `git_dirty`, `dataset_sha256`, `dataset_n_rows`, `tuning`, `variety`) **+ signature** (`infer_signature(X, y_pred)`) **+ input_example** (`X.head(5)`) **+ requirements snapshot** (`pip freeze`). Verificado en #4.10 check #7. | Sin signature, el Registry no valida payloads en serving; sin tags el quality gate de Parte 7 audita una caja negra. | Custom evaluators de MLflow (`mlflow.evaluate`) si pasás a regresión multi-target. |
| Code-quality gates | `ruff`, validación de Terraform, tests unitarios de contratos ML, integración Docker y smoke Batch. Mientras `tests/` no exista, CI debe bloquear releases productivos o declarar el despliegue experimental. | Un smoke confirma el camino feliz; no cubre leakage, selector, schemas, serialización ni errores de infraestructura. | `mypy`, SAST y cobertura guiada por riesgo. |
| CVE policy | `trivy image ml-training:local --severity HIGH,CRITICAL`. **Tramo Local**: warn-only. **Tramo II** (push a ECR): bloqueante. SBOM generado con `docker sbom` en cada `task ecr:build`. | Imagen base `python:3.13.1-slim-bookworm` tiene CVEs conocidos; para clientes regulados (HIPAA, SOC2, banca) el SBOM es requisito formal. | `cosign` para signing + admission controller en ECR. |
| Drift gate (PSI) | `task eda` calcula `psi_train_test` por feature numérica y escribe `artifacts/eda_<variety>.json`. `task train` lee ese JSON: `psi > 0.25` en cualquier feature **warn-loud** en stdout y se tagea el run con `psi_warn=true`. Bloqueante a futuro si pasa a ser política. | Drift severo entre train/test indica que el split de validación no representa training — un campeón sobre data drifteada es modelo sobre ruido. | `psi > 0.10` como warn temprano + `psi > 0.25` como bloqueante. |

---

### 2.A Baseline de calidad MLOps

Esta guía conserva AWS Batch, ECS Fargate, MLflow, Postgres, S3, Terraform,
Task y GitHub Actions. Subir el nivel MLOps no requiere cambiar esas piezas;
requiere definir contratos verificables y bloquear una promoción cuando no se
cumplen.

**Gate local mínimo antes del Tramo II**

1. `ruff` termina sin errores.
2. La suite de contratos existe y pasa; no se permite “0 tests collected”.
3. El split temporal demuestra que ninguna fila futura entra al fit de un fold.
4. El selector de campeón se prueba con empate, NaN, métricas faltantes y
   candidato que viola el máximo de gap.
5. El `.joblib` se recarga dentro de la misma imagen y predice contra un fixture
   con el schema esperado.
6. `docker compose config` y los health checks quedan verdes.
7. El run registra commit completo, estado dirty, digest de imagen, seed, hash
   completo del dataset, número de filas, key/VersionId de S3 y versiones de
   XGBoost, LightGBM, scikit-learn y MLflow.
8. El reporte estadístico corresponde al mismo `dataset_sha256`; un JSON EDA de
   otro dataset se ignora y obliga a regenerarlo.

**Métricas mínimas**

MAPE puede mantenerse como métrica de negocio, pero no debe ser la única:

- MAE o WAPE para evitar la inestabilidad de MAPE cuando el real se acerca a
  cero;
- `gap_oof_test` para generalización;
- cobertura y ancho medio si se publican intervalos conformes;
- latencia y tamaño del modelo para serving;
- error por ventana temporal y por variedad, no solo agregado.

Los umbrales se versionan como configuración y se registran como tags del run.
Cambiar un umbral requiere PR y deja evidencia; no se modifica desde la UI
durante una promoción.

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
wsl -d Ubuntu

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

Preferí credenciales temporales mediante IAM Identity Center (AWS SSO):

```bash
aws configure sso --profile ml-training-dev
aws sso login --profile ml-training-dev
export AWS_PROFILE=ml-training-dev
aws sts get-caller-identity
```

Si la cuenta todavía no usa SSO, un profile de desarrollo con claves es un
fallback transitorio, no el default recomendado:

```bash
aws configure --profile ml-training-dev
export AWS_PROFILE=ml-training-dev
aws sts get-caller-identity
```

> **Warning** — Nunca pongas `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` ni un
> token SSO en `.env`, el `Dockerfile` o el repositorio. Compose monta
> `~/.aws:/aws:ro`; ese montaje solo se entrega a los contenedores que realmente
> llaman AWS (`mlflow` y `trainer` en este diseño).

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
export AWS_DEFAULT_REGION="us-east-1"
export AWS_REGION="$AWS_DEFAULT_REGION"   # alias requerido por el backend S3 de Terraform (#4.2)
export AWS_PROFILE="default"              # o el profile que uses (debe quedar antes del STS de abajo)

export PROJECT="ml-training"
export ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export ACCOUNT_SUFFIX="${ACCOUNT_ID: -7}"

source "$(dirname "${BASH_SOURCE[0]}")/ensure-env.sh"
```

> 📂 **Pegar este bloque en**: `scripts/ensure-env.sh`

```bash
#!/usr/bin/env bash

: "${PROJECT:?ERROR: \$PROJECT vacia. Correr 'source scripts/prod.env' primero.}"
: "${ACCOUNT_SUFFIX:?ERROR: \$ACCOUNT_SUFFIX vacia. Correr 'source scripts/prod.env' primero.}"

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
docker build --progress=plain --no-cache -t ml-training:dryrun . 2>&1 | head -5
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
ARG PYTHON_VERSION=3.13.1-slim-bookworm

FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./

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
LABEL org.opencontainers.image.title="ml-training" \
      org.opencontainers.image.description="XGBoost and LightGBM training pipeline" \
      org.opencontainers.image.source="https://github.com/abantodca/ml_training" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates tini git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 mluser \
    && useradd  --system --uid 1001 --gid mluser --home ${APP_HOME} mluser

WORKDIR ${APP_HOME}

COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY --chown=mluser:mluser src/    ./src/
COPY --chown=mluser:mluser scripts/ ./scripts/
COPY --chown=mluser:mluser main.py  ./

RUN mkdir -p data/training logs artifacts reports \
    && chown -R mluser:mluser ${APP_HOME}

ENV USER=mluser
USER mluser
STOPSIGNAL SIGTERM

ENTRYPOINT ["/usr/bin/tini", "--", "python", "main.py"]
CMD ["--varieties", "POP", "--tuning", "smoke"]
```

> **Nota — por qué `git` viaja en el runtime.** `mlflow.utils.git_utils` y
> nuestro `collect_run_metadata` lo invocan en entrenamiento para taggear el
> run con el SHA del commit. Sin el binario todos los runs salen con
> `git_commit=unknown` y perdés la trazabilidad **modelo → commit**, la primera
> pregunta cuando un campeón regresione.

> **Reproducibilidad del sistema operativo.** No se ejecuta `apt-get upgrade`
> durante el build: haría que la misma revisión instalara paquetes distintos en
> fechas distintas. Los parches entran al actualizar en un PR controlado el
> digest de la imagen base y volver a ejecutar tests, SBOM y escaneo.

> **Nota — el uid 1001 no es casualidad.** `USER mluser` (uid 1001) está
> alineado con los bind-mount targets que crea `task _ensure_dirs` (#4.6.1) en
> el host. Si los dejás crear al vuelo, Docker los hace con uid `root` y el
> primer write del container falla con `Permission denied`.

**Verificación**

```bash
docker build -t ml-training:local .
docker images ml-training:local
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
      EXPERIMENT_PREFIX: ${MODEL_REGISTRY_PREFIX:-rnd-forest-}
      CORS_ORIGINS: ${CORS_ORIGINS:-http://localhost:8501}
      LOG_LEVEL: ${LOG_LEVEL:-info}
    ports:
      - "127.0.0.1:8000:8000"
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/api/health"]
      interval: 10s
      timeout: 5s
      retries: 6
      start_period: 40s
    logging: *default-logging

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

dotenv: [ ".env" ]

includes:
  local:
    taskfile: ./tasks/local.yml
    vars:
      PROJECT: '{{.PROJECT}}'
      REGION: '{{.REGION}}'

vars:
  VARIETIES: '{{.VARIETIES | default "POP"}}'
  PARALLEL:  '{{.PARALLEL  | default "1"}}'
  PROJECT:   '{{.PROJECT   | default "ml-training"}}'
  REGION:    '{{.AWS_DEFAULT_REGION | default "us-east-1"}}'
  HOST_UID:  { sh: id -u }
  HOST_GID:  { sh: id -g }
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

        Mas info:  task --list   docs/01-local.md
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
           MLflow Model Registry      http://localhost:5000/
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
> - **Los tests son un gate pendiente, no algo opcional**. Hasta crear
>   `tests/`, el flujo puede usarse para desarrollo y smoke, pero no debe
>   presentarse como release productivo certificado.

| Variable | Default | Override por CLI |
|---|---|---|
| `VARIETIES` | `POP` | `task train VARIETIES=POP,VENTURA` |
| `TUNING` | `prod_xl` | `task train TUNING=smoke` (`smoke` / `dev` / `prod` / `prod_xl`) |
| `PARALLEL` | `1` | `task train VARIETIES=all PARALLEL=3` |
| `SEED` | `42` (decisión Cap 2) | `task train SEED=1337` (reproducibilidad — el código del trainer la lee de `os.environ["SEED"]`) |

> **Vars auto-calculadas** (no se override, salen del entorno):
> `GIT_SHA` (`git rev-parse HEAD`), `GIT_DIRTY` (`git diff --quiet`),
> `DATA_SHA` (los 64 caracteres del `sha256sum` de
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
task local:ensure-buckets
task local:bucket-name KIND=data
task local:bucket-name KIND=artifacts
```

> **Dónde se invoca.** El bloque `includes: local:` ya está pegado en
> #4.6.1 (entre `dotenv:` y `vars:` del Taskfile raíz) — no hay que
> agregar nada más. Este archivo se define **una sola vez acá**; en
> Tramo II Parte 4 #4.1.8 solo se lo referencia para evitar drift.

```yaml
version: "3"

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

case "${1:-}" in
  git-sha)
    git rev-parse HEAD 2>/dev/null || echo unknown
    ;;
  git-dirty)
    git diff --quiet HEAD 2>/dev/null && echo false || echo true
    ;;
  data-sha)
    f=data/training/DB-HISTORICA.xlsx
    [ -f "$f" ] && sha256sum "$f" | cut -d' ' -f1 || echo missing
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
set -euo pipefail

acct=$(aws sts get-caller-identity --query Account --output text)
echo "${acct#?????}"
```

> 📂 **Pegar este bloque en**: `scripts/ensure-s3-bucket.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

name="${1:?falta <name>}"
region="${2:?falta <region>}"

if aws s3api head-bucket --bucket "$name" 2>/dev/null; then
  echo "  $name  EXISTE (reaplicando hardening)"
else
  echo "  $name  no existe -> creando..."
  if [ "$region" = "us-east-1" ]; then
    aws s3api create-bucket --bucket "$name" --region "$region"
  else
    aws s3api create-bucket --bucket "$name" --region "$region" \
      --create-bucket-configuration "LocationConstraint=$region"
  fi
fi

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
AWS_PROFILE=default
AWS_DEFAULT_REGION=us-east-1

S3_MLFLOW_BUCKET=ml-training-artifacts-XXXXXX
S3_ARTIFACTS_BUCKET=ml-training-artifacts-XXXXXX
```

#### 4.7.2 Copia para uso real

```bash
cp .env.example .env
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
```

> **Los dos buckets quedan vacíos — es lo esperado.** `ensure-buckets` solo los
> **crea**, no sube datos:
>
> - `{project}-artifacts-<suffix>` lo puebla el trainer al final de `task train`
>   ([scripts/s3_sync.py](../scripts/s3_sync.py)) y el server MLflow durante el run.
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
task build

task data:split

task eda VARIETIES=POP

task train VARIETIES=POP TUNING=smoke
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
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5000/health

docker compose exec postgres psql -U mlflow -d mlflow -c \
  "SELECT name, (SELECT COUNT(*) FROM runs WHERE experiment_id = e.experiment_id) AS n_runs
   FROM experiments e WHERE name = 'POP';"

SUFFIX=$(bash scripts/aws-suffix.sh)
aws s3 ls "s3://ml-training-artifacts-${SUFFIX}/artifacts/" --recursive \
  | grep -E "final_pipeline_POP_.*\.joblib$"

curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/reports/

cat artifacts/run_summary_AGGREGATE.json | jq '.champions'

docker compose run --rm --no-deps --entrypoint python trainer -c '
import glob, joblib, pandas as pd
path = sorted(glob.glob("artifacts/final_pipeline_POP_*.joblib"))[-1]
m = joblib.load(path)
sample = pd.read_excel("data/training/DB-HISTORICA.xlsx", sheet_name="POP").head(1)
y = m.predict(sample.drop(columns=["KG/JR_H"], errors="ignore"))
assert y.shape == (1,), f"prediccion mal: {y.shape}"
print(f"OK  joblib={path}  y[0]={y[0]:.3f}")
'

docker compose exec postgres psql -U mlflow -d mlflow -tA -c "
WITH last_run AS (
  SELECT r.run_uuid
  FROM runs r JOIN experiments e ON r.experiment_id = e.experiment_id
  WHERE e.name = 'POP' ORDER BY r.start_time DESC LIMIT 1
),
need AS (
  SELECT unnest(ARRAY['git_commit','git_dirty','dataset_sha256','dataset_n_rows',
                      'dataset_s3_version_id','seed','tuning','variety']) AS k
),
have AS (SELECT key AS k FROM tags WHERE run_uuid IN (SELECT run_uuid FROM last_run))
SELECT n.k AS missing FROM need n LEFT JOIN have h USING(k) WHERE h.k IS NULL;
"
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
task train VARIETIES=POP TUNING=dev
task train VARIETIES=POP TUNING=prod
task train VARIETIES=all PARALLEL=3

docker compose logs -f --tail=200 trainer mlflow

task build

trivy image ml-training:local --severity HIGH,CRITICAL --ignore-unfixed

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
> superficie). CI debe construir una sola imagen candidata, escanearla y
> desplegar exactamente su tag SHA o digest; no debe reconstruirla después del
> gate. Conocer las vulnerabilidades acá te evita el push fallido en Tramo II Parte
> 4, donde el scan-on-push del ECR puede bloquear el deploy.

Lo que **no cambia** entre tu laptop y AWS — la garantía central del
diseño "una sola imagen":

- El `Dockerfile` y el contexto son los mismos. La igualdad del artefacto se
  garantiza únicamente si se promueve el mismo digest; dos builds separados no
  son necesariamente bit a bit iguales.
- El código (`main.py`, `src/`) se configura enteramente por variables
  de entorno; sólo cambia el origen de esas variables (en local salen
  del `.env`, en Batch de la job definition).
- El resultado debe ser reproducible dentro de tolerancias documentadas: misma
  data versionada, digest de imagen, seed y recursos comparables. No se promete
  un `.joblib` byte a byte idéntico entre hosts.

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
> origen del dataset es una ruta donde la paridad de configuración se
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
