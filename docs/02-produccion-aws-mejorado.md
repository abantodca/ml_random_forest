# Guía experta — Producción en AWS (Terraform, AWS Batch + ECS Fargate + MLflow)

> [!WARNING]
> El camino base conserva el perfil económico del proyecto. Con ALB público por
> HTTP y sin autenticación aplicada a todos los paths, es un **entorno de
> laboratorio controlado**, no una exposición productiva segura. Antes de
> permitir acceso desde Internet se debe completar el gate de la Parte 10:
> HTTPS, autenticación, secretos separados, pruebas, imágenes inmutables y
> observabilidad de inferencia.

> Tramo II del stack: promover a AWS el **mismo contrato de imagen** que validaste en local
> ([`docs/01-local.md`](01-local.md)). Terraform copy-paste por módulos, estado remoto en S3, apply
> incremental en olas con checkpoint, y un lifecycle de cuatro modos —STAND-UP, TEAR-DOWN, REBUILD,
> DESTROY— para que la infra se apague cuando no la usás.
>
> La arquitectura, en un solo camino: el **entrenamiento** corre en **AWS Batch** (colas Spot +
> On-Demand, `c6i.2xlarge`, escala a cero entre corridas); **MLflow** vive en **ECS Fargate** detrás
> de un ALB con **RDS Postgres** + S3 como backend
> ([ADR-001](adr/ADR-001-mlflow-backend-postgres-s3.md)); la **API FastAPI** y la **UI Streamlit**
> corren como servicios ECS en ese mismo cluster y ALB; los **reports** los sirve un nginx Fargate
> Spot. Alrededor: **Lambdas** para disparar jobs y notificar, un **scheduler** que prende y apaga
> RDS + Fargate por cron, **CloudWatch + SNS** para alarmas, y **GitHub Actions con OIDC** (sin
> access keys) para deploy, training y destroy.
>
> Regla mental: **almacenar es barato y constante; computar es lo que cuesta, y solo cuando corrés.**
> Por eso Batch escala a cero, el scheduler apaga RDS y Fargate fuera de horario, y los artefactos
> viven en S3. **~$29/mes** con el ciclo miercoles+jueves 08-16 PET y teardown
> semanal (#9).

Índice:

- **Parte 1** — [Overview del lifecycle y stand-up](#parte-1--overview-del-lifecycle-y-stand-up)
- **Parte 2** — [Bootstrap irreversible](#parte-2--bootstrap-irreversible)
- **Parte 3** — [Módulos Terraform](#parte-3--modulos-terraform)
- **Parte 4** — [Apply incremental + smoke test](#parte-4--apply-incremental--smoke-test)
- **Parte 5** — [Patch del trainer (emitir MAPE a CloudWatch)](#parte-5--patch-del-trainer-emitir-mape-a-cloudwatch)
- **Parte 6** — [CI/CD con GitHub Actions](#parte-6--cicd-con-github-actions)
- **Parte 7** — [Promotion gate](#parte-7--promotion-gate-aliases-de-mlflow)
- **Parte 8** — [Runbook operativo](#parte-8--runbook-operativo-extendido)
- **Parte 9** — [Costos detallados](#parte-9--costos-detallados)
- **Parte 10** — [Aseguramiento MLOps y hardening mínimo](#parte-10--aseguramiento-mlops-y-hardening-mínimo)
- **Parte 11** — [Troubleshooting](#parte-11--troubleshooting-catalogo)
- **Parte 12** — [Apéndices](#parte-12--apendices)

> La Parte 10 ocupa el hueco histórico sin renumerar ninguna referencia
> existente. Reúne los controles que separan un laboratorio económico de una
> plataforma MLOps realmente publicable.

---

## Cómo leer esta guía

**Cada bloque dice dónde corre.** `[LOCAL]` es tu máquina, parada en la raíz del repo, con tus
credenciales AWS. `[AWS]` es un recurso gestionado (una task de ECS, un job de Batch), que se toca
por CLI o consola. `[CI]` corre dentro de GitHub Actions con el rol asumido por OIDC. La distinción
no es cosmética: `[LOCAL]` prueba que el recurso **existe**; `[CI]` prueba que el **rol de OIDC**
tiene los permisos, que es un fallo completamente distinto y se diagnostica distinto.

**Terraform es la fuente de verdad.** Todo lo que esta guía crea está en `infra/` y se aplica con
`task`. No toques recursos a mano en la consola: mientras Terraform los gestione, el próximo `apply`
revierte el cambio en silencio, y a los tres meses nadie sabe por qué el security group tiene esa
regla. Si algo ya lo creaste a mano y querés pasarlo a Terraform, primero `terraform import`.

**El orden de las Partes son las dependencias reales.** El bootstrap (Parte 2) es irreversible y va
primero porque crea el bucket del state; los módulos (Parte 3) se aplican en olas (Parte 4) porque
el grafo tiene puntos donde conviene frenar y verificar antes de seguir. Saltarse el orden funciona
hasta que falla, y cuando falla el error no señala la causa.

**Las decisiones ya tomadas no se discuten acá.** Están ratificadas en
[`docs/adr/`](adr/) y en la tabla de *Decisiones fijas* de
[`docs/01-local.md` Capítulo 2](01-local.md#capítulo-2--decisiones-fijas). Cambiar alguna implica un
ADR previo y reescribir las secciones afectadas — no un parche local.

> **Estado real del repo, para que no busques archivos que no existen.** `infra/` **sí** existe (45
> archivos `.tf`, 11 módulos + `envs/prod`). Lo que **no** existe todavía es
> `.github/workflows/{deploy,training,destroy}.yml`: los YAML de la Parte 6 son material **por
> crear**, y el módulo `infra/modules/cicd/` viene con `enable_cicd = false` por defecto. Tampoco hay
> una suite suficiente para liberar a producción; la Parte 10 define el mínimo
> y el workflow de esta versión ya ejecuta `pytest`. Los ADR sí existen en
> [`docs/adr/`](adr/).

---


> El mismo contrato de imagen que validaste en Tramo I se publica en AWS Batch +
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
6. **Partes 8-12** — runbook, costos, aseguramiento MLOps, troubleshooting y apéndices.

---

## Parte 1 — Overview del lifecycle y stand-up

La V1 mezcla los modos a lo largo del runbook. En la V2 son explicitos.
Cada uno responde a una pregunta concreta:

| Modo | Pregunta que responde | Tiempo | Costo despues |
|---|---|---|---|
| **STAND-UP** | "Es la primera vez, parto de cero" | 2-3 horas | ~$29/mes (operando) |
| **TEAR-DOWN** | "Termine el jueves / no uso la infra por unos dias" | 15-25 min | ~$1/mes (storage + backups; RDS y NAT liberados) |
| **REBUILD** | "Volvi y quiero levantar otra vez sin perder modelos/data" | 25-40 min | ~$29/mes |
| **DESTROY** | "Termine el proyecto / migro a otra cuenta, borra TODO" | 30-45 min | $0/mes |

TEAR-DOWN ↔ REBUILD es el par del **ciclo recurrente** (prender unos días,
trabajar, apagar): el estado viaja solo vía backup del RDS + S3, sin pasos
manuales. Ver #8.2.0 para el runbook y #8.5 para el mecanismo.

Diagrama de transiciones:

```
                          stand-up
            (vacio) ─────────────────► OPERATING (~$29/mes)
                                          │  ▲
                                          │  │
                                  tear-down  rebuild
                                          │  │
                                          ▼  │
                                       HIBERNATED (~$1/mes)
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

### 1.1 STAND-UP — primera vez, de cero a produccion

Cuando lo uso: la primera vez que despliego, o tras un **DESTROY**.

#### Camino completo

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
OPERATING (~$29/mes según el escenario de referencia de la Parte 9)
```

**Tiempo total realista**: 2-3 horas la primera vez, asumiendo que los
prereqs (0.3) estan OK y la imagen Docker del trainer ya esta probada
local (0.3.5).

#### Lo que NO se hace en stand-up

- TLS y autenticación: obligatorios antes de exponer el ALB a Internet; Parte
  10. Multi-AZ, WAF, KMS-CMK y DR cross-region siguen siendo decisiones según
  criticidad y cumplimiento.
- Workflows extras (cleanup, drift detection): hardening, futuro.
- Promotion gate: no se salta en un release productivo. El primer modelo también
  debe pasar el gate y recibir el alias `@champion`; el approval puede hacerlo
  una sola persona en un entorno personal.

### 1.2 Otros modos (TEAR-DOWN / REBUILD / DESTROY)

Estos modos son operaciones del runbook (ya tenes el sistema construido),
no del stand-up inicial. En tu primera lectura no los necesitas — saltalos
y volve cuando ya estes operando. Estan documentados en Parte 8:

- **#8.5 — TEAR-DOWN**: apagar todo preservando state + datos (~$1/mes
  hibernado — NAT liberado vía `enable_nat=false` —, reversible con rebuild).
  El RDS se **destruye** tomando un backup verificado antes; S3 (artifacts) intacto.
  Es el modo del ciclo recurrente de prender/apagar (#8.2.0).
- **#8.6 — REBUILD**: volver despues de un tear-down. **Restaura el RDS desde
  el ultimo backup** (Model Registry + tabla `forecasts`); cambia el ALB DNS.
  `task deploy` restaura igual: comparten el mismo resolver (#8.5).
- **#8.8 — VERIFICAR LIMPIO**: `task infra:verify-clean` tras cualquier
  destroy/nuke — `terraform destroy` deja residuos fuera del state.
- **#8.7 — DESTROY**: eliminar TODO de la cuenta AWS. Respalda el RDS solo,
  pero **vacía los buckets de S3**: los artifacts no vuelven. Requiere archivar
  a mano lo que quieras conservar. No es el modo para ahorrar — para eso,
  tear-down.

La matriz cruzada de costos entre modos (stand-up vs tear-down vs destroy)
esta en #9.3.

---

## Parte 2 — Bootstrap irreversible

### 2.1 Por que el bootstrap es a mano

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

### 2.2 Script de bootstrap (bash)

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

### 2.3 Ejecutar UNA vez

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

### 2.4 Verificacion post-bootstrap (3 checks)

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

### 2.5 OIDC provider para GitHub Actions (pre-Terraform)

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

#### Script `infra/bootstrap-oidc.sh`

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

#### Ejecutar UNA vez

```bash
chmod +x infra/bootstrap-oidc.sh
bash infra/bootstrap-oidc.sh
```

#### Verificacion

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

### 2.6 Snapshot del estado bootstrapped (commit + tag)

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

## Parte 3 — Modulos Terraform

> **Filosofia de la Parte 3**: cada modulo es una caja con interface
> publica (variables.tf + outputs.tf). El `envs/prod/main.tf` solo
> compone — no contiene `resource "aws_..."` directos. Esto te deja:
>
> - Tocar `modules/batch/` sin re-aplicar el resto.
> - Crear `envs/dev/` o `envs/staging/` copiando `envs/prod/` y
>   cambiando solo `terraform.tfvars`.
> - Hacer reviews de PR donde el diff de un cambio chico es chico
>   (no 200 lineas mezcladas).

### 3.1 Layout — el arbol de archivos

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

### 3.2 `envs/prod/` — la composicion

#### 3.2.1 `infra/envs/prod/versions.tf`

Locks de versiones — toda la guia esta probada con estas versiones. Si
las cambias, vas a tener que ajustar sintaxis (p.ej. `for_each` map en
v5 vs v4 del provider AWS).

```hcl
terraform {
  required_version = ">= 1.10.0, < 2.0.0" # ≥1.10 obligatorio: el backend usa use_lockfile=true (ignorado en silencio en 1.6–1.9)

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

#### 3.2.2 `infra/envs/prod/backend.tf`

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

#### 3.2.3 `infra/envs/prod/variables.tf`

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

variable "enable_cicd" {
  description = "Activa CI/CD (OIDC + module.cicd). Default false: el stand-up completo corre sin bootstrap-oidc.sh ni github_org/repo. Poner true tras #2.5 + re-apply."
  type        = bool
  default     = false
}

variable "github_org" {
  description = "Organizacion / usuario GitHub que aloja el repo (para OIDC trust). Solo requerido con enable_cicd=true."
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "Nombre del repo (sin la org). Para trust policy OIDC. Solo requerido con enable_cicd=true."
  type        = string
  default     = ""
}

variable "enable_nat" {
  description = "Gate del NAT gateway + EIP + ruta privada default (modulo network). Default true. `task teardown` lo pone en false para LIBERAR el NAT (~$33/mes idle) preservando VPC/subnets/SGs; rebuild/deploy lo recrean."
  type        = bool
  default     = true
}

variable "rds_deletion_protection" {
  description = "deletion_protection del RDS MLflow. Default true (protectivo). teardown/destroy lo levantan via AWS CLI antes del destroy."
  type        = bool
  default     = true
}

variable "rds_skip_final_snapshot" {
  description = "skip_final_snapshot del RDS MLflow. Default false (protectivo: un destroy a mano igual deja copia). teardown/destroy lo pasan en TRUE porque ya tomaron un backup verificado ANTES del destroy (ensure_backup, #8.5) y el final_snapshot seria un duplicado de ~8 min."
  type        = bool
  default     = false
}

variable "rds_final_snapshot_identifier" {
  description = "Identificador del snapshot final del RDS (solo si rds_skip_final_snapshot=false). Red de seguridad para destroys manuales; las tareas del repo no lo usan (respaldan antes). Vacio = <project>-mlflow-final."
  type        = string
  default     = ""
}

variable "rds_snapshot_identifier" {
  description = "Backup desde el que RESTAURAR el RDS al crearlo. Vacio (default) = instancia nueva y vacia. Lo inyectan `task deploy` y `task ops:rebuild` via resolve_restore_snapshot(); ver tasks/lib/snapshot.sh y #8.5."
  type        = string
  default     = ""
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
  default     = 16
}

# Dias que el scheduler considera laborables. DEBE wirearse a module.scheduler
# (#3.10.5): el modulo tiene su propio default y si no se pasa, el valor de aqui
# se ignora en silencio. Tokens de EventBridge cron ("WED,THU" o "MON-FRI").
variable "workdays_cron" {
  description = "Dias con ventana encendida (ciclo miercoles+jueves)."
  type        = string
  default     = "WED,THU"
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

#### 3.2.4 `infra/envs/prod/terraform.tfvars` (NO COMMITEAR)

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

#### 3.2.5 `infra/envs/prod/main.tf` (esqueleto — crece incrementalmente)

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
# Gateado por `var.enable_cicd` (default false): con CI/CD apagado el `data`
# tiene count=0 y NO se evalua → el stand-up completo (storage→network→mlflow
# →batch→api/ui) corre SIN `bash infra/bootstrap-oidc.sh` y sin github_org/repo.
# Solo si activas CI/CD (enable_cicd=true, tras #2.5) este `data` debe resolver.
# Pre-check antes de `terraform plan` (solo con enable_cicd=true):
#   aws iam list-open-id-connect-providers --query 'OpenIDConnectProviderList[?contains(Arn,`token.actions.githubusercontent.com`)]'
# Si devuelve [], correr `bash infra/bootstrap-oidc.sh` (#2.5).
data "aws_iam_openid_connect_provider" "github" {
  count = var.enable_cicd ? 1 : 0
  url   = "https://token.actions.githubusercontent.com"
}
```

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `data "aws_iam_openid_connect_provider"` | **IAM** | `IAM > Identity providers` → veras `token.actions.githubusercontent.com` (creado por `bootstrap-oidc.sh` en #2.5). El `data` lo "lee" para que `module.cicd` (#3.11) pueda asignar trust policies a los roles GHA sin hardcodear el ARN. Con `enable_cicd=false` (default) este `data` no se evalua y no necesitas haber corrido `bootstrap-oidc.sh`. |
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

#### 3.2.6 `infra/envs/prod/outputs.tf`

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
  description = "Role que asume GitHub Actions para `terraform apply`. null si enable_cicd=false."
  value       = var.enable_cicd ? module.cicd[0].gha_deploy_role_arn : null
}

output "gha_train_role_arn" {
  description = "Role que asume GitHub Actions para invocar Lambda dispatcher. null si enable_cicd=false."
  value       = var.enable_cicd ? module.cicd[0].gha_train_role_arn : null
}

# Patch 13.5
output "consumer_role_arn" {
  description = "Role que asume el repo consumer (ml_serving) via OIDC para descargar artifacts."
  value       = module.consumer_iam.consumer_role_arn
}
```

> **Gotcha #3.2**: `terraform.tfvars` debe existir con valores reales y NO commitearse (`.gitignore` debe contener `**/terraform.tfvars`). Sin el archivo `validate` aún pasa, pero `plan` falla con "No value for required variable".

### 3.3 `modules/network/` — VPC + subnets + NAT + SGs

Single-AZ a proposito (Sec 0.2 lockeada). El SG matrix es:

- `sg_alb`: ingress :80 from 0.0.0.0/0 (futuro: WAF + TLS, hardening)
- `sg_mlflow`: **dos reglas de ingress desde `sg_alb`**: :5000 (MLflow
  task escucha ahi) y :80 (modulo reports reusa este SG porque comparte
  ECS cluster; el container nginx escucha en :80). Nada desde 0.0.0.0/0.
- `sg_rds`: ingress :5432 from `sg_mlflow` + `sg_batch` (Batch necesita
  conectar a RDS para registrar runs via MLflow Python client)
- `sg_batch`: egress 443 a internet (S3, ECR, MLflow ALB)

#### 3.3.1 `modules/network/variables.tf`

```hcl
variable "project" { type = string }
variable "vpc_cidr" { type = string }
```

#### 3.3.2 `modules/network/main.tf`

Pegar los bloques siguientes **uno a continuacion del otro** en el
mismo archivo `modules/network/main.tf`. La separacion en sub-bloques
con `### 3.3.2.X` es solo para que puedas leer el "por que" de cada
pieza sin perderte; el archivo final es la concatenacion de los 5
bloques.

##### 3.3.2.a — Discovery de AZs + VPC

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

##### 3.3.2.b — Subnets (public + private, x2 AZs)

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

##### 3.3.2.c — Internet Gateway + NAT (single, en public[0])

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
>
> **Toggle `enable_nat`** (default `true`): el NAT gateway + EIP + la ruta privada default van gateados por `var.enable_nat`. `task teardown` corre `terraform apply -target=module.network -var enable_nat=false` tras destruir los volátiles → **libera el NAT (~$33/mes idle)** preservando VPC/subnets/SGs. `task rebuild`/`deploy` lo recrean (default `true`).

```hcl
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project}-igw" }
}

resource "aws_eip" "nat" {
  count  = var.enable_nat ? 1 : 0
  domain = "vpc"
  tags   = { Name = "${var.project}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  count         = var.enable_nat ? 1 : 0
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id
  tags          = { Name = "${var.project}-nat" }
  depends_on    = [aws_internet_gateway.igw]
}
```

##### 3.3.2.d — Route tables

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
  # Ruta default solo si enable_nat=true; con NAT liberado la RT queda sin 0.0.0.0/0.
  dynamic "route" {
    for_each = var.enable_nat ? [1] : []
    content {
      cidr_block     = "0.0.0.0/0"
      nat_gateway_id = aws_nat_gateway.main[0].id
    }
  }
  tags = { Name = "${var.project}-rt-private" }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}
```

##### 3.3.2.e — Security Groups (6: alb, mlflow, batch, rds, api, ui)

6 SGs en cascada (alb es la unica que acepta 0.0.0.0/0; las otras
solo aceptan trafico desde la anterior, formando una cadena de
defense-in-depth):

- `sg-alb`: Internet → :80.
- `sg-mlflow`: sg-alb → :5000 (MLflow) y :80 (reports), **+ sg-api y
  sg-batch → :5000**. API y trainer consultan MLflow por service discovery,
  sin pasar por el ALB público. Compartido por MLflow y reports porque ambos
  viven detrás del mismo ALB.
- `sg-batch`: solo egress (el trainer descarga de S3/ECR, escribe logs y
  llama a MLflow interno). No acepta ingress.
- `sg-rds`: 5432 desde sg-mlflow **+ sg-api** (la API persiste
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
> | `aws_security_group.rds` | **VPC** | **Inbound rules**: PostgreSQL :5432 desde `sg-mlflow` y `sg-api`. Batch no entra a Postgres: registra por la API HTTP de MLflow. **Outbound**: vacío. |
>
> **Conceptualmente — SGs son firewalls "stateful" a nivel recurso**:
> - **Stateful**: si permitís el ingreso, la respuesta saliente se permite sola (a diferencia de los NACLs stateless).
> - En `Source` podés poner **otro SG** en vez de un CIDR: `sg-rds` acepta a
>   `sg-mlflow`/`sg-api` aunque sus IPs cambien.
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
  ingress {
    description     = "MLflow desde AWS Batch (service discovery interno)"
    from_port       = 5000
    to_port         = 5000
    protocol        = "tcp"
    security_groups = [aws_security_group.batch.id]
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

#### 3.3.3 `modules/network/outputs.tf`

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
>   desde sg-alb + 5000 desde sg-api y sg-batch), `sg-batch` (egress all),
>   `sg-rds` (ingress 5432 desde sg-mlflow + sg-api),
>   `sg-api` (ingress 8000 desde sg-alb + sg-ui), `sg-ui` (ingress
>   8501 desde sg-alb).

#### 3.3.4 Apendear `module "network"` en `infra/envs/prod/main.tf`

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

### 3.4 `modules/storage/` — S3 buckets + ECR repos

#### 3.4.1 `modules/storage/variables.tf`

```hcl
variable "project" { type = string }
```

#### 3.4.2 `modules/storage/main.tf`

Pegar los bloques siguientes **uno a continuacion del otro** en el
mismo archivo `modules/storage/main.tf`. La separacion en sub-bloques
con `### 3.4.2.X` es solo para que puedas leer el "por que" de cada
pieza sin perderte; el archivo final es la concatenacion de los 5
bloques.

##### 3.4.2.a — Header: account suffix discovery

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

##### 3.4.2.b — S3 bucket `data` (input Excels) + hardening

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

resource "aws_s3_bucket_policy" "data_tls_only" {
  bucket = aws_s3_bucket.data.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}
```

##### 3.4.2.c — S3 bucket `artifacts` (modelos + reportes + MLflow store) + lifecycle

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

resource "aws_s3_bucket_policy" "artifacts_tls_only" {
  bucket = aws_s3_bucket.artifacts.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [
        aws_s3_bucket.artifacts.arn,
        "${aws_s3_bucket.artifacts.arn}/*"
      ]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
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

##### 3.4.2.d — ECR `trainer` + lifecycle policy

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

##### 3.4.2.e — ECR `mlflow` (IMMUTABLE) + `reports` + `api` + `ui` (MUTABLE)

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

#### 3.4.3 `modules/storage/outputs.tf`

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

#### 3.4.4 Apendear `module "storage"` en `infra/envs/prod/main.tf`

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

### 3.4.5 `modules/_shared/` — Trust policies compartidos

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

#### 3.4.5.1 Contenido de los archivos

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
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": [
            "repo:${org}/${repo}:ref:refs/heads/main",
            "repo:${org}/${repo}:environment:production"
          ]
        }
      }
    }
  ]
}
```

> **Gotcha #3.4.5**: los JSON deben validar como JSON estricto (sin comentarios, sin trailing commas). Diagnostico: `jq . infra/modules/_shared/assume-ecs-tasks.json` no debe errorizar.

---

### 3.5 `modules/mlflow/` — RDS + ECS Fargate + ALB

Este modulo es el mas pesado: arma el backend de tracking (RDS Postgres),
el server MLflow en Fargate y el ALB que expone todo. Aca lockean dos
contratos criticos del codigo del trainer:

- `--allowed-hosts` enumera el DNS del ALB y el nombre interno de Cloud Map.
  El DNS del ALB es un atributo conocido durante `apply`, igual que el origen
  CORS; no hace falta abrir un wildcard.
- El usuario Postgres se llama `mlflow` y la DB se llama `mlflow`
  (igual que en docker-compose local, para que el trainer no tenga que
  cambiar la connection string entre local y prod).

#### 3.5.0 Decisiones de arquitectura (rationale para experto)

Tres elecciones marcan el caracter de este modulo y vale la pena
justificarlas antes de leer el HCL:

**Fargate vs EC2 para el server MLflow.** El server MLflow es una
aplicación Python con varios workers y throughput modesto (decenas de
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

#### 3.5.1 `modules/mlflow/variables.tf`

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

#### 3.5.2 `modules/mlflow/` (split en 5 archivos)

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

##### 3.5.2.a — `rds.tf` — RDS Postgres (subnet group + instance)

> ⚠️ **La credencial master NO va en este modulo.** Antes de pegar `rds.tf`,
> crear `infra/envs/prod/rds_secret.tf` (raiz, fuera de `modules/`):
>
> ```hcl
> # infra/envs/prod/rds_secret.tf
> # Vive en la RAIZ para SOBREVIVIR a `task ops:teardown`, que destruye
> # module.mlflow (el RDS incluido). Si la password se destruyera con el modulo,
> # el rebuild generaria una nueva y no coincidiria con la del snapshot
> # restaurado -> el rebuild fallaria hasta rotar la credencial. Ver #8.5.
> resource "random_password" "rds" {
>   length  = 32
>   special = false # algunos chars rompen connection strings -> evitar
> }
>
> resource "aws_secretsmanager_secret" "rds" {
>   name = "${var.project}-rds-password"
> }
>
> resource "aws_secretsmanager_secret_version" "rds" {
>   secret_id     = aws_secretsmanager_secret.rds.id
>   secret_string = random_password.rds.result
> }
> ```
>
> Y en el bloque `module "mlflow"` de `envs/prod/main.tf`, pasarlas:
>
> ```hcl
>   rds_snapshot_identifier = var.rds_snapshot_identifier
>   rds_password            = random_password.rds.result
>   rds_password_secret_arn = aws_secretsmanager_secret.rds.arn
> ```
>
> `module.api` tambien toma `rds_password_secret_arn = aws_secretsmanager_secret.rds.arn`
> directamente de la raiz (antes venia por un output de `module.mlflow`, ya removido).
>
> **Si venis de una version anterior** donde estos 3 recursos vivian dentro de
> `module.mlflow`, correr **una sola vez** antes del primer apply:
>
> ```bash
> task infra:migrate-rds-secret   # terraform state mv, NO recrea nada
> task infra:plan                 # debe salir sin cambios en el RDS ni en la password
> ```
>
> Es `state mv` a proposito: un apply directo veria los recursos viejos como "a
> destruir" y los nuevos como "a crear" → **password nueva y RDS vivo inaccesible**.

> 📂 **Pegar este bloque en**: `infra/modules/mlflow/rds.tf`
> (incluye el `data "aws_region" "current" {}` — en el repo real ese
> data source vive aca, no en `main.tf`).

`random_password` + Secrets Manager evita hardcodear el password en HCL y evita
mostrarlo en salidas normales, pero **no evita que exista en el state**:
`random_password.result` y `secret_string` son atributos del estado. Por eso el
bucket de tfstate es un activo sensible, cifrado, versionado y con acceso
mínimo. **Ambos recursos viven en la raiz**
(`infra/envs/prod/rds_secret.tf`), **no en este modulo**: `task teardown`
destruye `module.mlflow` con el RDS dentro, y si la credencial se fuera con el,
el `rebuild` generaria una password nueva incompatible con la del backup
restaurado. El modulo las recibe como `var.rds_password` y
`var.rds_password_secret_arn`. Ver #8.5 "Ciclo backup → restauración".
`subnet_group` en private subnets x2
porque RDS exige 2 AZs aunque sea single-AZ. `skip_final_snapshot` y
`deletion_protection` son **variables con defaults protectivos**
(`deletion_protection=true`, `skip_final_snapshot=false`): el RDS arranca
protegido, de modo que un `terraform destroy` corrido a mano igual deja copia.
`task destroy`/`task teardown` levantan la protección vía AWS CLI
(`aws rds modify-db-instance --no-deletion-protection`, helper
`lift_rds_protection` en `tasks/lib/nuke.sh`) y pasan
`rds_skip_final_snapshot=true`, porque ellos ya tomaron un **backup verificado
antes** de empezar a destruir (#8.5).

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
> - **`skip_final_snapshot` + `deletion_protection`**: variables con defaults protectivos (`deletion_protection=true`, `skip_final_snapshot=false`) → el RDS arranca protegido y un destroy manual igual deja copia. No hay que tocarlos: `task teardown`/`task destroy` levantan la protección por CLI (`lift_rds_protection`) y pasan `rds_skip_final_snapshot=true` porque ya respaldaron antes de destruir.
> - **`snapshot_identifier`** (restauración): vacío = instancia nueva y vacía; con valor, el RDS **se crea a partir de ese backup**. Lo inyectan `task deploy` y `task ops:rebuild` vía `resolve_restore_snapshot()`. Lleva `ignore_changes` **obligatorio** porque es *ForceNew*: sin él, un apply posterior sin el mismo `-var` recrearía la instancia y perdería lo restaurado. Ver #8.5.

```hcl
# infra/modules/mlflow/rds.tf
data "aws_region" "current" {}

# La credencial master (random_password + secret) NO vive aca a proposito:
# teardown destruye este modulo y la password debe sobrevivir para poder
# restaurar los snapshots. Ver infra/envs/prod/rds_secret.tf.

resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.project}-rds-subnets"
  subnet_ids = var.private_subnet_ids
}

resource "aws_db_instance" "mlflow" {
  identifier             = "${var.project}-mlflow"
  engine                 = "postgres"
  engine_version         = "15"
  instance_class         = var.rds_instance_class
  allocated_storage      = var.rds_allocated_storage_gb
  storage_type           = "gp3"
  storage_encrypted      = true
  db_name                = "mlflow"
  username               = "mlflow"
  password               = var.rds_password
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids = [var.sg_rds_id]
  publicly_accessible    = false
  apply_immediately      = true

  # Proteccion de datos. Defaults protectivos para que un `terraform destroy`
  # corrido a mano igual deje copia. Las tareas de destroy/teardown levantan
  # deletion_protection via AWS CLI y pasan skip_final_snapshot=true: ya tomaron
  # un backup verificado ANTES de destruir (ensure_backup, #8.5), y duplicarlo
  # costaria ~8 min mas de espera.
  deletion_protection       = var.rds_deletion_protection
  skip_final_snapshot       = var.rds_skip_final_snapshot
  final_snapshot_identifier = var.rds_skip_final_snapshot ? null : (var.rds_final_snapshot_identifier != "" ? var.rds_final_snapshot_identifier : "${var.project}-mlflow-final")

  # Restauracion: vacio (default) = instancia NUEVA y VACIA. Con valor, la
  # instancia se crea a partir de ese backup. Lo inyectan `task deploy` y
  # `task ops:rebuild` via resolve_restore_snapshot() (tasks/lib/snapshot.sh),
  # cerrando el ciclo backup -> restauracion. Ver #8.5.
  #
  # Al restaurar, AWS conserva db_name/username/password DEL BACKUP y los
  # argumentos de arriba se ignoran; por eso la credencial master vive en la
  # raiz (infra/envs/prod/rds_secret.tf) y sigue siendo la correcta.
  snapshot_identifier = var.rds_snapshot_identifier != "" ? var.rds_snapshot_identifier : null

  backup_retention_period = 7
  backup_window           = "06:00-07:00"
  maintenance_window      = "Mon:07:00-Mon:08:00"

  tags = { Name = "${var.project}-mlflow" }

  # OBLIGATORIO, no cosmetico: snapshot_identifier es ForceNew. Sin este
  # ignore_changes, cualquier apply posterior que no repita el mismo -var
  # (p.ej. el `terraform apply` completo que hace `ops:up` cuando el ALB no
  # existe, tasks/ops.yml) veria "" contra el valor en state y DESTRUIRIA Y
  # RECREARIA el RDS, perdiendo todo lo restaurado. El snapshot solo debe
  # influir en la creacion inicial de la instancia.
  lifecycle {
    ignore_changes = [snapshot_identifier]
  }
}
```

##### 3.5.2.b — `alb.tf` — ALB (load balancer + target group + listener)

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

##### 3.5.2.c — `ecs.tf` (parte 1/2) — ECS cluster + Service Discovery

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

##### 3.5.2.d — `iam.tf` — IAM (execution role + task role)

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
      Resource = var.rds_password_secret_arn # el secret vive en la raiz
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

##### 3.5.2.e — `ecs.tf` (parte 2/2) — Log group + Task Definition + Service

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
> | `aws_ecs_task_definition.mlflow` | **ECS** | `ECS > Task definitions > Create new task definition (JSON)`. **Family**: `ml-training-mlflow`. **Launch type**: Fargate. **OS/Arch**: Linux/x86_64. **CPU/Memory**: 1 vCPU / 3 GB (rightsizing de #9.4.2; antes 2 vCPU / 4 GB). **Task role**: el `mlflow_task` que creaste. **Task exec role**: el `mlflow_exec`. **Container**: name `mlflow`, image `<ecr-url>:v3.12.0`, port 5000, command `mlflow server ...`, env vars + secret `RDS_PASSWORD` desde Secrets Manager, log config awslogs, healthcheck `python urllib /health`. |
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
  # Rightsizing (#9.4.2): 2 vCPU/4 GB -> 1 vCPU/3 GB, ~-$3.10/mes. El server es
  # IO-bound (Postgres + proxy a S3 con --serve-artifacts), no CPU-bound.
  #
  # Por que 3 GB y no 2: `mlflow server` levanta 4 workers gunicorn por defecto
  # (no pasamos --workers) y el trainer loguea 4 variedades en PARALELO. A
  # ~300 MB de RSS por worker, 2 GB queda sin margen para el buffer de artifacts.
  # Los 1 GB extra cuestan $0.30/mes — mas barato que un OOM a mitad de un run.
  # Combos Fargate validos con cpu=1024: memoria 2048..8192 en pasos de 1024.
  # Rollback: volver a "2048"/"4096" si ves OOMKilled en el log group de mlflow.
  cpu                      = "1024" # 1 vCPU
  memory                   = "3072" # 3 GB
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
          # MLflow 3.5+ valida Host. Admitimos únicamente los hosts usados por
          # el ALB y por service discovery interno; nunca '*'.
          "--allowed-hosts ${aws_lb.main.dns_name},${aws_lb.main.dns_name}:*,mlflow.${var.project}.local,mlflow.${var.project}.local:*",
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
          # Modo proxy coherente con local: nuevos runs reciben
          # mlflow-artifacts:/... y los clientes no necesitan permisos S3 para
          # artifacts de MLflow. No mezclar --default-artifact-root con proxy.
          "--artifacts-destination s3://${var.artifacts_bucket}/artifacts",
          "--serve-artifacts"
        ])
      ]
      secrets = [{
        name      = "RDS_PASSWORD"
        valueFrom = var.rds_password_secret_arn # el secret vive en la raiz
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

#### 3.5.3 `modules/mlflow/outputs.tf`

```hcl
output "tracking_uri" { value = "http://${aws_lb.main.dns_name}" }
output "internal_tracking_uri" { value = "http://mlflow.${var.project}.local:5000" }
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
# NOTA: el output `rds_password_secret_arn` se removio al mover el secret a la
# raiz (envs/prod/rds_secret.tf). module.api ahora lo toma directamente de
# aws_secretsmanager_secret.rds.arn, sin pasar por este modulo.
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

#### 3.5.4 Apendear `module "mlflow"` en `infra/envs/prod/main.tf`

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

### 3.6 `modules/reports/` — Fargate nginx sirviendo S3

Sirve `s3://artifacts/{reports,artifacts}/` como sitio estatico bajo
`http://<ALB>/reports/*` y `http://<ALB>/artifacts/*`. Usa el mismo ALB
listener via `path-pattern` rules.

Mecanismo: container = nginx + sidecar de `aws s3 sync` que copia el
bucket a `/usr/share/nginx/html/` cada 60s. Costo: $0.50/mes Fargate +
trafico S3 GET despreciable.

#### 3.6.1 `modules/reports/variables.tf`

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

#### 3.6.2 `modules/reports/main.tf`

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

#### 3.6.3 `modules/reports/outputs.tf`

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

#### 3.6.4 `docker/reports/Dockerfile`

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

#### 3.6.5 `docker/reports/nginx.conf`

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

#### 3.6.6 `docker/reports/entrypoint.sh`

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

#### 3.6.7 Apendear `module "reports"` en `infra/envs/prod/main.tf`

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

### 3.7 `modules/batch/` — Compute envs + queues + job-def + IAM

Pieza critica donde se respeta el contrato del trainer: el container
recibe via CMD `--varieties X --tuning Y` (matchea
`src/orchestration/cli.py:parse_args`) y las env vars S3 que
`main.py:_hydrate_data_from_s3` lee.

#### 3.7.1 `modules/batch/variables.tf`

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

#### 3.7.2 `modules/batch/iam.tf`

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

#### 3.7.3 `modules/batch/main.tf`

> 📂 **Pegar los 4 sub-bloques (a/b/c/d) en**: `infra/modules/batch/main.tf`
> (concatenados, en orden — cada sub-bloque trae el comentario
> `# infra/modules/batch/main.tf (parte N/4)` como recordatorio).

##### 3.7.3.a — Log group (parte 1/4 de `main.tf`)

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

##### 3.7.3.b — Compute Environments (Spot + OnDemand) (parte 2/4 de `main.tf`)

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

##### 3.7.3.c — Job queues (1 por CE) (parte 3/4 de `main.tf`)

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

##### 3.7.3.d — Job definition (contrato con el trainer) (parte 4/4 de `main.tf`)

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

#### 3.7.4 `modules/batch/outputs.tf`

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

#### 3.7.5 Apendear `module "batch"` en `infra/envs/prod/main.tf`

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
  tracking_uri         = module.mlflow.internal_tracking_uri
  artifacts_bucket     = module.storage.artifacts_bucket
  artifacts_bucket_arn = module.storage.artifacts_bucket_arn
  data_bucket_arn      = module.storage.data_bucket_arn
  log_retention_days   = var.log_retention_days
}
```

> **Checkpoint**: batch depende de network (subnets/SG), storage (ECR
> + buckets) y MLflow (`internal_tracking_uri`). Batch llama a MLflow por
> Cloud Map dentro de la VPC; no sale por NAT ni vuelve a entrar por el ALB
> público.

> **Gotcha #3.7**: módulo split en `main.tf + iam.tf`. Usar `name` (no `compute_environment_name` deprecado) — key para AWS provider v6.

> **Gotcha #3.7.b**: en la job definition NO pongas `networkConfiguration` (`assignPublicIp`) — es **solo Fargate**. Con compute environments EC2 el `RegisterJobDefinition` falla con `ClientException: networkConfiguration not applicable for EC2.` (HTTP 400) y el apply muere despues de crear las queues. En EC2 la IP publica la define la subnet/compute environment, no la job-def. `terraform validate` pasa igual porque el rechazo ocurre del lado de AWS, en apply.

---

### 3.8 `modules/monitoring/` — SNS + alarmas

Genera **una alarma MAPE por variedad** (no hardcoded a POP como en V1).
La alarma escucha la custom metric que el trainer va a emitir en Parte 5.

#### 3.8.1 `modules/monitoring/variables.tf`

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

#### 3.8.2 `modules/monitoring/main.tf`

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

#### 3.8.3 `modules/monitoring/outputs.tf`

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

#### 3.8.4 Apendear `module "monitoring"` en `infra/envs/prod/main.tf`

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

### 3.9 `modules/lambdas/` — dispatcher + notifier

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

#### 3.9.1 `modules/lambdas/variables.tf`

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

#### 3.9.2 `modules/lambdas/dispatcher.tf`

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

#### 3.9.3 `modules/lambdas/notifier.tf`

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

#### 3.9.4 `modules/lambdas/outputs.tf`

```hcl
output "dispatcher_function_name" { value = aws_lambda_function.dispatcher.function_name }
```

#### 3.9.5 `infra/lambdas/dispatcher.py`

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

#### 3.9.6 `infra/lambdas/notifier.py`

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

#### 3.9.7 Apendear `module "lambdas"` en `infra/envs/prod/main.tf`

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

### 3.10 `modules/scheduler/` — auto on/off RDS + Fargate

Una Lambda + 2 crons EventBridge. La Lambda hace `start`/`stop` segun
el payload. Antes de stop, chequea Batch jobs RUNNING — si hay, posterga
(no apaga). Lockeado a PET (UTC-5).

> **Orden de pegado**: igual que #3.9, `modules/scheduler/main.tf`
> empaca `infra/lambdas/scheduler.py` con `data "archive_file"`. Pegar
> **primero** #3.10.4 (`scheduler.py`) en `infra/lambdas/scheduler.py`,
> y **despues** #3.10.1-#3.10.3 (los `.tf`). Asi `task infra:validate` /
> el primer apply en Parte 4 no truena por archivo inexistente.

#### 3.10.1 `modules/scheduler/variables.tf`

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
  default = "WED,THU" # ciclo miercoles+jueves (antes: "MON,WED,FRI")
}
variable "log_retention_days" { type = number }
variable "lambdas_src_dir" { type = string }
```

#### 3.10.2 `modules/scheduler/main.tf`

3 sub-bloques al mismo `modules/scheduler/main.tf`:

##### 3.10.2.a — Lambda function + IAM

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

  # Guarda para cortes nocturnos. Con el default actual (16:00 PET -> 21:00 UTC)
  # NO se activa: el stop cae el mismo dia UTC y stop_days == workdays_cron.
  #
  # Se activa con cualquier corte >= 19:00 PET, que cruza medianoche en UTC
  # (20:00 PET = 01:00 UTC del dia SIGUIENTE). Como EventBridge evalua en UTC,
  # ahi los tokens de dia deben correrse uno: parar el miercoles 20:00 PET es
  # `cron(0 1 ? * THU *)`. Sin este shift el stop se adelantaria un dia entero
  # (apagaria el martes por la noche y dejaria el jueves encendido).
  stop_wraps_day = (var.work_end_hour_local - var.tz_offset_hours) >= 24

  # Solo soporta listas por comas ("WED,THU"), no rangos ("MON-FRI"): con un
  # rango habria que expandirlo antes. workdays_cron usa lista por convencion.
  next_day = {
    MON = "TUE", TUE = "WED", WED = "THU", THU = "FRI",
    FRI = "SAT", SAT = "SUN", SUN = "MON"
  }
  stop_days = local.stop_wraps_day ? join(",", [
    for d in split(",", var.workdays_cron) : local.next_day[trimspace(upper(d))]
  ]) : var.workdays_cron
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
      # Propagar workdays + ventana al _keepstop (sino los dias sin ventana
      # quedarian "dentro" y nunca re-pararia el RDS).
      #
      # La ventana viaja en hora LOCAL, no UTC. Con el default (08-16 PET =
      # 13-21 UTC) daria igual, pero en cuanto el corte pasa de las 19:00 PET
      # el rango UTC da la vuelta a medianoche (13..01) y rompe la comparacion
      # `start <= h < end` (13 <= h < 1 es siempre falso -> el keepstop
      # apagaria el RDS cada 6h en plena jornada). _keepstop convierte a local
      # con TZ_OFFSET_HOURS y compara sin wrap, asi la ventana es movible.
      WORKDAYS_CRON    = var.workdays_cron
      WORK_START_LOCAL = tostring(var.work_start_hour_local)
      WORK_END_LOCAL   = tostring(var.work_end_hour_local)
      TZ_OFFSET_HOURS  = tostring(var.tz_offset_hours)
    }
  }

  depends_on = [aws_cloudwatch_log_group.scheduler]
}
```

##### 3.10.2.b — EventBridge rules (start, stop, keepstop)

3 crons: `start` (8 AM PET), `stop` (12 PM PET), `keepstop` (cada 6h
defensa contra el auto-arranque de RDS post-7-dias-stopped). El offset
PET→UTC se calcula en `locals` y se enchufa al `cron(0 H ? * WED,THU *)`.
El `stop` usa `local.stop_days` (los tokens corridos un dia) porque con corte
a las 20:00 PET la hora UTC cae al dia siguiente — ver el `locals` de #3.10.2.a.

> **Equivalente en AWS Console**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_cloudwatch_event_rule.start` | **EventBridge** | `EventBridge > Rules > Create rule`. **Name**: `ml-training-start`. **Event bus**: default. **Rule type**: Schedule. **Schedule pattern**: A fine-grained schedule that runs at a specific time. **Cron expression**: `cron(0 13 ? * WED,THU *)` (13 UTC = 08:00 PET; default workdays_cron). |
> | `aws_cloudwatch_event_target.start` | **EventBridge** | Dentro de la rule: `Add target > Lambda function > ml-training-scheduler`. **Configure target input**: Constant (JSON text): `{"action": "start"}`. |
> | `aws_cloudwatch_event_rule.stop` + target | **EventBridge** | Mismo wizard, name `-stop`, cron `cron(0 21 ? * WED,THU *)` (21 UTC = 16:00 PET, mismo dia), input `{"action":"stop"}`. Si movés el corte a ≥19:00 PET los días se corren solos vía `local.stop_days` — ver #3.10.2.a. |
> | `aws_cloudwatch_event_rule.rds_keepstop` + target | **EventBridge** | Mismo wizard, name `-rds-keepstop`, **Schedule pattern**: A schedule that runs at a regular rate. **Rate expression**: `rate(6 hours)`. Input `{"action":"keepstop"}`. |
>
> **Conceptualmente — por qué 3 rules y no 1 sola**:
> - Una sola rule con 3 targets ejecutaría los 3 inputs en cada tick. Cada rule tiene 1 propósito y 1 cron.
> - **Cron de EventBridge**: `cron(min hour day-of-month month day-of-week year)` (6 campos, no los 5 de Linux). El `?` = "no me importa" (mutuamente excluyente entre day-of-month y day-of-week).
> - **`WED,THU`**: este repo opera 2 días; cualquier subset funciona. Usa lista por comas, no rangos — `local.stop_days` no expande `MON-FRI`.
> - **`rate(6 hours)` para keepstop**: RDS dejado `stopped` >7 días lo enciende AWS solo "por mantenimiento" (~$15 sorpresa/mes). El keepstop lo re-apaga si lo encuentra `available` fuera de ventana.

```hcl
resource "aws_cloudwatch_event_rule" "start" {
  name                = "${var.project}-start"
  description         = "${var.workdays_cron} ${var.work_start_hour_local}:00 PET start RDS+Fargate"
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
  description         = "${var.workdays_cron} ${var.work_end_hour_local}:00 PET stop RDS+Fargate"
  schedule_expression = "cron(0 ${local.stop_hour_utc} ? * ${local.stop_days} *)"
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

##### 3.10.2.c — Permissions (EventBridge → Lambda)

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

#### 3.10.3 `modules/scheduler/outputs.tf`

```hcl
# scheduler no expone outputs: opera por side-effects (Lambda scheduler +
# EventBridge start/stop rules). El parent (envs/prod) no consume nada de aqui.
```

#### 3.10.4 `infra/lambdas/scheduler.py`

> **NOTA**: este es el **archivo real de produccion** `infra/lambdas/scheduler.py`
> (paridad 1:1 con el repo). Incluye los patches integrados:
> - **Patch 13.1 — ventana configurable**: `_parse_workdays` parsea `WORKDAYS_CRON`
>   (`WED,THU`) y `_keepstop` usa `WORK_START_LOCAL`/`WORK_END_LOCAL`/`TZ_OFFSET_HOURS`
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
            Ventana parametrizada via WORKDAYS_CRON + WORK_START_LOCAL +
            WORK_END_LOCAL + TZ_OFFSET_HOURS (se evalua en hora local, no UTC).
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

# Patch 13.1: workdays + horas configurables via env (default = ciclo
# WED,THU 08-16 PET). EventBridge cron usa los mismos tokens (WED,THU).
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


def _set_desired(service: str, count: int) -> bool:
    """update_service tolerante a service inexistente o INACTIVE.

    `task sleep` (hibernacion) destruye el ALB y arrastra el service de MLflow
    (depends_on del listener). Los triggers que siguen vivos mientras tanto
    (batch_autostop, cron stop si esta habilitado) no deben romper por eso.
    """
    try:
        ecs.update_service(cluster=ECS_CLUSTER, service=service, desiredCount=count)
        log.info("ecs %s -> desiredCount=%d", service, count)
        return True
    except (ecs.exceptions.ServiceNotFoundException,
            ecs.exceptions.ServiceNotActiveException):
        log.warning("ecs %s no existe o esta INACTIVE (stack hibernado?) -> skip", service)
        return False


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
    if _set_desired(ECS_SVC_MLFLOW, 1):
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
        _set_desired(svc, 1)
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
        _set_desired(svc, 0)

    # RDS: stop si esta RUNNING
    db = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE)["DBInstances"][0]
    state = db["DBInstanceStatus"]
    if state == "available":
        rds.stop_db_instance(DBInstanceIdentifier=RDS_INSTANCE)
        log.info("rds stop_db_instance ack")
    else:
        log.info("rds en estado %s (skip stop)", state)


def _keepstop():
    """Defense: si RDS quedo RUNNING fuera de ventana, re-pararlo (Patch 13.1).

    La ventana se evalua en hora LOCAL (PET), no UTC. Con el default (08-16 PET)
    daria lo mismo, pero con un corte >= 19:00 PET el rango en UTC pasa a ser
    13..01 — cruza medianoche y `start <= h < end` nunca se cumple, con lo que
    TODO instante contaria como "fuera de ventana" y el keepstop pararia el RDS
    a mitad de jornada. En local el rango no da la vuelta y el weekday tampoco
    se corre de dia, asi la ventana se puede mover sin romper nada.
    """
    log.info("=== KEEPSTOP ===")
    workdays = _parse_workdays(os.environ.get("WORKDAYS_CRON", "WED,THU"))
    start_local = int(os.environ.get("WORK_START_LOCAL", "8"))
    end_local = int(os.environ.get("WORK_END_LOCAL", "16"))
    offset = int(os.environ.get("TZ_OFFSET_HOURS", "-5"))

    now_local = time.gmtime(time.time() + offset * 3600)
    hour = now_local.tm_hour
    weekday = now_local.tm_wday   # 0=lunes, ya en local
    in_window = (weekday in workdays) and (start_local <= hour < end_local)
    if in_window:
        log.info("dentro de ventana (local=%02d:00, weekday=%d, workdays=%s), skip",
                 hour, weekday, sorted(workdays))
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

#### 3.10.5 Apendear `module "scheduler"` en `infra/envs/prod/main.tf`

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
  # Sin esta linea el modulo cae a su propio default y el valor de envs/prod
  # se ignora en silencio (los crons quedarian en dias distintos al esperado).
  workdays_cron            = var.workdays_cron
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

### 3.11 `modules/cicd/` — OIDC trust + GHA roles

Dos roles: `gha-deploy` (terraform apply, push ECR) y `gha-train`
(invoke Lambda dispatcher). El trust acepta únicamente `main` y el GitHub
Environment `production`; una PR o cualquier branch no protegida no puede
obtener credenciales AWS.

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
> 4. El `sub` debe coincidir exactamente con `main` o con el environment
>    `production`; no se usa el wildcard `repo:<org>/<repo>:*`.
>
> Si el repo en GitHub cambia de nombre/org, hay que **regenerar tanto**
> el OIDC provider (#2.5) **como** los roles (este módulo): el `sub`
> hardcoded del trust no se ajusta solo.

#### 3.11.1 `modules/cicd/variables.tf`

```hcl
variable "project" { type = string }
variable "github_org" { type = string }
variable "github_repo" { type = string }
variable "oidc_provider_arn" { type = string }
```

#### 3.11.2 `modules/cicd/main.tf`

> **Equivalente en AWS Console — los 2 IAM Roles con trust OIDC**:
>
> | Recurso Terraform | Servicio | Que harias click-a-click |
> |---|---|---|
> | `aws_iam_role.deploy` con trust OIDC | **IAM** | Crear rol Web identity para GitHub, audience `sts.amazonaws.com`, limitado a `main` y al environment `production`. Nunca dejar branch vacía ni usar `repo:*`. |
> | `aws_iam_role.train` con mismo trust | **IAM** | Mismo wizard de Web identity. **Permissions**: SOLO `lambda:InvokeFunction` sobre el dispatcher + `batch:Describe/ListJobs` + `logs:GetLogEvents`. **Name**: `ml-training-gha-train`. |
>
> **Conceptualmente — el flujo OIDC paso a paso**:
> 1. GHA arranca un workflow con `permissions: id-token: write`.
> 2. GH genera un **JWT** firmado con claims `iss`, `aud=sts.amazonaws.com`, `sub=repo:org/ml_training:ref:refs/heads/main`, etc.
> 3. `aws-actions/configure-aws-credentials@v4` manda el JWT a STS (`sts:AssumeRoleWithWebIdentity`, `RoleArn=ml-training-gha-deploy`).
> 4. STS valida firma, `aud` y un `sub` exacto de `main` o `environment:production`.
> 5. STS devuelve credenciales temporales (~1h) con los permisos del rol.
> 6. **No hay secrets de larga duración en GitHub** — la gran ventaja vs `AWS_ACCESS_KEY_ID`/`SECRET` eternos como GH Secrets.
> - **Trust exacto**: limita repo y contexto. Restringir solo el repo todavía
>   permitiría que una PR o branch no protegida intentara asumir el rol.
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
      #   - trust policy limitada a main + environment production.
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

#### 3.11.3 `modules/cicd/outputs.tf`

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

#### 3.11.4 Apendear `module "cicd"` en `infra/envs/prod/main.tf`

Ultimo bloque. Pegar AL FINAL de `infra/envs/prod/main.tf` (despues
de `module "scheduler"` de #3.10.5):

```hcl
# -------------------------------------------------------------------------
# Capa 9: CI/CD (GHA IAM roles confiando en OIDC)
# -------------------------------------------------------------------------
module "cicd" {
  source = "../../modules/cicd"
  count  = var.enable_cicd ? 1 : 0

  project           = var.project
  github_org        = var.github_org
  github_repo       = var.github_repo
  oidc_provider_arn = data.aws_iam_openid_connect_provider.github[0].arn
}
```

> **Nota CI/CD opcional**: tanto el `data "aws_iam_openid_connect_provider"` (#3.2.5)
> como `module.cicd` van gateados por `var.enable_cicd` (default `false`). Con el
> default, el stand-up completo (storage→network→mlflow→batch→api/ui) corre **sin**
> `bash infra/bootstrap-oidc.sh` y **sin** `github_org`/`github_repo`. Los outputs
> `gha_*_role_arn` devuelven `null` con CI/CD apagado. Para activarlo: corré
> `bootstrap-oidc.sh` (#2.5), poné `enable_cicd=true` (+ `github_org`/`github_repo`)
> y re-aplicá.

> **Checkpoint final**: con este bloque pegado, el `main.tf` esta
> **completo** (los 9 modulos + el `data` source del OIDC provider). La validación
> sintáctica (`fmt -check` + `validate`) se ejecuta en #4.2, justo antes
> del primer apply — no acá, porque entre que terminás Parte 3 y arrancás
> Parte 4 pueden pasar días.

> **Gotcha #3.11**: el modulo `cicd` solo recibe `project`, `github_org`, `github_repo` y `oidc_provider_arn`. Las vars `*_arn` (buckets, ECR, queues, job-def) que en versiones previas se pasaban como pass-through fueron **eliminadas** del modulo y de su `variables.tf` por estar sin uso — el `main.tf` de cicd construye los ARNs que necesita con su propio `data "aws_caller_identity"` / `data "aws_region"`.

---

### 3.11.5 `modules/consumer-iam/` — Rol OIDC para repo consumer EXTERNO (Patch 13.5, OPCIONAL)

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

#### 3.11.5.1 `modules/consumer-iam/variables.tf`

```hcl
variable "project" { type = string }
variable "artifacts_bucket_arn" { type = string }
variable "consumer_oidc_arn" { type = string }
variable "consumer_org" { type = string }
variable "consumer_repo" { type = string }
```

#### 3.11.5.2 `modules/consumer-iam/main.tf`

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

#### 3.11.5.3 `modules/consumer-iam/outputs.tf`

```hcl
output "consumer_role_arn" { value = aws_iam_role.consumer.arn }
```

#### 3.11.5.4 Apendear `module "consumer_iam"` en `infra/envs/prod/main.tf`

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
  # Reusa el `data` del OIDC provider de GitHub (gateado por enable_cicd):
  # este modulo OPCIONAL requiere enable_cicd=true (+ bootstrap-oidc en #2.5).
  consumer_oidc_arn    = data.aws_iam_openid_connect_provider.github[0].arn
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

### 3.12 App stack — `modules/api` (FastAPI) + `modules/ui` (Streamlit) — Capa 4.5

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

#### 3.12.1 `modules/api/variables.tf`

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

#### 3.12.2 `modules/api/main.tf`

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

#### 3.12.3 `modules/api/outputs.tf`

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

#### 3.12.4 `api/Dockerfile` (contexto de build = RAÍZ del repo)

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

#### 3.12.5 `modules/ui/variables.tf`

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

#### 3.12.6 `modules/ui/main.tf`

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

#### 3.12.7 `modules/ui/outputs.tf`

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

#### 3.12.8 `ui/Dockerfile` (contexto de build = `ui/`)

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

#### 3.12.9 Apendear `module "api"` + `module "ui"` en `infra/envs/prod/main.tf`

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

## Parte 4 — Apply incremental + smoke test

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

### 4.1 Setup Task (orquestador local)

[Task](https://taskfile.dev) es un task runner single-binary (Go, ~10 MB)
que orquesta Terraform + Docker + Lambda + Batch desde una sola lista
descubrible (`task --list`). Si conocés Make, lo leés en 5 min.

Cada task es **idempotente y composable**: si falla en el paso 5/10, lo
corres de nuevo y empieza donde quedo (Terraform mismo es idempotente;
los hash de `sources:` evitan re-buildear imagenes ya construidas).

#### 4.1.1 Verificar / instalar Task

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

#### 4.1.2 Estructura final

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

#### 4.1.3 `tasks/infra.yml` — Terraform wrapper + bootstrap

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
#   task infra:reset-state                            borra tfstate remoto + .terraform (fresh start; no toca AWS)
# =============================================================================

version: "3"

vars:
  # TF_DIR y RDS_ID se inyectan desde el Taskfile raiz (fuente unica, compartida
  # con ops:). Ver `includes:` en Taskfile.yml.
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

  _init_validate:
    internal: true
    cmds:
      # Validación sintáctica no necesita credenciales ni acceso al state.
      - terraform -chdir={{.TF_DIR}} init -backend=false

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
    desc: "terraform fmt -check + validate sin backend remoto ni credenciales AWS"
    deps: [_init_validate]
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

  reset-state:
    desc: "Borra el tfstate remoto de envs/prod en S3 (todas las versiones + .tflock) y el .terraform local, para arrancar de cero. NO toca recursos AWS vivos: solo hace que Terraform los 'olvide'. Util al re-empezar en una cuenta con state viejo."
    prompt: "Esto borra el tfstate remoto de envs/prod (Terraform 'olvidara' los recursos, que SEGUIRAN vivos en AWS). Continuar?"
    cmds:
      # Borra todas las versiones del objeto tfstate + su .tflock en el bucket de state.
      - |
        export BUCKET="{{.PROJECT}}-tfstate-{{.SUFFIX}}"
        for KEY in envs/prod/terraform.tfstate envs/prod/terraform.tfstate.tflock; do
          aws s3api list-object-versions --bucket "$BUCKET" --prefix "$KEY" \
            --query '[Versions,DeleteMarkers][].{Key:Key,VersionId:VersionId}' --output json 2>/dev/null \
          | python3 -c 'import sys,json,subprocess,os; b=os.environ["BUCKET"]; [subprocess.run(["aws","s3api","delete-object","--bucket",b,"--key",o["Key"],"--version-id",o["VersionId"]]) for o in (json.load(sys.stdin) or [])]'
        done
      - rm -rf {{.TF_DIR}}/.terraform {{.TF_DIR}}/.terraform.lock.hcl
```

`_init` es `internal: true` (no aparece en `task --list`) y se dispara via `deps:` cuando hace falta. `SUFFIX` se resuelve dinamicamente con `aws sts get-caller-identity` — si cambias de cuenta AWS, el backend bucket name cambia con vos.

#### 4.1.4 `tasks/ecr.yml` — build + push 5 imagenes

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

#### 4.1.5 `tasks/batch.yml` — submit via Lambda dispatcher

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
#   task batch:train VARIETIES=POP                    una variedad (background)
#   task batch:train VARIETIES=POP,VENTURA            varias en un job
#   task batch:train VARIETIES=all                    todas las permitidas
#   task batch:train VARIETIES=POP TUNING=smoke       sanity check (~1 min)
#   task batch:train VARIETIES=POP WAIT=true          bloquear hasta terminar
#   task batch:smoke                                  atajo: POP + smoke (bloquea)
#   task batch:eda   VARIETIES=POP                    EDA exploratorio (opcional, on-demand)
#   task batch:watch                                  seguir el ULTIMO job hasta terminar
#   task batch:logs                                   tail de logs del ultimo job (FOLLOW=true en vivo)
#   task batch:status                                 jobs activos en queues
#   task batch:cancel                                 terminar el ultimo job (o JOB_ID=<id>)
#
# El submit es FIRE-AND-FORGET por defecto: el job corre en AWS Batch, asi que
# cerrar la terminal o apagar la maquina NO lo detiene. watch/logs/cancel se
# defaultean al ultimo job submiteado (persistido en .batch-last-job).
# =============================================================================

version: "3"

vars:
  DISPATCHER_FN: '{{.DISPATCHER_FN | default (printf "%s-dispatcher" .PROJECT)}}'
  JOBDEF:        '{{.JOBDEF        | default (printf "%s-trainer" .PROJECT)}}'
  QUEUE_SPOT:    '{{.QUEUE_SPOT    | default (printf "%s-job-queue-spot"     .PROJECT)}}'
  QUEUE_OD:      '{{.QUEUE_OD      | default (printf "%s-job-queue-ondemand" .PROJECT)}}'
  TUNING:        '{{.TUNING        | default "prod_xl"}}'
  # WAIT=false por defecto -> submit en background (el job vive en AWS Batch).
  # batch:smoke pisa WAIT=true para que el smoke de deploy siga verificando exito.
  WAIT:          '{{.WAIT          | default "false"}}'
  LOG_GROUP:     '{{.LOG_GROUP     | default (printf "/aws/batch/%s" .PROJECT)}}'

tasks:

  train:
    desc: "Entrena en Batch (background). Vars: VARIETIES=POP[,VENTURA|all] (REQ), TUNING, WAIT=true para bloquear"
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
        batch_record_job "$JOB_ID"
        if [ "{{.WAIT}}" != "true" ]; then
          echo "  El job corre en AWS Batch: podes cerrar la terminal sin afectarlo."
          echo "  Estado: task batch:watch  |  Logs: task batch:logs  |  Cancelar: task batch:cancel"
          exit 0
        fi
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
      # WAIT=true explicito: el smoke del deploy DEBE bloquear y verificar exito.
      - task: train
        vars: { VARIETIES: POP, TUNING: smoke, WAIT: "true" }

  watch:
    desc: "Seguir un job hasta SUCCEEDED/FAILED. Vars: JOB_ID (default: ultimo submiteado)"
    silent: true
    cmds:
      - |
        source tasks/lib/batch_wait.sh
        JOB_ID=$(batch_need_job "{{.JOB_ID}}") || exit 0
        follow_job "$JOB_ID" "watch"   # nunca rompe la terminal (exit 0 aun en FAILED)

  logs:
    desc: "Tail de logs (CloudWatch) de un job. Vars: JOB_ID (default: ultimo), FOLLOW=true para vivo"
    silent: true
    cmds:
      - |
        source tasks/lib/batch_wait.sh
        JOB_ID=$(batch_need_job "{{.JOB_ID}}") || exit 0
        tail_job_logs "$JOB_ID" "{{.LOG_GROUP}}" "{{.FOLLOW | default \"false\"}}"

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
    desc: "Terminar un job. Vars: JOB_ID (default: ultimo submiteado), REASON=<texto>"
    silent: true
    cmds:
      - |
        source tasks/lib/batch_wait.sh
        JOB_ID=$(batch_need_job "{{.JOB_ID}}") || exit 0
        # terminate-job es idempotente: sobre un job ya terminal es no-op.
        aws batch terminate-job --job-id "$JOB_ID" \
          --reason "{{.REASON | default \"cancelled via task\"}}" \
          && echo ">>> terminate enviado a job=$JOB_ID"
```

Una sola via de submit (el Lambda dispatcher) — valida que la variety exista en `data/training/DB-HISTORICA.xlsx` antes de encolar y maneja hydrate desde S3. El path raw con `aws batch submit-job` queda disponible si alguna vez lo necesitas.

#### 4.1.6 `tasks/ops.yml` — lifecycle cluster + MLflow registry (Dia 2)

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
  # TF_DIR / RDS_ID / BACKUP_MAX_AGE_MIN se inyectan desde el Taskfile raiz
  # (fuente unica, compartida con infra:). Ver `includes:` en Taskfile.yml.
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
      - aws rds describe-db-instances --db-instance-identifier {{.RDS_ID}}
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
    desc: "Encender stack (idempotente: recrea ALB+NAT si down los libero, invoca scheduler.start y espera RDS+ALB healthy)"
    silent: true
    cmds:
      - |
        set -e
        # down (hibernacion) destruyo el ALB y todo lo que dependia de el
        # (NAT, listener+rules, ECS services, task defs, modulos ui/reports
        # via depends_on, lambdas scheduler/dispatcher via outputs). Un apply
        # COMPLETO (como rebuild) recrea exactamente lo que falte y es
        # inmune a que el grafo de dependencias arrastre mas de lo previsto.
        if ! aws elbv2 describe-load-balancers --names {{.PROJECT}}-alb >/dev/null 2>&1; then
          echo ">>> ALB no existe (stack hibernado) -> terraform apply completo (~3-5 min)..."
          terraform -chdir={{.TF_DIR}} apply -auto-approve
          # El DNS del ALB cambia al recrearlo -> descartar cualquier valor
          # stale del env y resolver fresco desde terraform output.
          unset MLFLOW_ALB_DNS
        fi
        source tasks/lib/wake.sh
        PROJECT={{.PROJECT}} TF_DIR={{.TF_DIR}} SCHEDULER_FN={{.SCHEDULER_FN}} wake_cluster

  down:
    desc: "Apagar stack + liberar ALB y NAT (aborta si Batch RUNNING). Vars: COOLDOWN=N, RELEASE_NET=false (mantiene red, wake instantaneo)"
    vars:
      COOLDOWN:    '{{.COOLDOWN | default "0"}}'
      RELEASE_NET: '{{.RELEASE_NET | default "true"}}'
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
      # Hibernacion: ALB (~$16/mes) + NAT (~$33/mes) + IPv4 publicas (~$7/mes)
      # son el costo idle dominante con el stack "dormido". El destroy targeted
      # del ALB arrastra a TODOS sus dependientes en el grafo: listener + rules,
      # los 4 ECS services + task defs, los modulos ui/reports completos
      # (depends_on de modulo), la alarma alb-5xx y las lambdas scheduler/
      # dispatcher (sus env vars usan outputs de los services/ALB). Todo es
      # stateless y `up` lo recrea con un apply completo. RDS/S3 no se tocan.
      # OJO: (1) el DNS del ALB cambia en cada ciclo sleep/wake;
      #      (2) sin la lambda keepstop, si la hibernacion pasa de 7 dias AWS
      #          re-arranca el RDS solo y nadie lo re-para -> para idle largo
      #          usar `task ops:teardown` (destruye el RDS respaldandolo antes).
      - |
        if [ "{{.RELEASE_NET}}" != "true" ]; then
          echo ">>> RELEASE_NET=false -> ALB + NAT quedan encendidos (~\$1.8/dia idle)"
          exit 0
        fi
        set -e
        echo ">>> Liberando ALB + dependientes (el DNS cambiara al recrearlo)..."
        terraform -chdir={{.TF_DIR}} destroy -target=module.mlflow.aws_lb.main -auto-approve
        echo ">>> Liberando NAT gateway + EIP (enable_nat=false)..."
        terraform -chdir={{.TF_DIR}} apply -target=module.network -var enable_nat=false -auto-approve
        echo "OK stack hibernado (piso ~\$4/mes: storage + Route53). Volver: task wake"
        echo "AVISO: hibernacion >7 dias -> RDS auto-arranca (keepstop hibernado); usar teardown para idle largo"

  # ═══ Teardown / Rebuild ═════════════════════════════════════════════════════

  teardown:
    desc: "Down + terraform destroy de modulos volatiles. Preserva storage + network"
    prompt: "Destruira los modulos volatiles. Storage (S3+ECR) y network (VPC) quedan. Continuar?"
    cmds:
      # RELEASE_NET=false: liberar ALB/NAT aqui seria redundante — el destroy
      # de module.mlflow y el apply enable_nat=false de abajo hacen lo mismo.
      - task: down
        vars: { RELEASE_NET: "false" }
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
          | jq '.registered_model.latest_versions[] | {version, aliases, run_id, creation_timestamp}'

  promote:
    desc: "Validar y reasignar alias @champion. Vars: MODEL_NAME (REQ), VERSION=N (REQ), MAX_MAPE=20"
    requires:
      vars: [MODEL_NAME, VERSION]
    cmds:
      - python scripts/promote_model.py {{.MODEL_NAME}} {{.VERSION}} --max-mape {{.MAX_MAPE}} --tf-dir {{.TF_DIR}}
```

**Decisiones clave** (resumidas — el codigo es la fuente):

- `ops:up` es **idempotente**: pre-check `/health` y solo invoca `scheduler.start` si esta DOWN. Lo usan tanto el operador manual como el flujo CI auto-train (el `wake.sh` escribe el estado previo a `/tmp/wake-status` para que el workflow decida si tiene que apagar al final).
- `ops:down` con `COOLDOWN=N` (default 0) cubre el "down ahora" manual y el "espera N seg y apaga" del CI post-train con una sola task.
- `ops:teardown` preserva `module.network` (VPC/subnets/SGs) y `module.storage`, pero **libera el NAT** vía `terraform apply -target=module.network -var enable_nat=false` (el NAT cuesta ~$33/mes idle; el resto de network no cuesta encendido). `task rebuild`/`deploy` lo recrean (default `enable_nat=true`). Resultado: hibernado ~$1/mes (storage + backups del RDS), sin tener que destruir toda la red.
- `ops:promote` delega en `scripts/promote_model.py` (Python con
  `MlflowClient`) — valida calidad absoluta y comparación contra
  `@champion`, luego usa `set_registered_model_alias`. En producción se invoca
  desde el workflow protegido, no directamente desde una laptop.

#### 4.1.7 Helpers `tasks/lib/*.sh`

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
  local bucket="$1" delete="${2:-false}" prefix="${3:-}"
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

#### 4.1.8 Root `Taskfile.yml` — `includes` + atajos high-level

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
#
# CONVENCION DE VARS COMPARTIDAS: todo nombre que use mas de un archivo se
# declara UNA vez en el `vars:` de abajo y se INYECTA aca. El include no lo
# redeclara — si lo hiciera, tendriamos dos defaults capaces de divergir en
# silencio (paso con TF_DIR, declarado identico en infra.yml y ops.yml).
# Aplica a: TF_DIR, RDS_ID, QUEUE_SPOT, QUEUE_OD, BACKUP_MAX_AGE_MIN.
includes:
  infra:
    taskfile: ./tasks/infra.yml
    vars: { PROJECT: '{{.PROJECT}}', REGION: '{{.REGION}}', TF_DIR: '{{.TF_DIR}}', RDS_ID: '{{.RDS_ID}}' }
  ecr:
    taskfile: ./tasks/ecr.yml
    vars: { PROJECT: '{{.PROJECT}}', REGION: '{{.REGION}}' }
  batch:
    taskfile: ./tasks/batch.yml
    vars: { PROJECT: '{{.PROJECT}}', REGION: '{{.REGION}}', QUEUE_SPOT: '{{.QUEUE_SPOT}}', QUEUE_OD: '{{.QUEUE_OD}}' }
  ops:
    taskfile: ./tasks/ops.yml
    vars: { PROJECT: '{{.PROJECT}}', REGION: '{{.REGION}}', QUEUE_SPOT: '{{.QUEUE_SPOT}}', QUEUE_OD: '{{.QUEUE_OD}}',
            TF_DIR: '{{.TF_DIR}}', RDS_ID: '{{.RDS_ID}}', BACKUP_MAX_AGE_MIN: '{{.BACKUP_MAX_AGE_MIN}}' }
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
      #
      # LIMITACION CONOCIDA: `ops:state-drift` solo cubre buckets S3. Un nuke que
      # falla a la mitad puede dejar OTROS recursos con nombre fijo huerfanos
      # -verificado el 2026-07-20 con `ml-training-tg-api`, `ml-training-tg-mlflow`
      # y el DB subnet group `ml-training-rds-subnets`-, y el apply falla con
      # "already exists". Si pasa: comprobar que esten desligados (target groups
      # con 0 LoadBalancerArns; subnet group apuntando a una VPC inexistente) y
      # borrarlos a mano con `aws elbv2 delete-target-group` /
      # `aws rds delete-db-subnet-group` antes de reintentar.
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

##### 4.1.8.1 Expandir el `default` (menu unico local + AWS)

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
            task ops:promote MODEL_NAME=... VERSION=N  mover alias @champion

        ▸ AWS — Teardown / Recovery / Destroy
            task teardown           backup + destroy volatiles (preserva storage + network)
            task rebuild            restaura el RDS del backup + up  (recovery, no destructivo)
            task backups            listar los backups del RDS restaurables
            task ops:backup-now     backup manual sin destruir nada
            task ops:state-drift    detecta buckets S3 en AWS ausentes en tfstate
                                    (causa BucketAlreadyExists en apply; corre auto en deploy)
            task destroy            DESTRUCTIVO: S3 + ECR + RDS + todo lo demas
                                    respalda el RDS, pero VACIA S3: los artifacts no vuelven
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
                      docs/02-produccion-aws.md  (guia paso a paso AWS)
        EOF
```

> **Por qué cuatro secciones AWS y no una** — separa por ciclo de vida (bootstrap / primer stand-up / operación día-2 / destroy) para mostrar el **orden temporal**, no un listado plano. "Primer stand-up" embebe la secuencia 0→7 del mapa del camino (#4.1.10), así quien corre `task` ve qué ejecutar y en qué orden. "Teardown / Recovery / Destroy" agrupa los 4 comandos de ciclo de vida (los labels `DESTRUCTIVO` van por línea — rebuild **no** lo es).

#### 4.1.9 Verificacion

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

#### 4.1.10 Mapa del camino: del plan a `/reports/POP/` visible

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

### 4.2 Verificación sintáctica previa al apply

Antes del primer `task infra:apply` (Ola A), una sola verificación: que el HCL parsea y formatea. Esto cuesta ~10 s y atrapa typos antes de un apply de varios minutos. La validación se hace **acá, no en Parte 3**, porque entre que terminás de pegar los módulos y arrancás el deploy pueden pasar días — re-validar justo antes del apply es lo único que importa.

```bash
# Prereq: `source scripts/prod.env` en esta terminal (sección 3.5).
task infra:validate
# Esperado: "Success! The configuration is valid."
```

> **Qué hace la task**: `_init` (descarga providers + backend S3) → `terraform fmt -check -recursive` → `terraform validate`. Si falla `fmt -check`, correr `terraform fmt -recursive infra/` para auto-arreglar. Si falla `validate`, el error indica archivo y línea — fijar y re-correr. **No avanzar a 4.3 hasta que diga "Success"**.

> **Gotcha #4.2**: `validate` cold (sin `.terraform/`) funciona porque la task lleva `deps: [_init]`. Si igual falla con "Module not installed", correr `task infra:plan` una vez para forzar el init.

---

### 4.3 Ola A — apply storage solo

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

#### Verificacion Ola A

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

#### Subir el Excel inicial al bucket de data

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

### 4.4 Ola B — build + push 5 imagenes a ECR

Las 5 imagenes son:

| Imagen | Dockerfile | Contexto | Tag | Para que |
|---|---|---|---|---|
| `ml-training` | `./Dockerfile` (raiz, ya existe) | raiz | `latest` + `sha-<git-sha>` | Trainer en AWS Batch |
| `ml-training-mlflow` | `./docker/mlflow/Dockerfile` (ya existe en local) | raiz | `v3.12.0` | MLflow server en Fargate |
| `ml-training-reports` | `./docker/reports/Dockerfile` (creado en 3.6.4) | raiz | `stable` | Nginx que sirve S3 |
| `ml-training-api` | `./api/Dockerfile` (#3.12.4) | **raiz** (necesita `src/`) | `latest` + `sha-<git-sha>` | API FastAPI en Fargate |
| `ml-training-ui` | `./ui/Dockerfile` (#3.12.8) | **`ui/`** | `latest` + `sha-<git-sha>` | UI Streamlit en Fargate |

#### 4.4.1 Como funciona

La task `ecr:build-all` (definida en sección 4.1.4) encadena 5 invocaciones de
`ecr:build` con `IMG=trainer/mlflow/reports/api/ui`. Cada una hace
`docker build` con build args (`GIT_SHA`, `BUILD_DATE`, `VERSION`) y
pushea 2 tags: el solicitado (`latest`/`v3.12.0`/`stable`) y
`sha-<git-sha-corto>` para rollback determinista. El **contexto** de build lo
resuelve la task: `api` usa la raiz (para incluir `src/`), `ui` usa `ui/`.

#### 4.4.2 Overrides via variables CLI

```bash
# Override del tag (e.g. bump version de MLflow)
task ecr:build IMG=mlflow TAG=v3.13.0

# Solo trainer (re-build despues de cambio de codigo)
task ecr:build IMG=trainer
```

#### 4.4.3 Ejecutar

```bash
# Prereq: `source scripts/prod.env` (sección 3.5). La task `ecr:build-all` computa
# ACCOUNT_ID y la URI de ECR internamente, no las lee del shell.
task ecr:build-all
```

#### Verificacion Ola B

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

### 4.5 Ola C — apply full (en 4 sub-olas con checkpoint)

Ahora todo el resto (`network`, `mlflow`, `reports`, `batch`,
`monitoring`, `lambdas`, `scheduler`, `cicd`). Tiempo total: ~15-20
min. El que mas demora: RDS create (~8 min) + ALB warmup + Fargate
task launch (~3 min).

**Por que se parte en 4 sub-olas**: un `terraform apply` monolitico
falla "en silencio" — si el modulo 5 de 8 explota, te enteras 18 min
despues. Partiendo por capa de dependencia, cada checkpoint da
feedback en 2-5 min y el error es localizable. Tambien permite saltar
a la siguiente sub-ola sin re-planear las anteriores.

#### 4.5.1 Sub-ola C1 — `network` (red base, ~1-2 min)

```bash
task infra:plan TARGET=module.network
task infra:apply TARGET=module.network
```

> **En consola AWS**: VPC console → Your VPCs → `ml-training-vpc`;
> Subnets → 2 (public + private); NAT Gateways → 1 (state=available);
> Security Groups → 6 (`ml-training-sg-alb`, `-sg-mlflow`, `-sg-batch`,
> `-sg-rds`, `-sg-api`, `-sg-ui`).

#### 4.5.2 Sub-ola C2 — `mlflow` + `reports` + `api` + `ui` (RDS + ALB + 4 Fargate, ~12 min)

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

#### 4.5.3 Sub-ola C3 — `batch` + `lambdas` (compute + orquestacion, ~3 min)

> **Nota de dependencias**: `module.lambdas` referencia `module.monitoring.sns_topic_arn` para el notifier. Al usar `terraform apply -target=module.lambdas`, Terraform arrastra `module.monitoring` automaticamente — asi que aunque el orden documentado es C3 (batch + lambdas) antes que C4 (monitoring), en la practica `monitoring` se crea durante C3 como dependencia transitiva. C4 entonces es no-op para `monitoring`.

```bash
task infra:apply TARGET=module.batch
task infra:apply TARGET=module.lambdas
```

> **En consola AWS**: Batch → Job queues (2: spot + ondemand, ambos
> VALID); Compute environments (2: ml-training-ce-spot, -ondemand);
> Job definitions → `ml-training-trainer`; Lambda → Functions (2:
> dispatcher + notifier).

#### 4.5.4 Sub-ola C4 — `monitoring` + `scheduler` + `cicd` (~2 min)

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

#### 4.5.5 Apply full alternativo (cuando ya pasaste por C1-C4 una vez)

Para re-applies idempotentes (despues de algun cambio menor), una vez
validado que todo arriba existe, podes usar:

```bash
task infra:plan
task infra:apply
```

> **Cuando usar el apply monolitico**: re-deploys post-stand-up. NUNCA
> en el primer stand-up — si algun modulo falla, debug es mucho mas
> caro.

#### Recovery comun durante Ola C

| Sintoma | Causa probable | Fix |
|---|---|---|
| `RDSCreate` cuelga 15+ min | Subnet group sin AZs distintas o no hay capacity en la AZ | Re-apply (idempotente); si persiste, reduce a 1 AZ en module.network |
| `aws_ecs_service.mlflow: timeout waiting for steady state` | La imagen MLflow no esta en ECR o el comando rompe en startup | Re-corre 4.4.3; revisa `aws logs tail /ecs/ml-training/mlflow --follow` |
| `aws_lambda_function: source_code_hash mismatch` | Editaste el .py pero no re-zip-eo | `terraform apply` lo detecta y re-zipea (idempotente) |
| `permission denied: iam:CreateRole` | Tu profile AWS no tiene IAM permissions | `aws sts get-caller-identity` y revisa que sea admin/role-with-IAM |
| State lock acquire timeout | Otro `terraform apply` corriendo / state lock huerfano | `terraform force-unlock <LOCK_ID>` (mostrado en el error) |

#### Verificacion Ola C

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
> - **C3 (CI/CD)**: con `enable_cicd=false` (default) este chequeo NO aplica — el `data "aws_iam_openid_connect_provider"` tiene count=0 y el plan no lo evalua. Solo con `enable_cicd=true`: si el OIDC provider de #2.5 no existe, el `data` falla → bootstrapear OIDC antes de re-aplicar `module.cicd`.
> - **Tiempo**: C1 ~2m + C2 ~10m (RDS domina) + C3 ~3m + C4 ~2m = ~17m total.

### 4.6 Smoke test — entrenar 1 variedad end-to-end

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

#### 4.6.1 Como funciona la task `batch:smoke`

`batch:smoke` es un atajo a `batch:train VARIETIES=POP TUNING=smoke`. El path:

0. **Preflight** `assert_jobdef_image` (en `tasks/lib/batch_wait.sh`): resuelve la imagen de la job-def `ml-training-trainer` ACTIVE y confirma que ese tag exista en ECR. Si falta, aborta en ~2s con el comando exacto a correr — antes de gastar Lambda invoke + ~3 min de Spot.
1. **Lambda invoke** al `ml-training-dispatcher` con payload `{varieties:"POP", tuning:"smoke"}`.
2. El dispatcher valida variety + S3 key, llama `aws batch submit-job` con `S3_DATA_KEY` como env override, y devuelve el `jobId`. (El hydrate real ocurre dentro del trainer en `main.py::_hydrate_data_from_s3`.)
3. **Polling** via `aws batch describe-jobs` cada 30s hasta que `status` sea `SUCCEEDED` (exit 0) o `FAILED` (exit 1). Funcion `wait_job` en `tasks/lib/batch_wait.sh`.

Una sola via de submit (el dispatcher) — el dev local con `task batch:train` y el CI con `task batch:train` ejecutan IDENTICO el camino.

#### 4.6.2 Ejecutar

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

#### 4.6.3 Verificacion post-smoke

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

### 4.7 Confirmar suscripcion SNS

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

### 4.8 Catalogo de tasks operativas

Las implementaciones viven en `tasks/*.yml` (definidas en sección 4.1.3-4.1.8). Esta seccion es referencia de uso.

#### 4.8.1 Re-entrenar (`task batch:train`)

Submit via Lambda dispatcher (valida variety + S3 key) y **fire-and-forget por defecto**: el job corre en AWS Batch, así que cerrar la terminal o apagar la máquina no lo detiene. Las varias variedades van en **un solo job** (el dispatcher las une en un `--varieties POP,JUPITER`). El hydrate de S3 corre dentro del trainer (`main.py::_hydrate_data_from_s3`), no en el dispatcher.

```bash
task batch:train VARIETIES=POP                # background; vuelve al prompt al instante
task batch:train VARIETIES=POP,JUPITER        # varias variedades en un job
task batch:train VARIETIES=all                # todas las permitidas
task batch:train VARIETIES=POP TUNING=prod_xl # ~5-6h en On-Demand (evita kills Spot)
task batch:train VARIETIES=POP WAIT=true      # bloquea + hace polling hasta SUCCEEDED/FAILED
```

Ver el estado de un entrenamiento (watch/logs/cancel se defaultean al último job submiteado, persistido en `.batch-last-job`):

```bash
task batch:watch                 # sigue el último job hasta SUCCEEDED/FAILED (nunca rompe la terminal)
task batch:logs                  # tail de los logs en CloudWatch (FOLLOW=true para seguirlos en vivo)
task batch:status                # todos los jobs activos en ambas queues
task batch:cancel                # termina el último job (o JOB_ID=<id>)
```

#### 4.8.1.1 EDA exploratorio on-demand (`task batch:eda`)

El EDA es una necesidad **aparte y opcional** del entrenamiento: lo corres cuando querés inspeccionar la calidad/drift de los datos de una variedad, tantas veces como haga falta. No forma parte del pipeline de `batch:train` — es standalone y repetible.

Mismo camino que `batch:train` (dispatcher → Batch → hydrate de S3), pero el dispatcher recibe `mode=eda` y arma `command=["--eda", "--varieties", ...]`. El trainer, con `--eda`, corre `src.diagnostics.eda.run_eda` por variedad en vez de entrenar y sincroniza los HTML a S3 — quedan visibles en `http://$ALB/reports/EDA_<variety>_<ts>.html`.

```bash
task batch:eda VARIETIES=POP              # EDA de una variedad (background)
task batch:eda VARIETIES=POP,JUPITER      # varias en un solo job
task batch:eda VARIETIES=POP WAIT=true    # bloquea hasta que termina
```

> **Equivalente local**: `task eda VARIETIES=POP` (corre el mismo `run_eda` en docker compose contra la data local). El de Batch es para correrlo contra la data ya hidratada en S3, sin levantar el stack local.

> **Nota**: el `mode` lo agrega el dispatcher (`infra/lambdas/dispatcher.py`); si cambiás esa Lambda, redeployá con `task infra:apply TARGET=module.lambdas` para que el código nuevo quede activo.

#### 4.8.2 Lifecycle del cluster

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

#### 4.8.3 Destroy total

```bash
task destroy   # vacia buckets + purga ECR + terraform destroy (backend tfstate queda)
task nuke      # ↑ + borra tfstate + OIDC  (IRREVERSIBLE: requiere re-bootstrap)
```

#### 4.8.4 Resumen

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

## Parte 5 — Patch del trainer (emitir MAPE a CloudWatch)

> **STATUS: APLICADO** (auditoria 2026-05-18). El patch esta integrado en `src/orchestration/variety_runner.py` (import + llamada `emit_mape_metric(variety=variety, mape_value=champion.oof_mape)` despues del bloque mape_ok/gap_ok). `trainer_image_tag` bumpeado a `v0.2.0` en `infra/envs/prod/terraform.tfvars`. La proxima vez que rebuiledes con `task ecr:build IMG=trainer` la imagen llevara el patch.

Sin este patch, las alarmas de `module.monitoring` (Parte 3 sección 3.8) que escuchan el namespace `ml-training/Training` con dimension `variety` nunca disparan — el namespace esta vacio y un MAPE alto pasa silencioso.

### Mapa del camino — Parte 5

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

### 5.1 Donde se inserta

El patch crea `src/utils/cloudwatch_metrics.py::emit_mape_metric()` y lo invoca al final de `train_variety()` en `src/orchestration/variety_runner.py`, justo despues del quality gate (donde el log dice "CAMPEON pasa quality gate" o "RECHAZADO por calidad operativa") y antes de generar el Excel/Dashboard.

**Por que aca y no en `runners.py`**: el `champion` existe como objeto `ModelResult` (dataclass del paso 05) solo dentro de `train_variety`. En `runners.py` solo se ve el dict agregado. **Por que `champion.oof_mape` y no `champion.full_mape`**: la alarma mide degradacion en datos no vistos (OOF) — `full_mape` es in-sample (optimista) y mete ruido cuando el modelo memoriza el train.

### 5.2 Crear `src/utils/cloudwatch_metrics.py`

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

### 5.3 Invocar desde el runner

En `src/orchestration/variety_runner.py`:

```python
# 1) Import al inicio del modulo
from src.utils.cloudwatch_metrics import emit_mape_metric

# 2) Dentro de train_variety, despues del bloque `if not mape_ok / elif not gap_ok / else`
#    (linea ~104), antes de `losers = [...]`:
emit_mape_metric(variety=variety, mape_value=champion.oof_mape)
```

**Importante**: la llamada se hace **siempre**, incluso si el quality gate rechaza el modelo (`mape_ok=False`). Eso es deliberado — un MAPE alto rechazado es justamente el caso que la alarma `ml-training-mape-<variety>` quiere capturar. Si solo se emitiera cuando el gate aprueba, las alarmas nunca dispararian en los runs malos.

### 5.4 Verificar local que no rompe

```bash
# AWS_BATCH_JOB_ID solo existe en Batch -> en docker compose local, emit hace skip silencioso
# --user/MPLCONFIGDIR: evitan el PermissionError en los bind-mounts (logs/reports) al
# correr como mluser; en Batch no se usan (no hay bind-mount). En local podes correr
# `task train VARIETIES=POP TUNING=smoke`, que ya los aplica.
docker compose run --rm --user "$(id -u):$(id -g)" -e MPLCONFIGDIR=/tmp trainer --varieties POP --tuning smoke
```

El log NO debe tener "CloudWatch MAPE=..." (es Batch-only). El training termina exitoso y NO se contamina el namespace prod de CloudWatch.

### 5.5 Re-build + push + verificar end-to-end

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

## Parte 6 — CI/CD con GitHub Actions

> **STATUS: NO APLICADO TODAVIA** (auditoria 2026-05-22). El directorio `.github/workflows/` **NO existe** en el repo. La infra IAM (`module.cicd` con roles `ml-training-gha-deploy` y `ml-training-gha-train`) SI esta lista — los workflows YAML que vienen abajo hay que crearlos a mano siguiendo 6.2/6.3/6.4. Hasta entonces, los disparadores (`on: push`, `workflow_dispatch`, etc.) no existen y el pipeline se opera 100% a mano via `task`.
>
> **Nota — qué existe y qué no**: los ADR **ya existen** en [`docs/adr/`](adr/) (ADR-001..009) y los links de esta Parte resuelven. Lo que sigue sin existir son los workflows `.github/workflows/{deploy,training,destroy}.yml`: los YAML de abajo son material **por crear**, no archivos que puedas leer del repo.

### Mapa del camino — Parte 6

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

> **Gotcha Parte 6**: `Could not assume role` → el `sub` no coincide con
> `main` o `environment:production`. Revisar org/repo, branch y nombre exacto
> del GitHub Environment.

### 6.0 Modelo de trust

En Parte 2 sección 2.5 creaste el OIDC provider; en Parte 3 sección 3.11 el modulo `cicd` creo 2 roles que confian en ese provider para tu `org/repo`:

- `ml-training-gha-deploy` — workflows que tocan infra (terraform apply, push ECR).
- `ml-training-gha-train` — workflows que solo invocan Lambda dispatcher.

ARNs en outputs:

```bash
terraform -chdir=infra/envs/prod output gha_deploy_role_arn
terraform -chdir=infra/envs/prod output gha_train_role_arn
```

### 6.1 Filosofia: Task = single source of truth

Toda la logica vive en `Taskfile.yml` + `tasks/*.yml`. Los workflows son thin wrappers: autentican via OIDC, setean `TF_VAR_*` y llaman `task X`.

```
Si `task X` corre OK en tu laptop → corre OK en CI.
Si rompe en CI → rompe en local con `task X`.
```

#### 6.1.1 Matriz de decision: donde vive cada operacion

| Operacion | Herramienta | Por que |
|---|---|---|
| Crear VPC, RDS, Batch queue, Lambda, alarmas | **Terraform** (`infra/modules/*`) | Declarativo, idempotente, state |
| Build + push imagen a ECR | **Task** (`task ecr:build-all`) | Imperativo, encadena docker + ECR login + tag dual |
| Submit Batch job + polling hasta SUCCEEDED | **Task** (`task batch:train`) | Encadena `lambda invoke` + `jq` + `describe-jobs` con loop |
| Apply Terraform en oleadas + smoke | **Task** (`task deploy`) | Orquesta oleadas A+B+C |
| Wake / sleep RDS+Fargate | **Task** (`task wake`/`sleep`) | `ops:up` idempotente con polling RDS+ALB |
| Disparar entreno bajo demanda | **GHA workflow_dispatch** | UI con dropdown, accesible sin AWS CLI |
| Approval antes de deploy o promote | **GHA `environment: production`** | Solo GHA tiene approval gates |
| Cron Mi+Ju 08-16 PET encender/apagar | **EventBridge + Lambda** (`scheduler.py`) | Serverless, sin runner |
| Quality gate + comparación vs `@champion` | **Task** (`task ops:promote`) llamado por GHA | Lógica reutilizable y auditable |
| Apagar todo con confirmacion textual | **GHA workflow** (`destroy.yml`) + **Task** | Confirmacion + approval son features GHA; logica en Task |

#### 6.1.2 Anti-patrones (cuando NO meter en Taskfile)

| Anti-patron | Donde va en su lugar |
|---|---|
| Task que solo corre en CI (ej. `gha:upload-artifact`) | Step inline en el workflow |
| Task wrapper trivial de 1 linea (ej. `docker:build` = `docker build .`) | Eliminar |
| Logica que deberia ser Terraform (ej. `aws ec2 create-vpc`) | Modulo Terraform |

Regla: si un task no (a) encadena >=2 cosas, (b) usa vars computadas, o (c) agrega idempotencia/polling — no justifica el namespace.

### 6.2 Variables y secrets de GitHub

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

### 6.3 `deploy.yml` — lint + build + plan + apply

Trigger: push a `main`, PR a `main`, `workflow_dispatch`. Cinco jobs:

| Job | Disparo | Que hace |
|---|---|---|
| `changes` | siempre | `dorny/paths-filter@v3` setea outputs `infra` (toco `infra/**`) y `trainer` (toco `src/**`, `main.py`, `Dockerfile`, `requirements.txt`). |
| `lint` + `test` | siempre | `task lint`, validación Terraform y `pytest -q`. |
| `build-and-push` | push main **AND** `trainer == 'true'` | Asume `gha-deploy`, `task ecr:build IMG=trainer`. |
| `terraform-plan` | PR **AND** `infra == 'true'` | Asume `gha-deploy`, `task infra:plan` comentado en el PR via `github-script`. |
| `infra-apply` | push main **AND** `infra == 'true'` **AND** lint OK | `environment: production` (approval). `task deploy` con `TF_VAR_trainer_image_tag=sha-<sha>`. |

> **3 decisiones no obvias del workflow**:
> - **`if: always()` en `infra-apply`**: el job depende de `build-and-push`, que puede haberse skipped (PR sin cambios al trainer). Sin `always()`, GHA marca `infra-apply` como skipped. Con `always()` + check explicito `(success || skipped)`, el apply corre aunque solo cambie infra.
> - **`cancel-in-progress` condicional**: en PRs queremos cancelar runs viejos (ahorra CI minutes); en push a main NO (un apply a medias deja state inconsistente).
> - **Build directo, no via `task ecr:build-all`**: solo el trainer se rebuildea por commit; mlflow + reports se actualizan via bump manual de TAG (cambios infrecuentes).

#### 6.3.1 Archivo completo `.github/workflows/deploy.yml`

```yaml
name: Deploy

# Estrategia thin: cada job autentica via OIDC y delega en `task X`.
# Jobs y disparos:
#   lint + test     -> SIEMPRE (push, PR, manual).
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
      release: ${{ steps.filter.outputs.release }}
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
            release:
              - 'src/**'
              - 'scripts/**'
              - 'main.py'
              - 'Dockerfile'
              - 'api/**'
              - 'ui/**'
              - 'docker/**'
              - 'requirements*.txt'
              - 'pyproject.toml'

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.13', cache: 'pip' }
      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: 1.10.5 }
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

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.13', cache: 'pip' }
      - name: Install test dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt -r requirements-dev.txt
      - name: Unit and contract tests
        run: pytest -q

  build-and-push:
    needs: [lint, test, changes]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main' && needs.changes.outputs.release == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_GHA_DEPLOY_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - name: Build once and publish the five release images
        run: |
          SHA12=$(git rev-parse --short=12 HEAD)
          for image in trainer mlflow reports api ui; do
            task ecr:build IMG="$image" TAG="sha-$SHA12"
          done
        env:
          AWS_DEFAULT_REGION: ${{ vars.AWS_REGION }}
          PROJECT: ${{ vars.PROJECT }}
      - run: echo "::notice title=Pushed::release sha-$(git rev-parse --short=12 HEAD)"

  terraform-plan:
    needs: [lint, test, changes]
    if: github.event_name == 'pull_request' && needs.changes.outputs.infra == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
        with: { terraform_version: 1.10.5 }
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - name: Static Terraform validation (no AWS credentials on PRs)
        id: plan
        run: task infra:validate
      - uses: actions/github-script@v7
        with:
          script: |
            const body = [
              "### Terraform validation",
              "",
              "Formato, inicialización sin backend y `terraform validate`: OK.",
              "",
              "El plan remoto se ejecuta únicamente con un rol `plan` de solo lectura;",
              "nunca con el rol privilegiado de apply desde una PR."
            ].join("\n");
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });

  infra-apply:
    needs: [lint, test, changes, build-and-push]
    if: |
      always() &&
      github.event_name == 'push' && github.ref == 'refs/heads/main' &&
      needs.lint.result == 'success' &&
      needs.test.result == 'success' &&
      (needs.build-and-push.result == 'success' || needs.build-and-push.result == 'skipped') &&
      (needs.changes.outputs.infra == 'true' || needs.changes.outputs.release == 'true')
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
        with: { terraform_version: 1.10.5 }
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
        run: |
          SHA12=$(git rev-parse --short=12 HEAD)
          if [ "${{ needs.changes.outputs.release }}" = "true" ]; then
            export TF_VAR_trainer_image_tag="sha-$SHA12"
            export TF_VAR_mlflow_image_tag="sha-$SHA12"
            export TF_VAR_reports_image_tag="sha-$SHA12"
            export TF_VAR_api_image_tag="sha-$SHA12"
            export TF_VAR_ui_image_tag="sha-$SHA12"
          fi
          task deploy
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

> **App stack integrado.** El workflow canónico ya incluye trainer, MLflow,
> reports, API y UI en un único release SHA. Si cambia cualquier parte del
> código ejecutable, construye las cinco imágenes, publica `sha-<12>` y pasa
> exactamente ese tag a Terraform. Así desaparece el mismatch anterior entre
> tags de 40 y 12 caracteres y no queda una API nueva sirviendo un pipeline
> viejo.

> **Configuracion previa en GitHub** (UNA sola vez): Settings → Environments → New environment → `production` → Required reviewers → agregate. Sin esto, `infra-apply` arranca sin approval.

> **Gotcha #6.3 (deploy)**: `AccessDenied for
> sts:AssumeRoleWithWebIdentity` → el job no corre desde `main` ni desde
> `environment:production`, o el environment tiene otro nombre.

### 6.4 `training.yml` — train + auto-train + promote

Trigger: `workflow_dispatch` (UI manual) **o** `workflow_run` cuando `Deploy` completa con success. Seis jobs:

| Job | Disparo | Que hace |
|---|---|---|
| `detect` | siempre | Decide modo. Si `workflow_dispatch`: usa `inputs.action`. Si `workflow_run`: `git diff` entre el SHA y el ultimo `Deploy success` anterior (via `gh run list`) — si toco `src/`, `main.py`, `Dockerfile`, `scripts/`, `requirements.txt`: `mode=train`; sino `skip`. |
| `wake-services` | `mode == 'train'` | `task ops:up` (idempotente). Outputs `mlflow_was_up=true/false`. |
| `train` | `mode == 'train'` | `task batch:train VARIETIES=... TUNING=... WAIT=true`. |
| `cool-down-and-stop` | `mode == 'train'` **AND** `mlflow_was_up == 'false'` | `task ops:down COOLDOWN=600`. **Solo apaga si nosotros lo levantamos**. |
| `validate-promotion` | `mode == 'promote'` | Ejecuta gates automáticos sin mover aliases. |
| `promote` | validación OK | Espera approval de `environment: production`, revalida y reasigna `@champion`. |

> **Decisiones no obvias**:
> - **`permissions: actions: read`**: el job `detect` corre `gh run list --workflow Deploy --status success` para encontrar el BASE_SHA correcto (cubre push de N commits y squash merges; `HEAD~1` fallaba).
> - **Cool-down condicional**: si un humano ya prendio MLflow para mirar dashboards y dispara un train manual, no queremos apagarlo al terminar.
> - **`train` NO hace `aws batch submit-job` directo**: seguridad. El rol `gha-train` solo tiene `lambda:InvokeFunction` sobre `dispatcher`. La validacion de payload (varieties whitelist, tuning whitelist) vive en Lambda.

#### 6.4.1 Archivo completo `.github/workflows/training.yml`

```yaml
name: Training

# Workflow consolidado. Tres modos via input `action`:
#   action=train    -> entrena variedades elegidas (con wake/cool-down si MLflow esta apagado)
#   action=promote  -> valida y reasigna @champion después del approval
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

  validate-promotion:
    needs: detect
    if: needs.detect.outputs.mode == 'promote'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: arduino/setup-task@v2
        with: { version: 3.x, repo-token: "${{ secrets.GITHUB_TOKEN }}" }
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_GHA_TRAIN_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION }}
      - name: Automated promotion gates
        run: >-
          python scripts/promote_model.py
          "${{ inputs.model_name }}"
          "${{ inputs.version }}"
          --max-mape "${{ inputs.max_mape }}"
          --validate-only
        env:
          MLFLOW_ALB_DNS: ${{ vars.MLFLOW_ALB_DNS }}

  promote:
    needs: [detect, validate-promotion]
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
      - name: Revalidate and move @champion
        run: >-
          task ops:promote
          MODEL_NAME="${{ inputs.model_name }}"
          VERSION="${{ inputs.version }}"
          MAX_MAPE="${{ inputs.max_mape }}"
        env:
          MLFLOW_ALB_DNS: ${{ vars.MLFLOW_ALB_DNS }}
```

Uso desde la UI: Actions → Training → Run workflow. Para entrenar: `action=train`, `varieties=POP`, `tuning=smoke`. Para promover: `action=promote`, `model_name=rnd-forest-POP`, `version=5`. Espera "Waiting for review" y aproba.

> **Gotcha #6.4 (training)**: sin `permissions: actions: read`, `gh run list` falla con 403 silencioso y cae al fallback `${HEAD_SHA}^` que solo cubre single-commit pushes. Test: `gh workflow run training.yml -f action=train -f varieties=POP -f tuning=smoke` → `detect` debe resolver `mode=train`.

### 6.5 `destroy.yml` — 3 modos (TEAR-DOWN / DESTROY / NUKE)

Trigger: **solo `workflow_dispatch`**.

| Modo | Que destruye | Que preserva | Reversible con |
|---|---|---|---|
| **TEAR-DOWN** | Modulos volatiles (mlflow, reports, batch, lambdas, monitoring, scheduler, cicd, consumer_iam) + libera el NAT (`enable_nat=false`) | S3 + ECR + network (VPC/subnets/SGs, sin NAT) + tfstate + OIDC | `task rebuild` (~25-40 min: restaura el RDS del backup; artifacts intactos en S3). Costo restante: ~$1/mes. |
| **DESTROY** | TODOS los modulos administrados (incluye storage). Vacia buckets versionados + purga ECR antes del `terraform destroy`. | tfstate bucket + OIDC provider | Re-crear via `task deploy`. Costo: $0/mes. |
| **NUKE** | DESTROY + tfstate bucket + OIDC provider. Cuenta limpia. | Nada | Re-bootstrap desde cero (Parte 2). |

#### 6.5.1 Doble salvaguarda

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

#### 6.5.2 Archivo completo `.github/workflows/destroy.yml`

```yaml
name: Destroy

# Tres modos via input `modo`:
#   TEAR-DOWN -> destroy volatiles + libera NAT (enable_nat=false). Preserva
#                S3 + ECR + network (sin NAT) + tfstate + OIDC.
#                Reversible con `task rebuild` (restaura el RDS del backup).
#                Costo restante: ~$1/mes (S3 + backups).
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
        with: { terraform_version: 1.10.5 }
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

### 6.6 Branch protection

```bash
gh api "repos/${GITHUB_OWNER}/ml_training/branches/main/protection" -X PUT --input - <<EOF
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["Deploy / lint", "Deploy / test"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1
  },
  "restrictions": null
}
EOF
```

Los contexts corresponden a los jobs `lint` y `test` del workflow consolidado.
`pytest` debe recolectar al menos una prueba; cero pruebas es fallo, no éxito.

---

## Parte 7 — Promotion gate (aliases de MLflow)

> [!IMPORTANT]
> Los Model Stages están deprecados desde MLflow 2.9. Este flujo usa tags de
> validación y el alias `@champion`. La única vía normal de promoción es
> `training.yml action=promote`; una reasignación manual queda reservada al
> rollback de emergencia y debe registrarse como incidente.

### Mapa del camino — Parte 7

```mermaid
flowchart TD
    TRAIN["Batch entrena y registra versión<br/>validation_status=pending"]
    AUTO{"Gates automáticos<br/>lineage + calidad + A/B"}
    REVIEW["Review de residuos, drift,<br/>explicabilidad y costo"]
    APPROVAL{"GitHub Environment<br/>approval"}
    RECHECK{"Revalidación<br/>anti-TOCTOU"}
    CHAMP["set_registered_model_alias<br/>alias=champion"]
    FAIL["Rechazo<br/>tag validation_status=rejected"]
    ROLLBACK["Rollback<br/>reasignar @champion<br/>a versión anterior"]

    TRAIN --> AUTO
    AUTO -->|pasa| REVIEW --> APPROVAL
    APPROVAL -->|aprobado| RECHECK
    RECHECK -->|pasa| CHAMP
    AUTO -->|falla| FAIL
    RECHECK -->|falla| FAIL
    CHAMP -.incidente.-> ROLLBACK
```

El workflow se divide en dos jobs:

1. `validate-promotion`, sin environment protegido, ejecuta los gates y publica
   un resumen sin secretos.
2. `promote`, con `environment: production`, depende del anterior. Después del
   approval vuelve a validar la misma versión y recién entonces reasigna
   `@champion`. La segunda validación evita promover algo que cambió mientras
   esperaba aprobación.

### 7.1 Contrato de una versión candidata

La versión debe tener:

- `validation_status=pending`;
- run terminado en `FINISHED`;
- `git_dirty=false`;
- commit completo y digest de imagen;
- `dataset_sha256`, key y VersionId de S3;
- signature e input example;
- métricas `mape_test`, `mape_oof`, `gap_oof_test` y una métrica robusta
  adicional (`mae` o `wape`);
- reporte asociado al mismo hash de dataset;
- ausencia de NaN/inf en métricas y outputs.

### 7.2 Gates

| Gate | Condición |
|---|---|
| Integridad | Lineage completo, run finalizado, artifact y signature legibles |
| Calidad absoluta | MAPE y gap bajo umbral; MAE/WAPE dentro de política |
| Comparación | No degrada frente a `models:/<name>@champion` más allá de la tolerancia |
| Estabilidad | Resultados por ventana temporal y cobertura de intervalos aceptables |
| Operación | Latencia, tamaño y memoria compatibles con la API |
| Humano | Revisión de residuos, drift, explicabilidad y contexto de negocio |

Cada gate escribe tags (`validation_status`, `validation_reason`,
`validated_at`, `validated_by_workflow`) en la versión. Los umbrales viven en
Git y se registran en el run; no se editan durante la promoción.

### 7.3 Carga de modelos en serving

La API carga siempre:

```text
models:/rnd-forest-<VARIETY>@champion
```

Debe registrar en logs y en `/api/health` el nombre, versión, run ID, digest de
imagen y dataset hash actualmente cargados. Al cambiar el alias, la API invalida
su cache de forma controlada o se fuerza un redeploy; un proceso que cachea el
modelo indefinidamente no recibe la promoción.

### 7.4 Rollback

Rollback no reentrena: reasigna el alias a una versión conocida.

```bash
export MODEL_NAME="rnd-forest-POP"
export ROLLBACK_VERSION="3"

python - <<'PY'
import os
from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    os.environ["MODEL_NAME"],
    "champion",
    os.environ["ROLLBACK_VERSION"],
)
print(
    f"OK {os.environ['MODEL_NAME']}@champion -> "
    f"v{os.environ['ROLLBACK_VERSION']}"
)
PY
```

Después del rollback:

1. reiniciar o invalidar la cache de la API;
2. verificar `/api/health`;
3. ejecutar un prediction smoke conocido;
4. etiquetar la versión retirada con `rollback_reason`;
5. abrir incidente con la evidencia de la reasignación.

### 7.5 `scripts/promote_model.py`

El archivo faltaba en la versión anterior aunque `tasks/ops.yml` lo invocaba.
Esta implementación mínima usa aliases y puede separar validación de mutación:

```python
from __future__ import annotations

import argparse
import math
import os
import subprocess
from datetime import datetime, timezone

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def tracking_uri(tf_dir: str) -> str:
    explicit = os.getenv("MLFLOW_TRACKING_URI")
    if explicit:
        return explicit.rstrip("/")
    result = subprocess.run(
        ["terraform", f"-chdir={tf_dir}", "output", "-raw", "tracking_uri"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().rstrip("/")


def metric(run, name: str) -> float:
    value = run.data.metrics.get(name)
    if value is None:
        raise SystemExit(f"FAIL falta métrica obligatoria: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise SystemExit(f"FAIL métrica no finita: {name}={result}")
    return result


def required_tag(tags: dict[str, str], name: str) -> str:
    value = tags.get(name)
    if value in (None, "", "unknown", "missing"):
        raise SystemExit(f"FAIL falta tag válido: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("version")
    parser.add_argument("--max-mape", type=float, default=20.0)
    parser.add_argument("--max-gap", type=float, default=5.0)
    parser.add_argument("--max-regression-pct", type=float, default=0.0)
    parser.add_argument("--tf-dir", default="infra/envs/prod")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(tracking_uri(args.tf_dir))
    client = MlflowClient()
    candidate = client.get_model_version(args.model_name, args.version)
    run = client.get_run(candidate.run_id)

    if run.info.status != "FINISHED":
        raise SystemExit(f"FAIL run status={run.info.status}")

    tags = {**run.data.tags, **candidate.tags}
    required_tag(tags, "git_commit")
    required_tag(tags, "image_digest")
    required_tag(tags, "dataset_sha256")
    required_tag(tags, "dataset_s3_key")
    required_tag(tags, "dataset_s3_version_id")
    if tags.get("git_dirty", "").lower() != "false":
        raise SystemExit("FAIL git_dirty debe ser false")

    candidate_mape = metric(run, "mape_test")
    candidate_gap = abs(metric(run, "gap_oof_test"))
    if candidate_mape > args.max_mape:
        raise SystemExit(
            f"FAIL mape_test={candidate_mape:.4f} > {args.max_mape:.4f}"
        )
    if candidate_gap > args.max_gap:
        raise SystemExit(
            f"FAIL gap_oof_test={candidate_gap:.4f} > {args.max_gap:.4f}"
        )

    try:
        champion = client.get_model_version_by_alias(args.model_name, "champion")
    except MlflowException:
        champion = None

    if champion and str(champion.version) != str(candidate.version):
        champion_run = client.get_run(champion.run_id)
        champion_mape = metric(champion_run, "mape_test")
        allowed = champion_mape * (1 + args.max_regression_pct / 100)
        if candidate_mape > allowed:
            raise SystemExit(
                f"FAIL candidate={candidate_mape:.4f} > "
                f"champion_allowed={allowed:.4f}"
            )

    now = datetime.now(timezone.utc).isoformat()
    client.set_model_version_tag(
        args.model_name, args.version, "validation_status", "passed"
    )
    client.set_model_version_tag(
        args.model_name, args.version, "validated_at", now
    )

    if args.validate_only:
        print(f"OK validado {args.model_name} v{args.version}")
        return

    client.set_registered_model_alias(
        args.model_name, "champion", args.version
    )
    client.set_model_version_tag(
        args.model_name, args.version, "deployment_status", "champion"
    )
    print(f"OK {args.model_name}@champion -> v{args.version}")


if __name__ == "__main__":
    main()
```

> Este gate compara métricas ya registradas. Antes de usarlo, la suite debe
> probar los casos “primer champion”, versión inexistente, métrica NaN, tags
> ausentes, empate, regresión permitida y carrera entre validate/promote.

---

> **Cierre Partes 5-7.** Flujo MLOps funcional; la clasificación
> production-grade depende además de completar el gate de la Parte 10: trainer
> parchado
> emitiendo MAPE por variedad a CloudWatch, 3 workflows GHA consolidados
> (`deploy.yml` / `training.yml` / `destroy.yml` con OIDC + confirmación
> textual), promotion ciclo A/B documentado. **Próximos pasos**: #8 (runbook)
> → #9 (costos/modos) → #11 (troubleshooting) →
> #12 (apéndices).

---

## Parte 8 — Runbook operativo extendido

> **Por qué esta parte**: si el sistema está en producción y vos no estás,
> alguien tiene que operarlo sin leerse las 5 oleadas. Este es ese manual:
> comandos copy-paste con el "por qué" al lado, para entender qué se hace y no
> ejecutar a ciegas.

### Mapa del camino — Parte 8

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

### 8.1 Manual diario / mas frecuente

#### 8.1.1 Re-entrenar una variedad

**Por que se hace**: data nueva subida al bucket (`aws s3 cp` del Excel
acumulado), o pediste un re-train porque cambiaste hiperparametros.

```bash
# Opcion A — via Task (preferido para humanos, polling + exit-code visible)
task batch:train VARIETIES=POP TUNING=prod WAIT=true

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

#### 8.1.2 Re-entrenar TODAS las variedades en un dia (recovery)

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

#### 8.1.3 Spot vs On-Demand por preset (cuando elegir cual)

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

#### 8.1.4 Rollback de imagen del trainer

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

### 8.2 Manual semanal / mensual

#### 8.2.0 Ciclo de trabajo recurrente (2 días por semana)

El patrón de uso real de este proyecto: **prender dos días, trabajar, apagar**.
No hay nada que recordar de un ciclo al otro — el estado viaja solo, vía el
backup del RDS (metadata) y S3 (artifacts). Ver #8.5 para el mecanismo.

```bash
# ── DÍA DE TRABAJO ──────────────────────────────────────────────────────────
task deploy                  # levanta todo y RESTAURA el último backup si existe
                             #   (cuenta virgen o SNAPSHOT=none -> RDS vacío)
task status                  # confirma RDS available + servicios healthy + URLs

task batch:train VARIETIES=A,B,C,D PARALLEL=4     # entrenar en grupos de 4
task batch:watch                                   # bloquea hasta SUCCEEDED/FAILED
# ... UI :8501 para pronósticos, reports para los HTML del campeón ...

# ── AL TERMINAR ─────────────────────────────────────────────────────────────
task teardown                # backup verificado -> destroy -> poda. ~$1/mes
```

**Qué garantiza el ciclo**, y qué no:

| Al volver con `task deploy` | ¿Vuelve? |
|---|---|
| Runs de MLflow, métricas, params, tags | ✅ (backup del RDS) |
| Model Registry: versiones, stages, transitions | ✅ (backup del RDS) |
| Pronósticos persistidos por la API (`forecasts`) | ✅ (backup del RDS) |
| Modelos `.joblib` y reports HTML | ✅ (nunca se fueron: viven en S3) |
| Excels de entrada | ✅ (nunca se fueron: viven en S3) |
| DNS del ALB | ❌ **cambia en cada ciclo** — no lo pongas en bookmarks |

> [!TIP]
> **El único costo recurrente de este ciclo son ~30-40 min de espera** (destroy
> + backup al apagar; restore + cold start al prender). Si la pausa es de una o
> dos noches y no de días, `task sleep` / `task wake` es mucho más rápido (5 min)
> porque *para* el RDS en vez de destruirlo. La regla: **>1 semana → teardown;
> 1-2 noches → sleep**. Ojo: pasados 7 días AWS re-arranca solo un RDS parado, y
> ahí `sleep` deja de ahorrar (#8.5, "Período de gracia de RDS").
>
> **En el ciclo miércoles+jueves** la pausa es de ~6 días (jueves noche → miércoles
> mañana): roza el límite de los 7 días, así que va **`teardown` el jueves y
> `rebuild` el miércoles**, no `sleep`. Con `sleep` te quedarías a un día del
> auto-arranque de RDS y encima la lambda `keepstop` —que sería quien lo re-para—
> se destruye junto con el ALB, así que nadie lo apagaría.

> [!WARNING]
> **En este ciclo nunca uses `task destroy`.** `teardown` y `destroy` suenan
> parecido y hacen cosas muy distintas: `destroy` **vacía los buckets de S3**, y
> ahí viven los modelos y los reports. El backup del RDS te devolvería un Model
> Registry apuntando a artifacts que ya no existen. `destroy` es para cerrar el
> proyecto (#8.7), no para ahorrar.

#### 8.2.1 Bajar todo para ahorrar (tear-down)

**Por que se hace**: fin de mes, vacaciones, pausa del proyecto.
Conocido como "scale to near-zero".

```bash
task teardown
# Confirmar con "y" al prompt de Task
```

Antes de destruir nada toma un **backup del RDS y espera a que quede
`available`** — si eso falla, el teardown aborta con la infra intacta. Al
terminar verifica que el backup quedó y poda los viejos. El mecanismo completo
está en #8.5.

**Por que el orden importa** (ver Parte 4.8.3):
1. **scheduler primero**: si esta arriba, va a re-encender RDS+Fargate
   en el proximo cron y anular el tear-down.
2. **reports + mlflow despues**: son Fargate consumidores; bajan
   primero para que el ALB no tenga targets unhealthy.
3. **batch**: drena Spot CE a 0.
4. **network**: NAT GW es el ultimo en irse — `$32/mes` ahorro
   inmediato.

#### 8.2.2 Volver a levantar (rebuild)

```bash
task rebuild
```

Restaura el RDS desde el último backup. `task deploy` hace exactamente lo mismo
(comparten el resolver, #8.5): usá `rebuild` cuando volvés de un teardown y
`deploy` cuando además querés re-buildear imágenes.

**Por que cambia el ALB DNS**: el `aws_lb.main` se recrea con un nombre
distinto en su DNS. Si tenias bookmarks, actualizalos. Si quisieras
un DNS estable, TLS + Route 53 lo resuelve (hardening, futuro).

#### 8.2.3 Subir data nueva

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

### 8.3 Manual de incidentes

#### 8.3.1 Job se quedo en RUNNABLE eternamente

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

#### 8.3.2 MLflow 403 "Invalid Host header"

**Por que pasa**: MLflow 3.x rechaza requests cuyo `Host:` header no
esta en `--allowed-hosts`. La configuración enumera el DNS del ALB y Cloud Map;
un dominio custom también debe agregarse explícitamente.

**Fix**: editar `modules/mlflow/ecs.tf:container_definitions.command`
para incluir el host y ejecutar `task infra:plan` + `task infra:apply`.

#### 8.3.3 RDS "too many connections"

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

#### 8.3.4 Spot interrupt mid-training

**Sintoma**: job en FAILED con `statusReason = "Host EC2 ..."`.
**Por que pasa**: AWS necesito tu c6i.2xlarge para otro customer.
**Que pasa automaticamente**: `retry_strategy.attempts = 2` + filtro
`Host EC2*` (Parte 3.7.3). El job se re-encola en otra instancia.
**Que hace falta a mano**: nada si pasa una vez. Si pasa
sistematicamente (3+ jobs FAILED por Spot en un dia), considerar:

- Cambiar a `tuning=prod_xl` para esa variedad (queue OD).
- Ver `Best practices > Capacity` en consola Batch — quizas la AZ tiene
  presion. Cambiar `instance_type` a alternativa (`c6a.2xlarge`, `m6i.2xlarge`).

#### 8.3.5 `task infra:apply` falla con state lock

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
LOCK_ID="reemplazar-con-el-id-real"
task infra:force-unlock LOCK_ID="$LOCK_ID"
```

#### 8.3.6 S3 sync del trainer falla con 403

**Por que pasa**: el job-role no tiene `s3:PutObject` sobre
`artifacts_bucket`, o el bucket esta en otra region.

**Que mirar**:

```bash
# Inline policy del job role
aws iam list-role-policies --role-name ml-training-job-role
POLICY_NAME="reemplazar-con-el-nombre-real"
aws iam get-role-policy --role-name ml-training-job-role --policy-name "$POLICY_NAME"
```

**Fix**: en Parte 3.7.2 el inline policy `job_s3` ya cubre PutObject
sobre el bucket. Si falla, verificar que el bucket creado matchea
`var.artifacts_bucket`.

#### 8.3.7 Job arranca pero MLflow esta apagado (fuera de ventana)

**Por que pasa**: lanzaste un train fuera de Mi+Ju 08-16 PET. El cron
de stop apago MLflow (Fargate desired_count=0). El trainer intenta
conectar a `tracking_uri` y obtiene timeout.

> Si el stack ya paso por el `teardown` del jueves, MLflow no esta apagado sino
> **destruido**: `task wake` no alcanza, va `task rebuild` (recrea volatiles +
> restaura el RDS del backup). Y si levantas fuera de ventana, ojo con `keepstop`
> (#3.10.2.b): cada 6h re-para el RDS si lo encuentra encendido fuera de horario.

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

#### 8.3.8 Cold-start de RDS lento el primer request

**Por que pasa**: RDS post-`start_db_instance` tarda ~5 min en estar
disponible. El primer query desde MLflow puede demorar 10-20s extra
por warm-up de buffers.

**Que NO hacer**: no agregues timeout corto en el container — vas a
matar conexiones legitimas. **Que hacer**: el healthcheck del task
def tiene `startPeriod = 60`; aumentalo si tu RDS warmup es
consistentemente mas lento.

### 8.4 Shutdown limpio DENTRO del job de training

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

### 8.5 TEAR-DOWN — apagar todo preservando state + datos

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
- RDS instance: **DESTRUIDA** (es parte de `module.mlflow`, uno de los
  VOLATILE_MODULES). Antes de destruir nada se toma un **backup verificado**
  (`<project>-mlflow-backup-YYYYMMDDHHMMSS`), y `task rebuild`/`task deploy`
  **restauran desde ese backup** para recuperar Model Registry + `forecasts`.
  Ver "Ciclo backup → restauración" mas abajo.
  (Quien solo *para* el RDS es `task sleep`, no el teardown.)
- Batch compute environments: `desired_vcpus = 0` (no hay EC2 corriendo)
- ALB + listener: borrados (se recrean en rebuild — el DNS cambia)
- NAT Gateway + EIP: **liberados** vía `enable_nat=false` (~$33/mes ahorro), sin destruir el resto de la red
- Subnets/VPC/SGs: se preservan (el toggle `enable_nat` baja solo el NAT, no `module.network` entero)

**Costo despues del tear-down**: ~$1/mes (S3 + ECR + los backups retenidos; ver matriz #9.3). El tear-down preserva `module.network` (VPC/subnets/SGs) pero **libera el NAT** vía `enable_nat=false` (`terraform apply -target=module.network -var enable_nat=false`), eliminando el ~$33/mes idle. `task rebuild`/`deploy` lo recrean. Al destruirse el RDS tambien desaparece el storage gp3 (~$2.30/mes) y en su lugar se paga solo el backup, facturado sobre datos **usados** (no sobre los 20 GB asignados): centavos para esta base.

#### Comando `task teardown`

```bash
task teardown
# Pide confirmacion: Task pide "y" para proceder.
# Pasos internos (tasks/ops.yml :: teardown):
#  0. ops:down -> aborta si hay Batch jobs RUNNING; invoca la Lambda
#     ml-training-scheduler (action=stop) para apagar RDS + Fargate.
#  1. [1/4] ensure_backup: BACKUP ANTES DE TOCAR NADA, y se espera a que quede
#     `available`. Incluye ensure_rds_available, porque el paso 0 PARO el RDS y
#     AWS rechaza snapshotear una instancia detenida. Si esto falla, el teardown
#     ABORTA con la infra intacta: mejor no apagar que perder datos.
#  2. [2/4] lift_rds_protection + loop `terraform destroy -target=$mod` sobre los
#     modulos VOLATILES, en orden reverso de apply:
#     scheduler -> lambdas -> monitoring -> batch -> reports -> api -> ui -> mlflow -> cicd -> consumer_iam
#     Pasa rds_skip_final_snapshot=true: el backup del paso 1 ya esta verificado.
#  3. [3/4] `terraform apply -target=module.network -var enable_nat=false` -> LIBERA el NAT (~$33/mes).
#  4. [4/4] assert_backup_exists (falla ruidosamente si no quedo backup) +
#     prune_snapshots: conserva los SNAPSHOT_KEEP ultimos (default 6).
#  storage es PERMANENTE: NO se destruye. network se preserva pero con el NAT liberado.
```

#### Ciclo backup → restauración

Esta es la pieza que hace que apagar y volver a levantar sea **"como si nada
hubiera pasado"**. Es la sección de referencia del ciclo: #8.6 (rebuild) y #8.7
(destroy) apuntan acá en vez de repetirla.

##### Vocabulario único

La guía, los taskfiles y los mensajes en pantalla usan **estas tres palabras y
no sinónimos**. Antes convivían "snapshot final", "snapshot manual", "backup" y
"restore" para nombrar lo mismo, y eso hacía que dos secciones parecieran
describir mecanismos distintos:

| Palabra | Qué es exactamente | Dónde vive |
|---|---|---|
| **backup** | un snapshot **manual** del RDS | AWS RDS (no es un objeto de S3) |
| **restaurar** | crear el RDS a partir de un backup (`rds_snapshot_identifier`) | `terraform apply` |
| **artifacts** | modelos `.joblib` + reports HTML + Excels de entrada | **S3**, fuera de este ciclo |

> [!IMPORTANT]
> **Los backups de Postgres NO son objetos de S3.** Son snapshots de RDS,
> almacenamiento gestionado por AWS, y se listan con `task backups` (no con
> `aws s3 ls`). Lo que sí vive en S3 son los **artifacts**. Son dos mecanismos
> de persistencia distintos y se pierden por causas distintas — ver la tabla
> "Qué sobrevive a qué" abajo.

##### Qué sobrevive a qué

| Dato | Dónde vive | `sleep` | `teardown` | `destroy` | `nuke` |
|---|---|---|---|---|---|
| MLflow tracking + Model Registry (metadata) | RDS | ✅ vivo | ✅ **backup** | ✅ **backup** (¹) | ✅ backup, pero (²) |
| Tabla `forecasts` de la API | RDS | ✅ vivo | ✅ **backup** | ✅ **backup** (¹) | ✅ backup, pero (²) |
| Modelos `.joblib` + reports HTML | S3 `artifacts` | ✅ | ✅ intacto | ❌ **se vacía** | ❌ se vacía |
| Excels de entrada | S3 `data` | ✅ | ✅ intacto | ❌ **se vacía** | ❌ se vacía |
| Imágenes de contenedor | ECR | ✅ | ✅ intacto | ❌ purgado | ❌ purgado |
| Terraform state | S3 `tfstate` | ✅ | ✅ intacto | ✅ intacto | ❌ borrado |

(¹) El backup del RDS sobrevive al `destroy` — los snapshots no son un recurso
gestionado por Terraform. Pero **el `destroy` sí vacía los buckets de S3**, así
que los artifacts no vuelven. Restaurar el RDS te devuelve el Registry apuntando
a artifacts que ya no existen.
(²) `nuke` corre `purge_secret` de la password master: el backup queda en la
cuenta sin el secret original; puede restaurarse y luego rotarse el password
master, pero ya no es un rebuild automático (ver #8.7).

> **Conclusión operativa**: para el ciclo recurrente usá **`teardown` + `rebuild`**,
> nunca `destroy`. Es el único par que devuelve el sistema completo — metadata y
> artifacts. `destroy` es para cerrar el proyecto, no para ahorrar.

##### El flujo, de punta a punta

```
        ┌──────────────── APAGAR (task teardown / destroy) ────────────────┐
        │  1. ensure_backup    ← backup ANTES de destruir nada, y se       │
        │                        ESPERA a que quede `available`           │
        │  2. terraform destroy  (rds_skip_final_snapshot=true: el backup  │
        │                         ya está tomado y verificado)             │
        │  3. assert_backup_exists  ← falla ruidosamente si no quedó nada  │
        │  4. prune_snapshots       ← retención (SNAPSHOT_KEEP, default 6) │
        └──────────────────────────────────────────────────────────────────┘
                                      │
                          backup manual timestamped
                                      │
        ┌───────────── LEVANTAR (task deploy / rebuild — mismo camino) ────┐
        │  5. resolve_restore_snapshot                                     │
        │       ¿el RDS ya existe?  → no restaura (apply normal)           │
        │       SNAPSHOT=none       → RDS vacío                            │
        │       SNAPSHOT=<id>       → ese backup (se valida `available`)   │
        │       por defecto         → el backup más reciente               │
        │       no hay ninguno      → RDS vacío, y NO es un error          │
        │  6. terraform apply -var rds_snapshot_identifier=<id>            │
        └──────────────────────────────────────────────────────────────────┘
```

**Dos propiedades de este diseño, y por qué importan:**

1. **El backup se toma ANTES del destroy, no durante.** El `final_snapshot` de
   `aws_db_instance` se materializa *en medio* del destroy. Si falla ahí —el caso
   real es el RDS en `stopped` → `InvalidDBInstanceState`— el destroy aborta a la
   mitad y quedás con la infra rota **y sin backup**. Tomándolo antes y
   verificándolo, el peor caso pasa a ser "el teardown falló pero tus datos están
   a salvo". Por eso el destroy corre con `rds_skip_final_snapshot=true`: el
   backup ya existe, duplicarlo sólo agrega 8 min y storage.

2. **`deploy` y `rebuild` resuelven el backup con la MISMA función.** Antes sólo
   `rebuild` restauraba; un `destroy` seguido de `task deploy` levantaba un RDS
   **vacío** aunque el backup estuviera ahí, en silencio. Ahora los dos caminos
   llaman a `resolve_restore_snapshot()` y dan el mismo resultado.

> [!NOTE]
> **La primera vez NO hay backups, y es lo esperado.** En una cuenta virgen el
> `task deploy` inicial crea el RDS **vacío**: nunca hubo un teardown, así que no
> existe ningún backup. `task backups` sale vacío y eso no es un error.
>
> | Momento | Backups en la cuenta | Qué hace `deploy`/`rebuild` |
> |---|---|---|
> | 1er `task deploy` (cuenta virgen) | 0 | RDS **vacío** (no hay nada que restaurar) |
> | 1er `task teardown` | 1 | — |
> | `deploy`/`rebuild` siguiente | 1 | **restaura** desde ese backup |
> | ciclos sucesivos | hasta 6 (poda) | restaura el más reciente |
>
> `latest_snapshot()` maneja el caso vacío devolviendo string vacío (no falla):
> el apply corre sin `-var` y crea la instancia limpia. Por eso las mismas tareas
> sirven para el estreno y para los ciclos posteriores — no hay un "modo primera
> vez" que recordar.

##### Tres detalles de implementación que **no son opcionales**

1. **La credencial master vive en la raíz**, `infra/envs/prod/rds_secret.tf`, no
   dentro de `module.mlflow`. Si viviera dentro, el teardown la destruiría junto
   con el módulo y el rebuild generaría una password NUEVA — que no coincide con
   la del backup restaurado. MLflow y la API no autenticarían hasta rotar el
   password o recuperar el secret. Mantenerlo en la raíz evita esa interrupción
   durante el ciclo normal; no convierte la contraseña en parte del backup. Ver
   [ADR-009](adr/ADR-009-rds-secret-fuera-del-modulo.md).
2. **El RDS debe estar `available` para poder respaldarlo.** `ops:down` (que el
   teardown invoca primero) lo **para**, y AWS rechaza snapshotear una instancia
   detenida. `ensure_rds_available` (`tasks/lib/nuke.sh`) la re-arranca y espera.
   Con el backup movido al *pre*-destroy esto dejó de ser un riesgo de pérdida de
   datos y pasó a ser sólo ~5-10 min de espera.
3. **`snapshot_identifier` lleva `lifecycle { ignore_changes }`** en
   `modules/mlflow/rds.tf`. El argumento es *ForceNew*: sin eso, cualquier apply
   posterior que no repita el mismo `-var` (por ejemplo el apply completo que
   hace `ops:up` cuando el ALB no existe) vería `""` contra el valor del state y
   **destruiría y recrearía el RDS**, perdiendo lo restaurado.

##### Tareas de apoyo

```bash
task backups              # listar los backups restaurables (alias: task snapshots)
task ops:backup-now       # backup manual sin destruir nada (RDS debe estar available)
task ops:verify-backup    # ¿hay al menos un backup restaurable? exit 1 si no
```

##### Implementación

> Los **taskfiles sí viven en el repo** (`Taskfile.yml`, `tasks/`) y ya
> implementan lo de arriba — los bloques que siguen son la referencia de qué
> hace cada uno, no algo que haya que pegar. Lo que sí es **pegable** es el
> Terraform de esta guía: `infra/` se removió del repo y se copia desde acá.

**(a) `tasks/lib/snapshot.sh`** — el motor del ciclo. `backup_now`,
`ensure_backup`, `assert_backup_exists` y `resolve_restore_snapshot` se sumaron
a las tres funciones que ya existían:

```bash
#!/usr/bin/env bash
# =============================================================================
# tasks/lib/snapshot.sh  -  Backups del RDS: crear, verificar, restaurar, podar
# =============================================================================
# Sourceado (no ejecutado) desde tasks/ops.yml, tasks/infra.yml y Taskfile.yml.
# Asume el CWD en la raiz del repo (Task siempre corre desde ahi).
#
# VOCABULARIO UNICO (mismas palabras en tasks, docs y mensajes en pantalla):
#   backup    = un snapshot MANUAL del RDS. Unica copia de MLflow tracking +
#               Model Registry + la tabla `forecasts`.
#   restaurar = crear el RDS a partir de un backup (rds_snapshot_identifier).
#   artifacts = modelos .joblib + reports HTML. Viven en S3, NO en el RDS, y no
#               participan de este ciclo: sobreviven al teardown solos.
#
# CICLO — ver docs/02-produccion-aws.md #8.5 "Ciclo backup -> restauracion".
#   apagar:   ensure_backup -> destroy (skip_final_snapshot=true)
#             -> assert_backup_exists -> prune_snapshots
#   levantar: resolve_restore_snapshot -> apply [-var rds_snapshot_identifier]
#
# POR QUE EL BACKUP VA ANTES DEL DESTROY (y no como final_snapshot de Terraform):
#   El `final_snapshot` de aws_db_instance se toma DURANTE el destroy. Si algo
#   falla ahi —el caso real es el RDS en `stopped` -> InvalidDBInstanceState— el
#   destroy aborta a la mitad y quedas con la infra rota Y sin backup. Tomandolo
#   antes, verificado y `available`, el destroy ya no puede perder datos.
#
# Solo se miran snapshots `manual`: los `automated` (backup_retention_period = 7)
# se borran junto con la instancia y no sirven como fuente de
# `aws_db_instance.snapshot_identifier`.
#
# Uso:  source tasks/lib/snapshot.sh
# =============================================================================

# ─── Consulta ────────────────────────────────────────────────────────────────

# latest_snapshot <db-instance-id>
#   Imprime en stdout el identifier del backup mas reciente (por
#   SnapshotCreateTime) que este `available`. Si no hay ninguno, imprime vacio
#   y retorna 0 (NO es un error: es el caso de un stand-up desde cero).
#   Todo el ruido va a stderr para no contaminar la sustitucion de comandos.
latest_snapshot() {
  local db_id="$1"
  local snap
  snap=$(aws rds describe-db-snapshots \
    --snapshot-type manual \
    --query "sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}' && Status=='available'], &SnapshotCreateTime)[-1].DBSnapshotIdentifier" \
    --output text 2>/dev/null || echo "")

  # La CLI devuelve el string "None" cuando el query no matchea nada.
  if [ -z "$snap" ] || [ "$snap" = "None" ]; then
    echo "  No hay backups de $db_id -> el RDS se creara VACIO." >&2
    echo ""
    return 0
  fi
  echo "  Backup mas reciente de $db_id: $snap" >&2
  echo "$snap"
}

# list_snapshots <db-instance-id>
#   Tabla legible de los backups (mas nuevo primero).
list_snapshots() {
  local db_id="$1"
  aws rds describe-db-snapshots \
    --snapshot-type manual \
    --query "reverse(sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}'], &SnapshotCreateTime))[].[DBSnapshotIdentifier,Status,SnapshotCreateTime,AllocatedStorage]" \
    --output table
}

# ─── Creacion ────────────────────────────────────────────────────────────────

# backup_now <db-instance-id> [etiqueta]
#   Crea un backup y ESPERA a que quede `available`. Imprime el identifier en
#   stdout; todo lo demas va a stderr para poder capturarlo con $(...).
#   Retorna 1 si el RDS no existe: quien llama decide si eso es fatal.
backup_now() {
  local db_id="$1"
  local label="${2:-backup}"
  local snap

  if ! aws rds describe-db-instances --db-instance-identifier "$db_id" >/dev/null 2>&1; then
    echo "  RDS $db_id no existe -> no hay nada que respaldar." >&2
    return 1
  fi

  # AWS rechaza snapshotear una instancia detenida (InvalidDBInstanceState) y
  # `ops:down` la deja parada. ensure_rds_available vive en nuke.sh; se sourcea
  # aca si el caller no lo hizo, para que backup_now sea autosuficiente.
  if ! command -v ensure_rds_available >/dev/null 2>&1; then
    # shellcheck source=tasks/lib/nuke.sh
    source tasks/lib/nuke.sh
  fi
  ensure_rds_available "$db_id" >&2

  snap="${db_id}-${label}-$(date +%Y%m%d%H%M%S)"
  echo "  Creando backup $snap..." >&2
  aws rds create-db-snapshot \
    --db-instance-identifier "$db_id" \
    --db-snapshot-identifier "$snap" >/dev/null

  echo "  Esperando a que $snap quede available (~3-8 min)..." >&2
  aws rds wait db-snapshot-available --db-snapshot-identifier "$snap"
  echo "  OK backup $snap verificado y restaurable." >&2
  echo "$snap"
}

# ensure_backup <db-instance-id> [etiqueta] [max-edad-minutos]
#   "Si no hay backup, lo hace; si ya hay uno fresco, trabaja con ese."
#   Garantiza que exista un backup `available` ANTES de una operacion
#   destructiva. Imprime en stdout el identifier del backup vigente.
#
#   max-edad-minutos (default 0 = siempre crea uno nuevo). Con un valor > 0
#   reutiliza el ultimo backup si es mas nuevo que esa edad: sirve para
#   reintentar un teardown que fallo DESPUES de haber respaldado, sin volver a
#   pagar los ~8 min de espera.
#
#   Retorna 1 si el RDS no existe (nada que respaldar): hacer teardown de una
#   infra ya destruida es legitimo, y el caller lo trata como no-fatal.
ensure_backup() {
  local db_id="$1"
  local label="${2:-backup}"
  local max_age_min="${3:-0}"
  local last created age_min

  if ! aws rds describe-db-instances --db-instance-identifier "$db_id" >/dev/null 2>&1; then
    echo "  RDS $db_id no existe -> no hay nada que respaldar (skip)." >&2
    return 1
  fi

  if [ "$max_age_min" -gt 0 ]; then
    last=$(aws rds describe-db-snapshots \
      --snapshot-type manual \
      --query "sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}' && Status=='available'], &SnapshotCreateTime)[-1].[DBSnapshotIdentifier,SnapshotCreateTime]" \
      --output text 2>/dev/null || echo "")
    if [ -n "$last" ] && [ "$last" != "None" ]; then
      created=$(echo "$last" | awk '{print $2}')
      age_min=$(( ( $(date -u +%s) - $(date -u -d "$created" +%s) ) / 60 ))
      if [ "$age_min" -le "$max_age_min" ]; then
        echo "  Ya hay un backup de hace ${age_min} min (<= ${max_age_min}) -> se reutiliza." >&2
        echo "$last" | awk '{print $1}'
        return 0
      fi
    fi
  fi

  backup_now "$db_id" "$label"
}

# ─── Verificacion ────────────────────────────────────────────────────────────

# assert_backup_exists <db-instance-id>
#   Falla ruidosamente si NO quedo ningun backup restaurable. Se corre DESPUES
#   del teardown/destroy: convierte una perdida silenciosa de datos (el bug
#   historico de este repo) en un error visible mientras todavia se puede
#   reaccionar.
assert_backup_exists() {
  local db_id="$1"
  local snap
  snap=$(aws rds describe-db-snapshots \
    --snapshot-type manual \
    --query "sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}' && Status=='available'], &SnapshotCreateTime)[-1].DBSnapshotIdentifier" \
    --output text 2>/dev/null || echo "")

  if [ -z "$snap" ] || [ "$snap" = "None" ]; then
    echo ""
    echo "ERROR No quedo NINGUN backup de $db_id."
    echo "      MLflow Registry y la tabla forecasts NO son recuperables."
    echo "      Si el RDS todavia existe: task ops:backup-now"
    return 1
  fi
  echo "  OK backup vigente: $snap (lo consumira el proximo deploy/rebuild)."
}

# ─── Restauracion ────────────────────────────────────────────────────────────

# resolve_restore_snapshot <db-instance-id> [preferencia]
#   Fuente UNICA de la decision "restaurar o arrancar limpio". La comparten
#   `task deploy` y `task rebuild` para que el resultado sea identico por
#   cualquiera de los dos caminos.
#
#   Imprime en stdout el identifier a restaurar, o vacio si corresponde crear un
#   RDS nuevo. Los mensajes van a stderr.
#
#   preferencia:
#     ""      (default) -> el backup mas reciente, si existe
#     "none"            -> forzar RDS vacio (ignora los backups)
#     "<id>"            -> ese backup exacto (se valida que exista y este available)
#
#   Precedencia deliberada: si el RDS YA existe, nunca se restaura. Restaurar es
#   una operacion de CREACION (snapshot_identifier es ForceNew); pasarlo sobre
#   una instancia viva la recrearia y destruiria los datos actuales.
resolve_restore_snapshot() {
  local db_id="$1"
  local pref="${2:-}"
  local st

  if aws rds describe-db-instances --db-instance-identifier "$db_id" >/dev/null 2>&1; then
    echo "  RDS $db_id ya existe -> no se restaura nada (apply normal)." >&2
    echo ""
    return 0
  fi

  if [ "$pref" = "none" ]; then
    echo "  SNAPSHOT=none -> RDS nuevo y VACIO (no se restaura)." >&2
    echo ""
    return 0
  fi

  if [ -n "$pref" ]; then
    st=$(aws rds describe-db-snapshots --db-snapshot-identifier "$pref" \
      --query 'DBSnapshots[0].Status' --output text 2>/dev/null || echo "")
    if [ "$st" != "available" ]; then
      echo "ERROR El backup '$pref' no existe o no esta available (estado: ${st:-inexistente})." >&2
      echo "      Ver los disponibles con: task backups" >&2
      return 1
    fi
    echo "  Backup fijado por el usuario: $pref" >&2
    echo "$pref"
    return 0
  fi

  latest_snapshot "$db_id"
}

# ─── Retencion ───────────────────────────────────────────────────────────────

# prune_snapshots <db-instance-id> <keep-n>
#   Borra los backups mas viejos, conservando los <keep-n> ultimos.
#   Idempotente. Con menos de keep-n backups no hace nada.
prune_snapshots() {
  local db_id="$1"
  local keep="${2:-6}"
  local all count victims

  all=$(aws rds describe-db-snapshots \
    --snapshot-type manual \
    --query "reverse(sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}'], &SnapshotCreateTime))[].DBSnapshotIdentifier" \
    --output text 2>/dev/null || echo "")
  if [ -z "$all" ] || [ "$all" = "None" ]; then
    echo "  No hay backups de $db_id -> nada que podar."
    return 0
  fi

  # `tr` porque --output text separa por tabs en una sola linea.
  count=$(echo "$all" | tr '\t' '\n' | grep -c . || true)
  if [ "$count" -le "$keep" ]; then
    echo "  $count backup(s) <= retencion ($keep) -> nada que podar."
    return 0
  fi

  victims=$(echo "$all" | tr '\t' '\n' | tail -n +$((keep + 1)))
  echo "  $count backups, retencion $keep -> borrando $((count - keep)):"
  for s in $victims; do
    echo "    - $s"
    aws rds delete-db-snapshot --db-snapshot-identifier "$s" >/dev/null
  done
}
```

**(b) `tasks/ops.yml`** — `teardown` respalda antes de destruir y verifica
después; `rebuild` delega la decisión al resolver compartido. Reemplazar el
bloque `teardown`/`rebuild` y agregar las dos tareas de apoyo:

```yaml
  # ═══ Teardown / Rebuild ═════════════════════════════════════════════════════

  teardown:
    desc: "Backup + destroy de modulos volatiles. Preserva storage + network"
    prompt: "Se tomara un backup del RDS y se destruiran los modulos volatiles. Storage (S3+ECR) y network (VPC) quedan. Continuar?"
    cmds:
      # RELEASE_NET=false: liberar ALB/NAT aqui seria redundante — el destroy
      # de module.mlflow y el apply enable_nat=false de abajo hacen lo mismo.
      - task: down
        vars: { RELEASE_NET: "false" }
      # PASO 1 — BACKUP ANTES DE TOCAR NADA. Si esto falla, se aborta con la
      # infra intacta: preferimos un teardown que no ocurre a datos perdidos.
      # `down` (arriba) paro el RDS; ensure_backup lo re-arranca (ensure_rds_
      # available) porque AWS no snapshotea instancias detenidas.
      # BACKUP_MAX_AGE_MIN>0 permite reintentar un teardown fallido reutilizando
      # el backup recien tomado, sin pagar otros ~8 min de espera.
      - 'echo ">>> [1/4] Backup del RDS ANTES de destruir (obligatorio)..."'
      - |
        set -euo pipefail
        source tasks/lib/snapshot.sh
        if ensure_backup "{{.RDS_ID}}" "backup" "{{.BACKUP_MAX_AGE_MIN}}" >/dev/null; then
          echo "OK backup verificado. El destroy ya no puede perder datos."
        else
          echo "AVISO el RDS no existe -> nada que respaldar; se sigue con el destroy."
        fi
      # PASO 2 — destroy. rds_skip_final_snapshot=true: el backup del paso 1 ya
      # esta tomado y VERIFICADO, un final_snapshot seria un duplicado de ~8 min.
      - 'echo ">>> [2/4] Destroy de modulos volatiles (orden reverso de apply)..."'
      - |
        set -euo pipefail
        source tasks/lib/nuke.sh
        # El RDS tiene deletion_protection=true -> levantarla antes del destroy.
        lift_rds_protection "{{.RDS_ID}}"
        for mod in {{.VOLATILE_MODULES}}; do
          echo ">>> terraform destroy -target=$mod"
          terraform -chdir={{.TF_DIR}} destroy -target=$mod -auto-approve \
            -var "rds_skip_final_snapshot=true" || {
            echo "FAIL destroy de $mod fallo. Revisar manualmente."
            echo "     Tus datos ESTAN a salvo: el backup del paso 1 ya existe."
            exit 1
          }
        done
      # Libera el NAT gateway + EIP (~$33/mes idle) sin tocar VPC/subnets/SGs.
      - 'echo ">>> [3/4] Liberando NAT gateway (enable_nat=false)..."'
      - terraform -chdir={{.TF_DIR}} apply -target=module.network -var enable_nat=false -auto-approve
      # PASO 4 — verificar + podar. assert_backup_exists convierte una perdida
      # silenciosa en un error visible mientras todavia se puede reaccionar.
      - 'echo ">>> [4/4] Verificando el backup + poda (retencion {{.SNAPSHOT_KEEP}})..."'
      - |
        set -euo pipefail
        source tasks/lib/snapshot.sh
        assert_backup_exists "{{.RDS_ID}}"
        prune_snapshots "{{.RDS_ID}}" "{{.SNAPSHOT_KEEP}}"
      - 'echo "OK teardown completo (NAT liberado). Para volver: task rebuild"'
      - 'echo "     MLflow Registry + forecasts viven en el backup; rebuild los restaura."'
      - 'echo "     Los artifacts (modelos, reports) siguen intactos en S3."'

  # Apply completo que RESTAURA el RDS del backup cuando corresponde. Es el
  # camino UNICO de "levantar": lo comparten `task deploy` (oleada C) y
  # `ops:rebuild`, para que los dos den el mismo resultado y no exista una
  # variante que pierda datos en silencio.
  #
  # Un apply a secas crearia una instancia VACIA y el backup quedaria huerfano:
  # ese era el bug historico —teardown -> rebuild perdia registry + forecasts
  # sin avisar—.
  apply-restore:
    desc: "terraform apply completo restaurando el RDS del ultimo backup si existe. Vars: SNAPSHOT=<id>|none"
    vars:
      SNAPSHOT: '{{.SNAPSHOT | default ""}}'
    cmds:
      - |
        set -euo pipefail
        source tasks/lib/snapshot.sh
        SNAP=$(resolve_restore_snapshot "{{.RDS_ID}}" "{{.SNAPSHOT}}")
        if [ -n "$SNAP" ]; then
          echo ">>> Restaurando RDS desde $SNAP (~5-10 min extra)..."
          terraform -chdir={{.TF_DIR}} apply -auto-approve \
            -var "rds_snapshot_identifier=$SNAP"
        else
          terraform -chdir={{.TF_DIR}} apply -auto-approve
        fi

  rebuild:
    desc: "Re-apply de modulos volatiles + up. RESTAURA el RDS desde el ultimo backup (SNAPSHOT=<id> para fijar uno, SNAPSHOT=none para empezar vacio)"
    vars:
      SNAPSHOT: '{{.SNAPSHOT | default ""}}'
    cmds:
      - 'echo ">>> Apply completo (modulos volatiles se re-crean, resto no-op)..."'
      - task: apply-restore
        vars: { SNAPSHOT: '{{.SNAPSHOT}}' }
      - task: up

  backups:
    desc: "Listar los backups del RDS (los que puede consumir `deploy`/`rebuild`)"
    aliases: [snapshots]
    silent: true
    cmds:
      - |
        source tasks/lib/snapshot.sh
        list_snapshots "{{.RDS_ID}}"

  backup-now:
    desc: "Tomar un backup del RDS AHORA (sin destruir nada). Arranca el RDS si esta parado"
    aliases: [snapshot-now]
    silent: true
    cmds:
      - |
        set -euo pipefail
        source tasks/lib/snapshot.sh
        backup_now "{{.RDS_ID}}" "manual" >/dev/null

  verify-backup:
    desc: "Verificar que existe al menos un backup restaurable del RDS (exit 1 si no)"
    silent: true
    cmds:
      - |
        set -euo pipefail
        source tasks/lib/snapshot.sh
        assert_backup_exists "{{.RDS_ID}}"
```

Y en el bloque `vars:` de `tasks/ops.yml`, junto a `SNAPSHOT_KEEP`:

```yaml
vars:
  # Cuantos backups del RDS conserva el teardown al podar. Con 1 ciclo por
  # semana (deploy miercoles / teardown jueves), 6 = ~6 semanas de historia.
  SNAPSHOT_KEEP: '{{.SNAPSHOT_KEEP | default "6"}}'
  # Reutilizar un backup mas nuevo que N minutos en vez de crear otro. 0 =
  # siempre crear uno nuevo. >0 sirve para reintentar un teardown que fallo
  # despues de respaldar, sin pagar otra vez los ~8 min de espera.
  BACKUP_MAX_AGE_MIN: '{{.BACKUP_MAX_AGE_MIN | default "0"}}'
```

**(c) `Taskfile.yml`** — `deploy` busca backups igual que `rebuild`, y `destroy`
respalda antes. La Oleada C de `deploy` delega en la tarea compartida (el
Taskfile raíz no define `TF_DIR`, y duplicar el bloque reintroduciría justo la
divergencia que este diseño elimina):

```yaml
      # La implementacion vive en ops:apply-restore, compartida con `rebuild`.
      - 'echo ">>> Oleada C: apply resto (network, mlflow, batch, ...)..."'
      - task: ops:apply-restore
        vars: { SNAPSHOT: '{{.SNAPSHOT}}' }
```

...declarando `SNAPSHOT: '{{.SNAPSHOT | default ""}}'` en las `vars:` de `deploy`
(acepta `SNAPSHOT=none` y `SNAPSHOT=<id>` igual que `rebuild`), y
`BACKUP_MAX_AGE_MIN` en las `vars:` globales (el include `ops:` sólo recibe las
vars listadas en `includes:`, así que el default vive declarado en los dos
lados — si cambiás uno, cambiá el otro).

Y en `destroy`, insertar el backup como **primer** paso, antes de vaciar nada:

```yaml
  destroy:
    desc: "AWS DESTRUCTIVO: backup del RDS, drena Batch, vacia S3/ECR, terraform destroy total"
    prompt: "Destruira envs/prod (S3 + ECR + RDS + ...). Los artifacts de S3 NO se recuperan. Continuar?"
    cmds:
      - task: ops:down
      # El backup va primero: si el destroy falla a la mitad, los datos del RDS
      # ya estan a salvo. OJO: esto NO respalda los artifacts de S3 — ver #8.7.
      - 'echo ">>> Backup del RDS antes de destruir..."'
      - |
        set -euo pipefail
        source tasks/lib/snapshot.sh
        ensure_backup "{{.RDS_ID}}" "predestroy" "{{.BACKUP_MAX_AGE_MIN}}" >/dev/null \
          || echo "AVISO el RDS no existe -> nada que respaldar."
      - 'echo ">>> Vaciando buckets S3 versionados + purgando ECR..."'
      # ... (resto del destroy sin cambios)
```

En `tasks/infra.yml::destroy`, el `-var` del final snapshot ya no hace falta
(el backup lo toma quien llama): pasar `-var "rds_skip_final_snapshot=true"`.

#### Periodo de gracia de RDS — por que el teardown lo evita

> Esta limitacion aplica a `task sleep` (que **para** el RDS), no al teardown
> (que lo **destruye**). Se documenta aca porque es la razon principal para
> preferir teardown en idle largo.

RDS auto-arranca despues de **7 dias** de estar stopped (limitacion AWS). Si la
instancia arranca sola y nadie la vuelve a parar, se factura completa sin que
nadie la use. Opciones:

- **Teardown (recomendada para 1+ semana)**: no hay instancia que pueda
  auto-arrancar. El estado queda en el backup y `task rebuild` lo
  restaura. Es el flujo automatizado descrito arriba.
- **Lambda `ml-training-rds-keepstop`** (Parte 3.11): detecta que el RDS
  arranco solo y lo vuelve a parar (cron cada 6h). Sirve mientras el stack esta
  meramente dormido, pero **se destruye con el teardown** (es parte de
  `module.lambdas`), asi que no protege una hibernacion larga hecha con `sleep`
  + liberacion de red. `tasks/ops.yml::down` lo advierte explicitamente.

### 8.6 REBUILD — volver despues de tear-down

Cuando lo uso: vuelvo de vacaciones, retomo el proyecto, necesito la UI
de MLflow para mirar runs viejos.

**Precondicion**: la cuenta tiene los recursos preservados del tear-down
(buckets S3, ECR, IAM, OIDC).

#### Comando `task rebuild`

```bash
task rebuild                       # restaura desde el backup mas reciente (default)
SNAPSHOT_ID="reemplazar-con-el-id-real"
task rebuild SNAPSHOT="$SNAPSHOT_ID" # restaura uno específico (ver `task backups`)
task rebuild SNAPSHOT=none         # arranca con un RDS VACIO (descarta el historico)
# Pasos internos (tasks/ops.yml :: rebuild):
#  1. resolve_restore_snapshot() decide que restaurar (ver #8.5). Es la MISMA
#     funcion que usa `task deploy`, no logica duplicada:
#     - si el RDS YA existe -> no restaura nada (apply normal, idempotente).
#     - SNAPSHOT=none -> instancia vacia.
#     - SNAPSHOT=<id> -> ese (se valida que exista y este `available`).
#     - por defecto -> el backup mas reciente `available`.
#       Si no hay ninguno (stand-up desde cero) sigue con instancia vacia.
#  2. terraform apply COMPLETO (sin -target) con -var rds_snapshot_identifier=<id>:
#     los modulos volatiles se re-crean, el resto es no-op.
#  3. ops:up -> invoca la Lambda scheduler (action=start): arranca RDS (~5 min
#     cold start) + servicios Fargate de forma secuencial y espera ALB 200.
#  El ALB DNS nuevo cambia respecto al stand-up original.
```

> [!IMPORTANT]
> El rebuild restaura **Model Registry + tabla `forecasts`** desde el backup que
> dejó el teardown. Sin ese restore (comportamiento anterior a esta versión) el
> apply creaba un RDS **vacío** y el histórico se perdía en silencio.
> Si el resultado no es el esperado, `task backups` lista los candidatos y
> `task rebuild SNAPSHOT=<id>` permite elegir otro.
>
> **`deploy` y `rebuild` restauran igual.** Comparten `resolve_restore_snapshot()`
> (#8.5), así que no hay un camino "correcto" y otro que pierde datos. La
> diferencia es de alcance, no de seguridad: `deploy` además corre el pre-check
> de drift y re-buildea las 5 imágenes; `rebuild` asume que ya están en ECR.
>
> **Ojo si es tu primera vez**: esta sección presupone que ya hubo un teardown.
> En una cuenta virgen no hay backups y el flujo correcto es `task deploy`
> (#1.1) — ver la nota "la primera vez NO hay backups" en #8.5.

**Tiempo**: 25-40 min, dominado por el restore del snapshot (~5-10 min) + RDS
cold start (5 min) + Fargate task launch (~3 min) + ALB target registration (~2 min).

**Lo unico que cambia respecto al stand-up original**: el DNS del ALB.
Si tenías bookmark, actualizalo. (Si agregaste un dominio (TLS, hardening)
custom via Route53, el dominio sigue igual; sólo el record A apunta al
nuevo ALB.)

### 8.7 DESTROY — eliminar TODO de la cuenta AWS

> [!CAUTION]
> **Operación irreversible**. `task destroy` borra buckets S3 (incluido
> `tfstate`), repos ECR (todas las tags), la instancia RDS (Model Registry
> completo), los IAM roles y el OIDC provider. Pide confirmación textual
> **dos veces** (dos prompts y/n de Task). Antes de correrlo: hacer los 3
> backups de la sub-sección siguiente (export Registry → JSON, snapshot
> manual RDS, export tfstate) — sin ellos no hay forma de recuperar.
>
> **Ojo con la credencial**: nunca exportes el password a `/tmp` ni lo subas como
> objeto S3. Para una migración, conserva el secret en un stack de custodia
> independiente con KMS y permisos mínimos, o restaura el snapshot y rota el
> password master con RDS antes de conectar aplicaciones. Un snapshot no queda
> “inservible” por haber eliminado el secret; el password puede restablecerse.

> [!IMPORTANT]
> **`destroy` NO es "teardown pero más".** Es la diferencia entre pausar y
> cerrar, y se confunde justamente porque el RDS se comporta igual en los dos:
>
> | | `teardown` | `destroy` |
> |---|---|---|
> | Backup del RDS antes de destruir | ✅ | ✅ |
> | `deploy`/`rebuild` restauran ese backup | ✅ | ✅ |
> | Artifacts en S3 (modelos, reports) | ✅ **intactos** | ❌ **buckets vaciados** |
> | Excels de entrada en S3 | ✅ intactos | ❌ vaciados |
> | Imágenes en ECR | ✅ intactas | ❌ purgadas |
> | Credencial master del RDS | ✅ se preserva | ❌ `purge_secret` |
>
> O sea: tras un `destroy`, `task deploy` **sí** te devuelve el Model Registry —
> pero apuntando a `.joblib` y reports que ya no existen si no los archivaste.
> La credencial se rota después del restore; los artifacts sí deben copiarse o
> regenerarse.
>
> **Para el ciclo recurrente de prender/apagar usá `teardown` (#8.5), nunca
> `destroy`.**

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

**ANTES de destruir, archivar esto fuera del proyecto** (el RDS lo respalda
`task destroy` solo; lo demás no lo respalda nadie):

```bash
# Pre-requisito: bucket de archivo FUERA del proyecto, idealmente en otra
# cuenta. Este ejemplo aplica el mínimo: versioning, cifrado y bloqueo público.
export ARCHIVE_BUCKET="${PROJECT}-archive-${ACCOUNT_SUFFIX}"
if ! aws s3api head-bucket --bucket "${ARCHIVE_BUCKET}" 2>/dev/null; then
  aws s3api create-bucket --bucket "${ARCHIVE_BUCKET}" --region "${AWS_DEFAULT_REGION}"
  aws s3api put-bucket-versioning --bucket "${ARCHIVE_BUCKET}" \
    --versioning-configuration Status=Enabled
  aws s3api put-bucket-encryption --bucket "${ARCHIVE_BUCKET}" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-public-access-block --bucket "${ARCHIVE_BUCKET}" \
    --public-access-block-configuration \
    'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'
fi

# (1) Export del Model Registry a JSON (corre con MLflow encendido)
export MLFLOW_TRACKING_URI="http://<ALB-DNS>/"
mlflow models search > artifacts/model-registry-export.json
aws s3 cp artifacts/model-registry-export.json \
  "s3://${ARCHIVE_BUCKET}/ml-training-$(date +%Y-%m-%d)/"

# (2) Backup del RDS (queda independiente de la instancia)
#     `task destroy` ya lo hace solo (paso 2 del comando, ver abajo); esto es
#     el equivalente manual. Automatizado y con espera: `task ops:backup-now`.
aws rds create-db-snapshot \
  --db-instance-identifier ml-training-mlflow \
  --db-snapshot-identifier "ml-training-mlflow-final-$(date +%Y-%m-%d)"

# (2-bis) NO exportar la credencial. Si la restauración será posterior al nuke:
#     a) conservar el secret en un stack externo de custodia; o
#     b) restaurar el snapshot y rotar el master password con `modify-db-instance`.

# (2-ter) Los ARTIFACTS. `task destroy` vacia los buckets del proyecto: si no
#     los copias afuera, los modelos y reports no vuelven (el backup del RDS
#     solo trae la metadata que los referencia).
aws s3 sync "s3://ml-training-artifacts-${ACCOUNT_SUFFIX}/" \
  "s3://${ARCHIVE_BUCKET}/ml-training-$(date +%Y-%m-%d)/artifacts/"
aws s3 sync "s3://ml-training-data-${ACCOUNT_SUFFIX}/" \
  "s3://${ARCHIVE_BUCKET}/ml-training-$(date +%Y-%m-%d)/data/"

# (3) El state contiene secretos. Exportarlo solo si el bucket de archivo tiene
#     acceso restringido y cifrado; nunca adjuntarlo a tickets ni commits.
cd infra/envs/prod
terraform state pull > /tmp/tfstate-final-backup.json
aws s3 cp /tmp/tfstate-final-backup.json \
  "s3://${ARCHIVE_BUCKET}/ml-training-$(date +%Y-%m-%d)/"
rm -f /tmp/tfstate-final-backup.json
cd ../../..
```

> **Por que un bucket separado**: `task destroy` borra todos los buckets
> del proyecto (`ml-training-data-*`, `ml-training-artifacts-*`,
> `ml-training-tfstate-*`). Si pusieras los backups ahi, se borrarian
> en el mismo apply. El `${PROJECT}-archive-${ACCOUNT_SUFFIX}` queda
> intacto porque Terraform no lo conoce.

#### Comando `task destroy`

```bash
task destroy
# Pide confirmacion: Task pide "y" en dos prompts.
# Pasos internos (Taskfile :: destroy):
#  1. ops:down (apaga scheduler/RDS/Fargate; aborta si hay jobs RUNNING).
#  2. ensure_backup del RDS: backup verificado ANTES de tocar nada (#8.5).
#     Cubre la metadata (Registry + forecasts), NO los artifacts de S3.
#  3. Vaciar buckets versionados data + artifacts (incluye versions + delete markers).
#     <- ACA se pierden los modelos y los reports. Es irreversible.
#  4. purge_ecr de los 5 repos (ml-training, -mlflow, -reports, -api, -ui).
#  5. infra:destroy -> terraform destroy total de envs/prod (incluye S3 + RDS),
#     con rds_skip_final_snapshot=true porque el backup del paso 2 ya existe.
#  6. purge_secret del rds-password. Para restaurar después, rotar el password
#     del RDS restaurado o usar un secret de custodia externo.
# NOTA: el bucket tfstate y el OIDC provider NO los borra `task destroy`;
#       para eso esta `task nuke` (destroy + empty tfstate + delete_oidc).
```

**Tiempo**: 30-45 min, dominado por el vaciado de buckets versionados
(cada modelo es ~10 MB con N versions).

### 8.8 Verificar que NO quedo nada (post-destroy)

> [!IMPORTANT]
> `terraform destroy` **no deja la cuenta perfectamente limpia**. Hay residuos
> que no viven en el state y que por lo tanto nadie borra. Correr **siempre**:
>
> ```bash
> task infra:verify-clean              # audita 20 categorias; exit 1 si queda algo
> task infra:verify-clean PURGE=true   # ademas borra los residuos borrables
> ```

Que se queda atras y por que:

| Residuo | Por que sobrevive | Cuesta? |
|---|---|---|
| **Task definitions ECS INACTIVE** | `destroy` las *deregistra*, no las borra; quedan listadas para siempre | No, pero ensucian |
| **Snapshots RDS finales** | Sobreviven **a proposito** (`skip_final_snapshot=false`) | Si, por GB usado |
| **Target groups / DB subnet groups** | Solo si un destroy fallo a la mitad: nombre fijo, fuera del state | No, pero **rompen el proximo deploy** con `already exists` |
| **Secrets en pending deletion** | Recovery window de 30 dias | No |

Detalles verificados el **2026-07-20** ejecutando el ciclo completo:

- Borrar task definitions es `aws ecs delete-task-definitions` — **PLURAL**. El
  singular `delete-task-definition` **no existe**; si se ignora el stderr falla
  en silencio y parece que funciono. Acepta hasta 10 ARNs por llamada.
- Tras un `nuke`, los snapshots quedan pero su secret fue purgado -> son
  **irrestaurables**. Si no vas a volver, borralos (`PURGE=true` lo hace).
- Un destroy a medias dejo huerfanos `<project>-tg-api`, `<project>-tg-mlflow`
  y el subnet group `<project>-rds-subnets`; el siguiente `deploy` fallo con
  `already exists` hasta borrarlos.

> **Ojo al auditar a mano**: `task infra:verify-clean` filtra por el prefijo
> `{{.PROJECT}}`. Recursos de **otros** proyectos en la misma cuenta (EIPs,
> instancias EC2, etc.) apareceran en un `aws ec2 describe-*` generico pero **no
> son de este stack** — comprobar el tag `Project` antes de borrar nada.

---

## Parte 9 — Costos detallados

> [!IMPORTANT]
> Los importes son una estimación de referencia, no una cotización. Antes de
> desplegar, recalcular en AWS Pricing Calculator para la región, fecha, moneda
> y patrón reales. Incluir ALB LCU, NAT por hora y por GB, IPv4 pública,
> almacenamiento y snapshots RDS, CloudWatch ingestion/retention, métricas
> custom, ECR, transferencia inter-AZ e impuestos. Los tiempos de uso también
> deben salir de Cost Explorer, no de una suposición permanente de 69 horas.

> **Por qué esta parte**: AWS no muestra un total previsto antes de gastar; sin
> entender el desglose te llevás sorpresa en la factura. Acá está el número
> realista por modo de operación y los `dials` para bajarlo.

### Mapa del camino — Parte 9

Parte 9 es referencia: tablas de costos para razonar "cuánto cuesta cada modo" y "cuánto bajo apagando X". Llegás a la tabla que responde tu pregunta.

**Prerrequisitos:**

- Infra desplegada o conocida (sabés cuántas variedades, Multi-AZ on/off, etc.).
- AWS Cost Explorer habilitado: Billing → Cost Explorer → Enabled (una vez por cuenta).
- Tags `Project` aplicados (Parte 3, sección 3.2): `aws ce get-cost-and-usage --group-by Type=TAG,Key=Project` muestra `ml-training`.

```mermaid
flowchart TD
    Q([Tu pregunta de costos])

    subgraph Base[Presupuesto base]
        B1["Operar normal Mi+Ju 08-16<br/>≈ <b>$29/mes</b><br/>→ 9.1"]
    end

    subgraph Alt[Escenarios alternativos]
        A1["Hibernado (tear-down) ≈ $3/m<br/>(NAT liberado en teardown)<br/>→ 9.2"]
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

- Los números son estimados con asunciones específicas (10 trainings/mes, 5 GB S3, 69h/mes encendido + 139h/mes de existencia). Tus números reales se desvían según variedades, frecuencia de re-train y tráfico ALB+NAT.
- El item que más pesa estando operando es **Fargate: $20.54/mes** (49% del total), repartido en 4 tasks. El NAT dejó de ser el líder al pasar a teardown semanal ($32 → $7). Para bajar Fargate: recortar la ventana de 12h, o apagar `reports`/`ui` cuando no se usen (son los dos FARGATE_SPOT, ~$5 juntos).
- Los "(futuro)" en 9.4 son intencionales: cada optimización cuesta horas de ingeniería. Re-leer a los 60 días de operación real.

> **Gotcha Parte 9**: confundir las horas de **encendido** con las de **existencia**. El default lockeado es Mi+Ju 08-16 = 69h/mes de Fargate y RDS, pero el ALB y el NAT facturan las ~139h que el stack existe (incluida la noche del miércoles con todo apagado) — y facturarían 720h si te olvidás del `teardown` del jueves, que es el error de $42/mes. Validar con `aws events describe-rule --name ml-training-start` y comparar con Cost Explorer (±20% de #9.1).

### 9.1 Escenario lockeado: ciclo miércoles+jueves 08-16 PET — ~$29/mes

Hay **dos relojes distintos** y confundirlos es el error clásico al estimar:

| Reloj | Qué factura | Horas/mes |
|---|---|---|
| **Existencia** — el recurso existe en la cuenta | ALB, NAT GW, storage del RDS | ~139h (miércoles 08:00 → jueves 16:00 = 32h/semana × 4.3) |
| **Encendido** — el scheduler lo tiene arriba | Fargate ×4, cómputo del RDS | ~69h (8h × 2 días × 4.3) |

La diferencia son las ~16h de la noche del miércoles: el ALB y el NAT siguen
facturando (existen) mientras el scheduler tiene Fargate en 0 y el RDS parado.
Es exactamente lo que el auto-stop ahorra en este ciclo — nada más.

Asume además: 10 trainings/mes (1h cada uno), 5 GB de S3, ~3.5 GB de ECR (5 repos).

| Item | Calculo | Mensual (USD) |
|---|---|---|
| S3 (5 GB Standard) | 5 × $0.023 | $0.12 |
| S3 (versiones no-current con lifecycle 90d) | ~10 GB | $0.23 |
| ECR (~3.5 GB, 5 repos) | 3.5 × $0.10 | $0.35 |
| RDS db.t4g.small (69h encendido; hostea MLflow + forecasts) | 69 × $0.032 | $2.21 |
| RDS storage (20 GB gp3, solo mientras la instancia existe) | 20 × $0.115 × (139/720) | $0.44 |
| RDS snapshots (retencion 6 = ~6 semanas, ~10 GB) | 10 × $0.095 | $0.60 |
| Fargate MLflow (**1 vCPU, 3 GB** tras el rightsizing de #9.4.2, 69h) | 69 × ($0.04048 × 1 + $0.004445 × 3) | $3.71 |
| Fargate Reports (0.5 vCPU, 1 GB, 69h) — corre en **FARGATE_SPOT** (~70% más barato; el monto listado es el techo on-demand) | 69 × ($0.04048 × 0.5 + $0.004445 × 1) | $1.70 |
| Fargate API (1 vCPU, 2 GB, 69h) | 69 × ($0.04048 × 1 + $0.004445 × 2) | $3.41 |
| Fargate UI (0.5 vCPU, 1 GB, 69h) — corre en **FARGATE_SPOT** (~70% más barato; el monto listado es el techo on-demand) | 69 × ($0.04048 × 0.5 + $0.004445 × 1) | $1.70 |
| ALB (139h de existencia, no 720) | 139 × $0.0225 | $3.13 |
| ALB LCU | <0.5 LCU promedio | $0.15 |
| NAT Gateway (139h de existencia, no 720) | 139 × $0.045 | $6.25 |
| NAT egress data | 10 GB | $0.45 |
| EC2 Spot c6i.2xlarge (10 jobs × 1h) | 10 × $0.102 | $1.02 |
| Lambdas (negligible) | ~1000 invocs/mes | $0.10 |
| EventBridge | ~50 events/mes | $0.10 |
| SNS | ~10 publishes/mes | $0.01 |
| CloudWatch Logs (14d retention, 1 GB) | 1 × $0.50 | $0.50 |
| CloudWatch Custom Metrics (N MAPE + 3 base; N=6 ejemplo) | (N+3) × $0.30 | $2.70 |
| CloudWatch PutMetricData API | ~10k calls/mes | $0.01 |
| Data transfer (ALB out a Internet) | 5 GB | $0.45 |
| **Total** | | **~$29** |

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

**Por que el numero bajo de ~$75 a ~$29**, en tres movimientos:

| Movimiento | Delta | Acumulado |
|---|---|---|
| Baseline anterior (L-V 08-12, ALB+NAT 24/7) | — | $75 |
| **Teardown semanal**: ALB+NAT pasan de 720h a ~139h de existencia | **−$41** | $34 |
| **Ventana 8h × 2 dias** (69h encendido, antes 80h) | −$2 | $32 |
| **Rightsizing MLflow** 2 vCPU/4 GB → 1 vCPU/3 GB (#9.4.2) | −$3 | **$29** |

El 90% del ahorro vino del primer movimiento. ALB + NAT eran $48.60/mes —el 65%
del total— porque facturaban 24/7 **aunque el scheduler apagara todo lo demas**:
el auto-stop nunca los toco. Destruirlos cada jueves es lo que mueve la aguja.

Los otros dos son de segundo orden pero baratos de aplicar (dos numeros en
`envs/prod`, cero cambios de arquitectura). Post-rightsizing el reparto queda
mas parejo: Fargate $10.53 (36%), red $9.38 (32%), el resto observabilidad y
storage — ya no hay un item dominante que atacar.

### 9.2 Comparativa con escenarios alternativos

| Escenario | Cambio vs default | Costo total/mes | Cuando elegirlo |
|---|---|---|---|
| **Hibernado** | tear-down (sección 8.5; libera NAT vía `enable_nat=false`) | ~$3 | Las 5 semanas/mes que el stack no existe |
| **Default (lockeado)** | Ciclo Mi+Ju 08-16 PET + teardown semanal + MLflow rightsizeado | **~$29** | Operacion normal |
| **Ventana 12h (corte 20:00)** | `work_end_hour_local = 20` | ~$37 | Si necesitás las tardes; +$2/mes por hora de ventana |
| **Sin teardown** | Mismo scheduler, pero el stack queda parado en vez de destruido (ALB+NAT 24/7) | ~$71 | Nunca: son $42/mes de red ociosa |
| **24/7** | Scheduler OFF, stack (MLflow+Reports+API+UI) + RDS siempre on | ~$195 | Equipo distribuido multi-timezone |
| **No-NAT** | VPC endpoints en vez de NAT GW (hardening, futuro) | ~$36 | Trafico NAT < 10 GB/mes |
| **Multi-AZ RDS** | RDS Multi-AZ (hardening, futuro) | +$13 sobre default | Compliance / SLA estricto |
| **TLS + Custom Domain** | Route 53 zone + ACM cert (hardening, futuro) | +$1 sobre default | Exposicion publica |

### 9.3 Matriz de costos por modo de lifecycle

Tabla resumen que cruza los 4 modos (STAND-UP / TEAR-DOWN / DESTROY) con
los recursos: util para razonar "cuanto bajo apagando X" antes de
ejecutar `task teardown` o `task destroy` (los modos en si estan
documentados en secciones 8.5 a 8.7).

| Recurso | STAND-UP (operando) | TEAR-DOWN (hibernado) | DESTROY (vacio) |
|---|---|---|---|
| S3 (todos los buckets, ~5 GB) | $0.12 | $0.12 | $0 |
| ECR (5 repos, ~3.5 GB) | $0.35 | $0.35 | $0 |
| RDS db.t4g.small (Mi+Ju 08-16 PET ≈ 69h/mes) | $2.21 | $0 (destruido) | $0 |
| RDS allocated storage (20 GB gp3, prorrateado 139/720h) | $0.44 | $0 (instancia destruida) | $0 |
| RDS snapshots manuales (retencion 6 ≈ 6 semanas, ~10 GB) | $0.60 | $0.60 | $0 |
| ECS Fargate MLflow (1 vCPU, 3 GB, 69h/mes) | $3.71 | $0 | $0 |
| ECS Fargate Reports (0.5 vCPU, 1 GB, 69h/mes) | $1.70 | $0 | $0 |
| ECS Fargate API (1 vCPU, 2 GB, 69h/mes) | $3.41 | $0 | $0 |
| ECS Fargate UI (0.5 vCPU, 1 GB, 69h/mes) | $1.70 | $0 | $0 |
| ALB (139h de existencia) | $3.13 | $0 (borrado) | $0 |
| NAT Gateway (139h de existencia) | $6.25 | $0 (liberado vía `enable_nat=false`) | $0 |
| Batch EC2 (Spot c6i.2xlarge, ~10 jobs/mes × 1h) | $1.02 | $0 | $0 |
| Lambdas (negligible) | $0.10 | $0.10 | $0 |
| EventBridge / SNS / CloudWatch | $3.30 | $0.30 | $0 |
| Data transfer (NAT egress + ALB) | $0.90 | $0 | $0 |
| **Total mensual** | **~$29** | **~$1** | **$0** |

> **Ojo con la columna TEAR-DOWN**: no es un escenario alternativo al STAND-UP,
> es la otra mitad del mismo mes. En el ciclo Mi+Ju convivís con las dos: ~32h
> por semana en STAND-UP y el resto en TEAR-DOWN. El ~$29 de #9.1 **ya integra
> las dos columnas** — no las sumes.

> La suma directa de esta tabla da ~$41; los ~$42 de sección 9.1 incluyen
> items consolidados aca: ALB LCU + S3 lifecycle de versiones. La columna
> hibernado NO cambia con el App stack: el scheduler escala API+UI a 0 igual
> que MLflow/Reports. Tabla pensada como **delta entre modos**, no como suma
> auditable — para esa ver sección 9.1.
>
> **NAT $0 en hibernado ya no es aspiracional**: `task teardown` lo libera con
> `enable_nat=false` (`terraform apply -target=module.network`), preservando
> VPC/subnets/SGs; `task rebuild`/`deploy` lo recrean (default `true`).

Si queres bajar mas el modo operando, ver sección 9.4. El orden de palancas
cambio con el ciclo semanal: ahora manda **Fargate** (#9.4.2), no la red.

### 9.4 Optimizaciones adicionales (futuro)

**Por que se llaman "futuras" en vez de aplicarlas dia 1**: cada una
tiene un costo de ingenieria o un trade-off. Aplicarlas dia 1 te frena
sin que aporten valor hasta que la operacion tenga datos.

#### 9.4.1 VPC endpoints en vez de NAT — ⚠️ DEJO DE CONVENIR

**Con el ciclo semanal esta optimizacion se dio vuelta y ahora sale mas cara.**
Se documenta el porque para que nadie la re-proponga leyendo cifras viejas.

El calculo original comparaba contra un NAT 24/7 ($32/mes). Con `teardown`
semanal el NAT solo existe ~139h/mes → **$6.25**. Del otro lado, los endpoints
de tipo *interface* facturan **$0.01/h por endpoint y por AZ**:

| Opcion | Calculo | Mensual |
|---|---|---|
| NAT GW (hoy) | 139h × $0.045 | **$6.25** |
| 3 interface endpoints (ECR api, ECR dkr, Logs) × 2 AZ | 6 × 139h × $0.01 | **$8.34** |
| Ídem con 1 sola AZ (pierde HA) | 3 × 139h × $0.01 | $4.17 |

El endpoint de S3 es *gateway* y es gratis — vale la pena solo por el egress
($0.45/mes), no por el NAT. **Conclusion**: el teardown semanal ya capturo el
ahorro que perseguian los endpoints; migrar ahora costaria ~100 lineas de
Terraform para gastar ~$2 mas al mes, o ahorrar ~$2 resignando multi-AZ.
No hacerlo.

#### 9.4.2 Rightsizing de las tasks Fargate — ✅ APLICADO (MLflow)

Fargate era el 49% de la factura. Estas opciones **no tocan la arquitectura**:
son cambios de valor, sin mover subredes, ALB, capacity providers ni el ruteo
por path del #6.

| Cambio | Donde | Ahorro/mes | Estado |
|---|---|---|---|
| MLflow 2 vCPU/4 GB → **1 vCPU/3 GB** | `aws_ecs_task_definition.mlflow` (#3.5) | **−$3.10** | ✅ aplicado |
| API 1 vCPU/2 GB → 0.5 vCPU/1 GB | `api_cpu=512`, `api_memory=1024` | −$1.70 | ❌ no aplicar sin medir |

**Por que MLflow quedo en 3 GB y no en 2**: `mlflow server` levanta 4 workers
gunicorn por defecto (no pasamos `--workers`) y el trainer loguea 4 variedades
en paralelo. A ~300 MB de RSS por worker, 2 GB no deja margen para el buffer de
artifacts de `--serve-artifacts`. El GB extra cuesta $0.30/mes; un OOM a mitad
de un run de 1h cuesta mucho mas. Si aparece `OOMKilled` en el log group de
mlflow, el rollback es volver a `2048`/`4096`.

**Por que la API NO se toca**: carga pipelines sklearn en memoria. Con
`api_preload_models=true` o varias variedades cargadas, 1 GB puede OOM — y a
diferencia de MLflow, aca el fallo es cara al usuario. Ahorra $1.70; no vale
el riesgo sin medir el RSS real primero.

La palanca mas grande no es tecnica sino de politica: **cada hora que le sacas
a la ventana diaria son ~$2/mes** (1h × 2 dias × 4.3 semanas × $0.23/h de
Fargate+RDS). El corte ya esta en 16:00 (8h); bajarlo mas empieza a comerse la
jornada util.

> **Lo que NO se recomienda tocar** (romperia la arquitectura, no solo el costo):
> mover las tasks a subredes publicas para eliminar el NAT, sacar el ALB, o
> pasar MLflow a Spot (#9.4.5). Los tres ahorran entre $3 y $7 y a cambio
> rompen, respectivamente, el aislamiento en subredes privadas, el ruteo por
> path que convive con `/api/2.0/mlflow-artifacts/*` (invariante #6), y la
> durabilidad de los runs largos de training.

#### 9.4.3 S3 Intelligent-Tiering

Auto-tier `artifacts/` despues de 90d a Standard-IA (-50% storage).
**Por que no dia 1**: tu volumen S3 es ~5 GB. Ahorro real: <$0.5/mes.
No vale la pena hasta que pases los 100 GB.

#### 9.4.4 ECR scan policies

Borrar imagenes con vulnerabilidades CVSS > 7. **Por que no dia 1**:
genera ruido (la imagen base puede tener CVEs que upstream parchea
en semanas). Aplicar despues de la primera vuelta.

#### 9.4.5 Fargate Spot para MLflow

**Reports y UI YA corren en FARGATE_SPOT** (~70% más baratos; son stateless,
una interrupción solo reinicia la task). El cluster expone capacity providers
`FARGATE` + `FARGATE_SPOT`. **MLflow y API quedan on-demand a propósito**:
MLflow es crítico durante runs largos de training — si Spot lo reclama mid-run,
se pierde el run. 50-70% off Fargate, pero interrupcion = MLflow caido.
**Por que no día 1 para MLflow**: es path-critical para CI/CD; downtime mid-day
rompe tu workflow. Mover MLflow a Spot solo en envs/dev.

---

## Parte 10 — Aseguramiento MLOps y hardening mínimo

Esta Parte no agrega otro orquestador ni sustituye el stack. Conserva Terraform,
AWS Batch, ECS Fargate, MLflow, RDS, S3, CloudWatch, SNS, Lambda, Task y GitHub
Actions. Su objetivo es convertir el camino económico en un sistema cuya
calidad, seguridad y recuperación puedan demostrarse.

### 10.1 Dos perfiles explícitos

| Control | Laboratorio controlado | Producción expuesta |
|---|---|---|
| ALB | HTTP temporal, acceso restringido | HTTPS obligatorio; HTTP solo redirige |
| Autenticación | Red privada o CIDR de operador | Auth para MLflow, API, UI y reportes |
| Artifacts | Navegación local | Sin autoindex público; descarga autorizada |
| RDS | single-AZ aceptado | Según SLO: Multi-AZ o RTO de restore probado |
| Tests | smoke + unitarios mínimos | unitarios, contratos, integración y restore |
| Imágenes | tags de desarrollo | SHA/digest inmutable + SBOM + scan |
| Monitoreo | logs y fallo de jobs | SLO, servicio, data, modelo y costo |
| Promoción | manual controlada | gates + approval + alias + rollback |

Si falta un control de la columna productiva, el sistema sigue siendo útil, pero
se documenta como laboratorio o preproducción.

### 10.2 Seguridad mínima

1. **TLS antes de Internet.** Crear certificado ACM y listener HTTPS; el
   listener 80 solo devuelve redirect 301 a 443. Nunca enviar credenciales,
   datos de predicción o cookies por HTTP.
2. **Autenticar todos los paths.** Proteger MLflow, API, UI y reports. No basta
   con `--allowed-hosts`: ese flag evita DNS rebinding, no autentica usuarios.
3. **No publicar artifacts crudos.** Eliminar `/artifacts/*` del listener
   público y el `autoindex` productivo. La API entrega descargas autorizadas o
   URLs S3 prefirmadas de vida corta.
4. **Separar identidades de base de datos.** El master de RDS se usa solo para
   bootstrap/rotación. MLflow recibe un usuario limitado a la DB `mlflow`; la
   API recibe otro limitado a `forecasts`. Cada credencial vive en un secret
   distinto.
5. **Tratar el state como secreto.** `random_password` y `secret_string`
   aparecen en Terraform state aunque estén marcados sensibles. Restringir
   lectura del bucket, cifrar, auditar con CloudTrail y no copiar el state a
   logs, PRs o artifacts de CI.
6. **OIDC sin wildcard.** Los roles aceptan solo `main` y
   `environment:production`. Una PR no recibe el rol de apply. Si se desea un
   plan remoto en PR, crear un tercer rol de solo lectura.
7. **IAM por prefijo.** El trainer lee el dataset y escribe solo sus prefijos.
   La API no recibe permisos S3 cuando descarga modelos mediante el proxy de
   MLflow. Batch nunca entra directamente a RDS.
8. **Secretos fuera de comandos y archivos temporales.** No exportar passwords
   a `/tmp` ni S3. Para DR, conservarlos en custodia independiente o rotarlos
   después de restaurar.

**Gate de red verificable**

```bash
# RDS no es público
aws rds describe-db-instances \
  --db-instance-identifier "${PROJECT}-mlflow" \
  --query 'DBInstances[0].PubliclyAccessible' --output text
# False

# Las tasks Fargate no reciben IP pública
aws ecs describe-services \
  --cluster "${PROJECT}-cluster" \
  --services mlflow api ui reports \
  --query 'services[].networkConfiguration.awsvpcConfiguration.assignPublicIp'
# DISABLED para todos

# Ningún listener productivo debe terminar en HTTP sin redirect
aws elbv2 describe-listeners --load-balancer-arn "$ALB_ARN" \
  --query 'Listeners[].{Port:Port,Protocol:Protocol,Actions:DefaultActions[].Type}'
```

### 10.3 Suite de pruebas por riesgo

No se persigue cobertura por vanidad. Se prueban los contratos que pueden
producir un modelo silenciosamente incorrecto:

**Unitarias**

- selector de campeón: empate, NaN, candidato sin métrica y violación de gap;
- transformaciones: fit/transform estable, columnas y dtypes;
- métrica MAPE con cero/casi cero y métrica robusta complementaria;
- normalización del dispatcher y rechazo de payloads;
- función de promoción: primer champion, regresión, alias y validate-only.

**Anti-leakage**

- cada fold usa solo timestamps anteriores;
- lags y estadísticas se ajustan dentro del `Pipeline`;
- imputadores, scalers y selección de features no se fitean globalmente;
- el holdout final no participa en Optuna ni en selección de campeón.

**Contratos**

- schema del Excel por hoja;
- columnas requeridas, rangos, nulos, unicidad y monotonía temporal;
- signature MLflow contra el payload real de la API;
- roundtrip `.joblib` dentro de la imagen;
- compatibilidad trainer ↔ API con `models:/<name>@champion`.

**Integración**

- compose completo con Postgres + MLflow + API + UI;
- artifact upload/download a través de `mlflow-artifacts:/`;
- Batch smoke con IAM real y MLflow interno;
- promoción seguida de invalidación de cache y prediction smoke;
- restore de RDS + reconexión de artifacts S3.

CI ejecuta `pytest -q` y falla si no recolecta pruebas. El smoke no reemplaza
esta suite: prueba el camino feliz, no la corrección estadística.

### 10.4 Lineage de datos e idempotencia

El input canónico no debe ser solamente una key mutable. Al enviar un job:

1. el dispatcher ejecuta `HeadObject`;
2. captura `VersionId`, tamaño, ETag y checksum;
3. pasa `S3_DATA_VERSION_ID` al job;
4. el trainer descarga esa versión exacta;
5. calcula SHA-256 completo después de descargar;
6. registra key, VersionId, hash, filas y fecha máxima del dataset en MLflow.

La identidad lógica del entrenamiento es:

```text
run_key = sha256(
  git_commit + image_digest + dataset_sha256 + variety + tuning + seed
)
```

`run_key` se usa como tag y clave de idempotencia. Si Batch reintenta por una
interrupción Spot, el segundo intento no crea dos versiones candidatas sin
relación: reanuda o marca explícitamente el intento anterior.

El JSON de EDA incluye `dataset_sha256`. `task train` ignora cualquier EDA cuyo
hash no coincide con el dataset actual y exige regenerarlo.

### 10.5 Release y supply chain

- Construir una sola vez en CI y promover el mismo SHA/digest.
- Registrar `image_digest` en cada run y en `/api/health`.
- No desplegar `latest` ni `stable` en producción.
- Conservar `.terraform.lock.hcl` y hashes de dependencias Python.
- Generar SBOM por imagen y guardar el resultado asociado al commit.
- Bloquear HIGH/CRITICAL corregibles según una excepción con responsable y
  fecha de expiración; `scan_on_push` por sí solo es informativo.
- Fijar actions de terceros por commit SHA en un entorno regulado.
- Actualizar base images y providers mediante PR con tests y rollback.

### 10.6 Observabilidad MLOps

CloudWatch + SNS ya existen; se amplían las señales, no la plataforma.

**Servicio**

- ALB 4xx/5xx, latencia p50/p95/p99 y targets unhealthy;
- desired/running tasks de ECS y reinicios;
- CPU/memoria de ECS, conexiones/CPU/storage de RDS;
- jobs Batch por estado, tiempo RUNNABLE, duración y reintentos;
- Lambda errors, throttles, duration y DLQ.

**Datos**

- freshness de la última observación;
- filas por variedad, nulos, valores fuera de rango y cambios de schema;
- hash/VersionId no observado anteriormente;
- distribución por ventana temporal.

**Modelo**

- versión y alias cargados;
- distribución de predicciones y features;
- PSI/KS como señales, no como verdad universal;
- error real cuando llega el label: MAPE + MAE/WAPE por variedad y ventana;
- cobertura y ancho de intervalos;
- staleness: días desde último entrenamiento aprobado.

**Negocio**

- volumen de predicciones;
- porcentaje de requests rechazados por contrato;
- sesgo sistemático por variedad;
- costo por training aprobado, no solo costo mensual de infraestructura.

Cada alarma tiene `owner`, severidad, umbral, ventana, acción y enlace a
runbook. Una métrica sin respuesta operativa definida es telemetría, no control.

### 10.7 SLO, RTO y RPO

Definirlos antes de elegir Multi-AZ:

| Objetivo | Perfil económico sugerido |
|---|---|
| Disponibilidad de UI/API en ventana | 99 % mensual durante Mi/Ju 08–16 PET |
| Latencia API | p95 bajo el umbral acordado con negocio |
| RPO metadata MLflow/forecasts | último snapshot verificado |
| RPO artifacts/datasets | 0 para objetos versionados no eliminados |
| RTO tras teardown | 40–60 min, medido con restore real |
| RTO rollback de modelo | 10 min incluyendo cache + smoke |

Estos valores son ejemplos; se reemplazan por compromisos reales. El punto
experto es medirlos. Un snapshot que nunca se restauró no demuestra DR.

**Ejercicio trimestral**

1. restaurar el snapshot más reciente con otro identifier;
2. rotar/inyectar la credencial sin exponerla;
3. comprobar experiments, aliases y tabla `forecasts`;
4. cargar un modelo desde S3 y ejecutar prediction smoke;
5. medir RTO/RPO;
6. destruir el entorno de prueba.

### 10.8 Checklist para declarar release productivo

```text
[ ] HTTPS + autenticación en todos los paths públicos
[ ] /artifacts y autoindex no son públicos
[ ] roles DB separados; master fuera de aplicaciones
[ ] OIDC restringido; PR sin rol de apply
[ ] pytest, lint, Terraform validate y smoke verdes
[ ] imagen por SHA/digest, SBOM y scan aceptado
[ ] dataset key + VersionId + SHA completo registrados
[ ] signature e input example verificados contra API
[ ] gates automáticos + approval + @champion
[ ] rollback ensayado e invalida cache de API
[ ] alarmas de servicio, datos y modelo con owner
[ ] restore de RDS probado y RTO/RPO medidos
[ ] costo recalculado para región y fecha actuales
```

---

## Parte 11 — Troubleshooting (catalogo)

> **Por qué esta parte**: errores que YA pasaron (en V1 o durante el desarrollo
> del V2), cada uno con síntoma, causa y fix concreto. Si tu error no está acá,
> `aws logs tail /aws/batch/ml-training --follow` es siempre el primer paso.

### Mapa del camino — Parte 11

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
| 16 | NAT GW cuesta mas de lo esperado | Casi siempre: te salteaste el `teardown` del jueves y el NAT quedo facturando 24/7 ($32 vs $7). Si el teardown corrio, revisar egress alto (S3 cross-region o ECR pulls grandes) | `task status` + `aws ec2 describe-nat-gateways`; correr `task teardown`. NO migrar a VPC endpoints: con el ciclo semanal salen mas caros (sección 9.4.1) |
| 17 | `task batch:smoke` se cuelga mostrando `POP None`; al invocar la Lambda, `StatusCode=200` pero `body.jobId=null` y `errorMessage: AccessDeniedException ... batch:SubmitJob` (o `batch:TagResource`) | El rol `ml-training-dispatcher` no autoriza el SubmitJob: (a) la Lambda pasa la job-def por NAME, que autoriza contra el ARN **sin** `:revision` y no matchea el patron `...:*`; (b) `submit_job` pasa `tags={...}`, lo cual exige `batch:TagResource` aparte. El `200` es de la *invocacion*, no del resultado — leer `body.errorMessage`, no el StatusCode | Agregar al inline policy (#3.9) el ARN de job-def **sin** revision (junto al `:*`) y la accion `batch:TagResource`. Aplicar: `task infra:apply TARGET='module.lambdas.aws_iam_role_policy.dispatcher'`. Verificar con `aws lambda invoke ... | jq '.body.jobId'` |

---

## Parte 12 — Apendices

### Mapa del camino — Parte 12

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

### Apendice A — Glosario (referencia rapida)

#### A.1 AWS / DevOps

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

#### A.2 Estadistica / EDA (usado en `task eda`)

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

### Apendice B — Conceptos fundamentales (lectura opcional)

> **Skip si sos experto MLOps.** Este apendice arranca desde nivel
> 101 ("por que MLOps y no un script en CRON") y existe por
> completitud — para un lector con experiencia previa en CI/CD,
> tracking servers, model registries y promotion gates, el contenido
> es redundante. Volver aca solo si necesitas defender la eleccion
> de tooling ante alguien no tecnico.

#### B.1 Por que MLOps y no "el script de Python que corre en CRON"

3 problemas que MLOps resuelve:
1. **Reproducibilidad**: tu modelo no se reproduce desde una notebook
   sin Docker + git SHA + MLflow tag.
2. **Auditabilidad**: cuando un cliente reclama una prediccion mala
   de hace 3 meses, sabes que version del modelo predijo y con que
   data.
3. **Observabilidad**: MAPE silencioso es bug silencioso. Alarmas
   automaticas convierten degradacion en pager.

#### B.2 Por que Terraform y no CDK / Pulumi

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

#### B.3 Por que Task si Terraform ya orquesta

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

#### B.4 Por que GitHub Actions con OIDC

OIDC remplaza el caso comun de access keys de larga duracion:

- AWS access keys leakean (PRs maliciosos, dumps de logs accidentales).
- GitHub PAT leakean en clonados.
- OIDC = credenciales de 60 min firmadas por GitHub para tu sub
  (repo:org/repo:ref). Si tu repo se hackea, el atacante NO obtiene
  AWS keys.

### Apendice C — Mapa de archivos creados por la guia

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
├── docs/{01-local,02-produccion-aws,03-arquitectura}.md  # las guias
├── docs/adr/                                    # ADR-001..009
└── (resto del proyecto: README.md, pyproject.toml, requirements*.txt, .dockerignore,
    .env.example, .gitignore, .editorconfig, notebooks/, scripts/, .claude/, etc.
    NOTA: docs/adr/ ya existe (ADR-001..009). Lo que sigue faltando es
    .github/workflows/ — ver la Parte 6.)
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
