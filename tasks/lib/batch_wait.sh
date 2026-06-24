# Helper bash compartido por tasks/batch.yml y tasks/ops.yml (Batch jobs).
# Sourceado, no ejecutado. Requiere awscli configurado.

# Estados "activos" de Batch (trabajo en vuelo). Fuente unica para
# ops:_batch-jobs y batch:status: si AWS cambia el set, se edita aqui.
BATCH_ACTIVE_STATES="SUBMITTED PENDING RUNNABLE STARTING RUNNING"

# Archivo (gitignored) donde se persiste el ultimo jobId submiteado. Permite que
# batch:watch / batch:logs / batch:cancel se defaulten al ultimo job sin tener que
# copiar/pegar el id -> el flujo "submit en background, vuelvo despues a ver" no
# depende de que la terminal haya guardado nada en pantalla.
BATCH_LAST_JOB_FILE="${BATCH_LAST_JOB_FILE:-.batch-last-job}"

# batch_record_job <job_id>: persiste el id (no-op si vacio/None).
batch_record_job() {
  [ -n "$1" ] && [ "$1" != "None" ] && printf '%s\n' "$1" > "$BATCH_LAST_JOB_FILE" 2>/dev/null
  return 0
}

# batch_resolve_job [job_id]: echo el id dado, o el ultimo submiteado si viene vacio.
batch_resolve_job() {
  local jid="$1"
  if [ -z "$jid" ] && [ -f "$BATCH_LAST_JOB_FILE" ]; then
    jid=$(cat "$BATCH_LAST_JOB_FILE" 2>/dev/null)
  fi
  printf '%s' "$jid"
}

# batch_need_job [job_id]: echo el id resuelto a stdout; si no hay ninguno, imprime
# guia a stderr y retorna 1. Preambulo unico de batch:watch/logs/cancel:
#   JOB_ID=$(batch_need_job "{{.JOB_ID}}") || exit 0
batch_need_job() {
  local jid; jid=$(batch_resolve_job "$1")
  if [ -z "$jid" ]; then
    echo "No hay JOB_ID. Pasa JOB_ID=<id>, o submitea con 'task batch:train ...' primero." >&2
    echo "Jobs activos: task batch:status" >&2
    return 1
  fi
  printf '%s' "$jid"
}

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

# dispatch_job <jobdef> <region> <dispatcher_fn> <payload> <label> [wait=true]
# Flujo comun de submit via Lambda dispatcher, compartido por batch:train y
# batch:eda: preflight de imagen -> invoke -> parseo de jobId (con fallback si el
# Lambda envuelve el body como string JSON) -> espera opcional. Un tipo de job
# nuevo = una linea que arma su PAYLOAD y llama aqui.
dispatch_job() {
  local jobdef="$1" region="$2" fn="$3" payload="$4" label="$5" wait="${6:-true}"
  assert_jobdef_image "$jobdef" "$region"
  aws lambda invoke \
    --function-name "$fn" \
    --cli-binary-format raw-in-base64-out \
    --payload "$payload" \
    /tmp/dispatcher-out.json \
    --query 'StatusCode' --output text
  cat /tmp/dispatcher-out.json
  local job_id
  job_id=$(jq -r '.body.jobId // (.body|fromjson|.jobId)' /tmp/dispatcher-out.json 2>/dev/null \
           || jq -r '.jobId' /tmp/dispatcher-out.json)
  echo ">>> Submitted  $label  job=$job_id"
  batch_record_job "$job_id"
  if [ "$wait" != "true" ]; then
    cat <<EOF

  El job corre en AWS Batch: podes cerrar la terminal / apagar la maquina
  sin afectarlo. Para volver a verlo despues:
    task batch:watch                  seguir ESTE job hasta SUCCEEDED/FAILED
    task batch:status                 todos los jobs activos en las queues
    task batch:logs                   tail de los logs (FOLLOW=true para vivo)
    task batch:cancel                 terminar el job
  (todos defaultean a job=$job_id; pasa JOB_ID=<id> para otro)
EOF
    return 0
  fi
  wait_job "$job_id" "$label"
}

# wait_job <job_id> <label>
# Polling cada 30s hasta SUCCEEDED (return 0) o FAILED (return 1). Robusto:
# un job inexistente (id malo / purgado por Batch) corta con return 2 en vez de
# loopear infinito sobre status=None.
wait_job() {
  local job_id="$1" label="$2"
  while :; do
    local status
    status=$(aws batch describe-jobs --jobs "$job_id" --query 'jobs[0].status' --output text 2>/dev/null)
    case "$status" in
      None|"")
        echo "  job '$job_id' no encontrado (id invalido o ya purgado por Batch)"
        return 2
        ;;
    esac
    echo "  $(date +%H:%M:%S)  $label  $status"
    case "$status" in
      SUCCEEDED) return 0 ;;
      FAILED)
        local reason
        reason=$(aws batch describe-jobs --jobs "$job_id" --query 'jobs[0].statusReason' --output text 2>/dev/null)
        echo "FAIL $label  reason=$reason"
        return 1
        ;;
      *) sleep 30 ;;
    esac
  done
}

# follow_job <job_id> <label>
# Wrapper de wait_job para el visor de estado (batch:watch): reporta el desenlace
# pero SIEMPRE retorna 0 -> "ver el estatus" nunca rompe la terminal con un exit
# rojo de task, ni cuando el job termino en FAILED ni cuando el id no existe.
follow_job() {
  wait_job "$1" "$2" || true
  return 0
}

# tail_job_logs <job_id> <log_group> [follow]
# Resuelve el logStreamName del intento actual del job y hace tail de su stream en
# CloudWatch. Robusto: si el stream aun no existe (job RUNNABLE/STARTING) lo avisa
# en vez de fallar.
tail_job_logs() {
  local job_id="$1" log_group="$2" follow="${3:-false}"
  local stream
  stream=$(aws batch describe-jobs --jobs "$job_id" \
             --query 'jobs[0].container.logStreamName' --output text 2>/dev/null)
  if [ -z "$stream" ] || [ "$stream" = "None" ]; then
    echo "  el job '$job_id' todavia no tiene log stream (probablemente RUNNABLE/STARTING)."
    echo "  reintenta en ~1 min, o segui el estado con: task batch:watch JOB_ID=$job_id"
    return 0
  fi
  echo ">>> logs de job=$job_id  stream=$stream  (grupo $log_group)"
  local extra=""
  [ "$follow" = "true" ] && extra="--follow"
  aws logs tail "$log_group" --log-stream-names "$stream" --since 6h $extra 2>/dev/null \
    || echo "  (sin eventos aun; reintenta en unos segundos)"
}
