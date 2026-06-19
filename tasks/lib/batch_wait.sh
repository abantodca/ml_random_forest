# Helper bash compartido por tasks/batch.yml y tasks/ops.yml (Batch jobs).
# Sourceado, no ejecutado. Requiere awscli configurado.

# Estados "activos" de Batch (trabajo en vuelo). Fuente unica para
# ops:_batch-jobs y batch:status: si AWS cambia el set, se edita aqui.
BATCH_ACTIVE_STATES="SUBMITTED PENDING RUNNABLE STARTING RUNNING"

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
  [ "$wait" != "true" ] && return 0
  wait_job "$job_id" "$label"
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
