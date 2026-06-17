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
