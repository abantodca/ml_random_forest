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

# Path relativo al repo root (CWD cuando go-task corre las tasks), igual que el
# resto de libs (ops.yml: `source tasks/lib/mlflow_uri.sh`). NO usar
# $(dirname "${BASH_SOURCE[0]}"): el shell embebido de go-task (mvdan/sh) no
# puebla BASH_SOURCE -> queda vacio -> dirname=. -> buscaba ./mlflow_uri.sh en el
# repo root y fallaba con "no such file or directory".
source tasks/lib/mlflow_uri.sh

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
  # Invocacion ASINCRONA (igual que EventBridge): la Lambda bloquea hasta 15 min
  # esperando el cold-start de RDS, mucho mas que el read-timeout de boto3 (~60s).
  # Si la invocaramos sincrona (default) el CLI falla con "Read timeout on
  # endpoint URL" aunque la Lambda siga corriendo bien. Aca solo disparamos el
  # start; la espera real la hace el polling de RDS+ALB de abajo.
  aws lambda invoke \
    --function-name "$scheduler_fn" \
    --invocation-type Event \
    --cli-binary-format raw-in-base64-out \
    --payload '{"action":"start"}' \
    /tmp/wake-start.out >/dev/null
  echo "    scheduler.start invocado (async, 202). Esperando readiness..."

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
