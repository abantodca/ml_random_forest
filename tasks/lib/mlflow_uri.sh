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
