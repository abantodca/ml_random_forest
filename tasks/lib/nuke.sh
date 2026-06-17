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

# purge_secret <secret-name>
#   Force-delete de un Secrets Manager secret SIN ventana de recuperacion.
#   Sin esto, el nombre queda reservado 30d y el siguiente apply falla con
#   "secret already scheduled for deletion".
purge_secret() {
  local name="$1"
  if ! aws secretsmanager describe-secret --secret-id "$name" >/dev/null 2>&1; then
    echo "  secret $name no existe, skip"; return 0
  fi
  echo "  Force-delete secret $name (sin recovery window)..."
  aws secretsmanager delete-secret \
    --secret-id "$name" \
    --force-delete-without-recovery >/dev/null
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
