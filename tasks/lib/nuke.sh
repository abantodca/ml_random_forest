# Helpers para destroy/nuke: vaciar buckets versionados, borrar repos ECR,
# borrar el OIDC provider. Sourceados, no ejecutados.

# empty_bucket <bucket> [delete] [prefix]
#   Vacia versiones + delete markers. Si delete=true, ademas borra el bucket.
#   Con prefix, acota el borrado a ese prefijo (y entonces delete=true no aplica:
#   un bucket con otros prefijos vivos no se puede borrar). Lo usa
#   `infra:reset-state` para llevarse solo envs/prod/ del bucket de tfstate,
#   que antes reimplementaba este mismo patron version-aware por su cuenta.
empty_bucket() {
  local bucket="$1" delete="${2:-false}" prefix="${3:-}"
  # return 0 (no 1) a proposito: los callers corren con `set -e` y un bucket ya
  # inexistente es exito, no fallo.
  if ! aws s3api head-bucket --bucket "$bucket" 2>/dev/null; then
    echo "  $bucket no existe, skip"; return 0
  fi
  echo "  Vaciando s3://$bucket/${prefix} (versiones + delete markers)..."
  aws s3api delete-objects --bucket "$bucket" \
    --delete "$(aws s3api list-object-versions --bucket "$bucket" ${prefix:+--prefix "$prefix"} \
      --query '{Objects: [Versions[].{Key:Key,VersionId:VersionId},DeleteMarkers[].{Key:Key,VersionId:VersionId}][]}' \
      --max-items 1000)" 2>/dev/null || echo "  (ya vacio, nada que borrar)"
  if [ "$delete" = "true" ]; then
    if [ -n "$prefix" ]; then
      echo "  (prefix + delete=true: no se borra el bucket, solo el prefijo)"
      return 0
    fi
    echo "  Borrando bucket $bucket..."
    aws s3 rb "s3://$bucket"
  fi
}

# lift_rds_protection <db-instance-id>
#   Quita deletion_protection del RDS para permitir un terraform destroy.
#   Idempotente: si el RDS no existe, no hace nada. AWS aplica el cambio de
#   deletion_protection al instante (no requiere instancia available ni reboot).
lift_rds_protection() {
  local id="$1"
  if ! aws rds describe-db-instances --db-instance-identifier "$id" >/dev/null 2>&1; then
    echo "  RDS $id no existe, skip lift"; return 0
  fi
  echo "  Levantando deletion_protection de $id (para permitir destroy)..."
  aws rds modify-db-instance --db-instance-identifier "$id" \
    --no-deletion-protection --apply-immediately >/dev/null
}

# ensure_rds_available <db-instance-id>
#   Deja el RDS en estado `available`, arrancandolo si hace falta.
#
#   OBLIGATORIO antes de cualquier `terraform destroy` que tome snapshot final
#   (skip_final_snapshot=false, el default). AWS RECHAZA snapshotear una
#   instancia detenida:
#       InvalidDBInstanceState: Cannot create a snapshot because the database
#       instance <id> is not currently in the available state.
#
#   El choque es real y no teorico: tanto `ops:teardown` como `infra:destroy`
#   invocan antes a `ops:down`, que llama al scheduler con action=stop y PARA el
#   RDS. Sin este helper el destroy aborta a la mitad, dejando la infra
#   parcialmente destruida y SIN el snapshot final -> el ciclo
#   teardown -> snapshot -> rebuild se rompe justo donde importa.
#
#   Idempotente: si no existe, o ya esta available, no hace nada.
ensure_rds_available() {
  local id="$1" st
  if ! aws rds describe-db-instances --db-instance-identifier "$id" >/dev/null 2>&1; then
    echo "  RDS $id no existe, skip"; return 0
  fi
  st=$(aws rds describe-db-instances --db-instance-identifier "$id" \
    --query 'DBInstances[0].DBInstanceStatus' --output text 2>/dev/null)
  case "$st" in
    available) echo "  RDS $id ya esta available."; return 0 ;;
    stopped)
      echo "  RDS $id esta stopped -> arrancando (necesario para el snapshot final)..."
      aws rds start-db-instance --db-instance-identifier "$id" >/dev/null
      ;;
    starting|stopping|modifying|backing-up|configuring-enhanced-monitoring)
      echo "  RDS $id en estado transitorio ($st) -> esperando..."
      # `stopping` no se puede interrumpir: hay que dejar que llegue a stopped
      # y recien ahi arrancarlo.
      if [ "$st" = "stopping" ]; then
        until [ "$(aws rds describe-db-instances --db-instance-identifier "$id" \
              --query 'DBInstances[0].DBInstanceStatus' --output text 2>/dev/null)" = "stopped" ]; do
          sleep 20
        done
        echo "  RDS $id ya stopped -> arrancando..."
        aws rds start-db-instance --db-instance-identifier "$id" >/dev/null
      fi
      ;;
    *) echo "  RDS $id en estado '$st' -> intentando esperar a available..." ;;
  esac
  echo "  Esperando a que $id quede available (puede tardar ~5-10 min)..."
  # `wait db-instance-available` hace hasta 60 intentos cada 30s (30 min).
  aws rds wait db-instance-available --db-instance-identifier "$id"
  echo "  OK RDS $id available."
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
