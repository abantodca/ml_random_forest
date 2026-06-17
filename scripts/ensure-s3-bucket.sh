#!/usr/bin/env bash
# Crea bucket S3 si no existe + aplica hardening (versioning, AES256, no public).
# Idempotente en dos niveles:
#   1) Si el bucket no existe, lo crea.
#   2) Aplica versioning + encryption + public-access-block SIEMPRE (no solo al
#      crear). Los tres son PUT idempotentes; re-aplicarlos auto-corrige drift
#      si alguien tocó el bucket a mano sin esos settings.
#
# Uso: ensure-s3-bucket.sh <name> <region>
# Consumido por: tasks/local.yml `_ensure-bucket` y el bootstrap del tfstate
# documentado en GUIA_MLOPS_AWS_V2.md (no hay carpeta `infra/` en este repo).
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
