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
