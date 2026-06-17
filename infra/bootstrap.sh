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
