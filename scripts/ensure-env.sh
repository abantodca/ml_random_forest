#!/usr/bin/env bash
# Guard + derivacion de nombres compuestos para apuntar al stack prod.
# Uso: `source scripts/ensure-env.sh` antes de cualquier `aws ... | terraform ...`
# que componga nombres de bucket / role / ARN con $PROJECT y $ACCOUNT_SUFFIX.
#
# Aborta si las vars base estan vacias (tipico: terminal nueva sin
# `source scripts/prod.env`): sin esto los exports posteriores producen
# strings tipo "-data-" (sin prefijo ni suffix) y el comando aws falla con
# `Invalid bucket name ""` / `argument --bucket: expected one argument`.
#
# Ademas DERIVA los nombres de bucket canonicos para que ningun snippet del
# runbook tenga que recomponerlos a mano (origen del bug Parte 4.3: el snippet
# usaba $DATA_BUCKET, que nunca se exportaba).

: "${PROJECT:?ERROR: \$PROJECT vacia. Correr 'source scripts/prod.env' primero.}"
: "${ACCOUNT_SUFFIX:?ERROR: \$ACCOUNT_SUFFIX vacia. Correr 'source scripts/prod.env' primero.}"

# Nombres canonicos de bucket — fuente unica de verdad.
# `:=` respeta un override explicito en el shell y es idempotente al re-sourcear.
: "${DATA_BUCKET:=${PROJECT}-data-${ACCOUNT_SUFFIX}}"
: "${ARTIFACTS_BUCKET:=${PROJECT}-artifacts-${ACCOUNT_SUFFIX}}"
export DATA_BUCKET ARTIFACTS_BUCKET
