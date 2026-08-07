#!/usr/bin/env bash

: "${PROJECT:?ERROR: \$PROJECT vacia. Correr 'source scripts/prod.env' primero.}"
: "${ACCOUNT_SUFFIX:?ERROR: \$ACCOUNT_SUFFIX vacia. Correr 'source scripts/prod.env' primero.}"

: "${DATA_BUCKET:=${PROJECT}-data-${ACCOUNT_SUFFIX}}"
: "${ARTIFACTS_BUCKET:=${PROJECT}-artifacts-${ACCOUNT_SUFFIX}}"
export DATA_BUCKET ARTIFACTS_BUCKET
