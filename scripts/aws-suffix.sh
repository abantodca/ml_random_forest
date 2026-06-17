#!/usr/bin/env bash
# Imprime los ultimos 7 digitos del AWS Account ID (sufijo de bucket).
# Consumido por tasks/local.yml `vars.SUFFIX.sh`.
#
# `${acct#?????}` (POSIX) quita los primeros 5 chars del Account ID de 12.
# Evita `tail -c 7` que en Windows-coreutils (scoop) parsea el `7` como filename.
set -euo pipefail

acct=$(aws sts get-caller-identity --query Account --output text)
echo "${acct#?????}"
