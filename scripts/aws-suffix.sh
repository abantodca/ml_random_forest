#!/usr/bin/env bash
set -euo pipefail

acct=$(aws sts get-caller-identity --query Account --output text)
echo "${acct#?????}"
