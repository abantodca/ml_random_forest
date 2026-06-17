#!/usr/bin/env bash
# infra/bootstrap-oidc.sh — OIDC provider de GitHub Actions. UNA VEZ por cuenta.
set -euo pipefail

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
PROVIDER="arn:aws:iam::${ACCOUNT}:oidc-provider/token.actions.githubusercontent.com"

if aws iam get-open-id-connect-provider --open-id-connect-provider-arn "$PROVIDER" >/dev/null 2>&1; then
    echo "OIDC provider ya existe: $PROVIDER"
else
    aws iam create-open-id-connect-provider \
        --url "https://token.actions.githubusercontent.com" \
        --client-id-list "sts.amazonaws.com" \
        --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1" >/dev/null
    echo "OIDC provider creado: $PROVIDER"
fi
