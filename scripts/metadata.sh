#!/usr/bin/env bash
# Metadata del run de training. Cada subcomando imprime un valor en stdout
# y lo consume Taskfile.yml `train.vars` via `sh:`.
#
# Sin `set -e`: los comandos usan `||` para fallback ("unknown", "missing").

case "${1:-}" in
  git-sha)
    git rev-parse HEAD 2>/dev/null || echo unknown
    ;;
  git-dirty)
    git diff --quiet HEAD 2>/dev/null && echo false || echo true
    ;;
  data-sha)
    f=data/training/DB-HISTORICA.xlsx
    [ -f "$f" ] && sha256sum "$f" | cut -d' ' -f1 | cut -c1-12 || echo missing
    ;;
  *)
    echo "Usage: $0 {git-sha|git-dirty|data-sha}" >&2
    exit 1
    ;;
esac
