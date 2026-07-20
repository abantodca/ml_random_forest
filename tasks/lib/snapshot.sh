#!/usr/bin/env bash
# =============================================================================
# tasks/lib/snapshot.sh  -  Snapshots RDS: resolver el ultimo y podar viejos
# =============================================================================
# Cierra el ciclo teardown -> snapshot -> rebuild:
#
#   ops:teardown  destruye module.mlflow pasando rds_final_snapshot_identifier
#                 timestamped -> queda un snapshot MANUAL en la cuenta.
#   ops:rebuild   llama a latest_snapshot() y se lo pasa a terraform como
#                 -var rds_snapshot_identifier=... -> el RDS se crea RESTAURADO.
#
# Solo mira snapshots de tipo `manual`: los `automated` (backup_retention_period
# = 7) se borran solos con la instancia y no sirven como fuente de restore
# directa en `aws_db_instance.snapshot_identifier`.
#
# Uso:  source tasks/lib/snapshot.sh
# =============================================================================

# latest_snapshot <db-instance-id>
#   Imprime en stdout el identifier del snapshot manual mas reciente (por
#   SnapshotCreateTime) que este `available`. Si no hay ninguno, imprime vacio
#   y retorna 0 (NO es un error: es el caso de un stand-up desde cero).
#   Todo el ruido va a stderr para no contaminar la sustitucion de comandos.
latest_snapshot() {
  local db_id="$1"
  local snap
  # --snapshot-type manual: excluye los automated (no restaurables por nombre
  # una vez borrada la instancia). sort_by + [-1] = el mas reciente.
  snap=$(aws rds describe-db-snapshots \
    --snapshot-type manual \
    --query "sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}' && Status=='available'], &SnapshotCreateTime)[-1].DBSnapshotIdentifier" \
    --output text 2>/dev/null || echo "")

  # La CLI devuelve el string "None" cuando el query no matchea nada.
  if [ -z "$snap" ] || [ "$snap" = "None" ]; then
    echo "  No hay snapshots manuales de $db_id -> el RDS se creara VACIO." >&2
    echo ""
    return 0
  fi
  echo "  Snapshot mas reciente de $db_id: $snap" >&2
  echo "$snap"
}

# list_snapshots <db-instance-id>
#   Tabla legible de los snapshots manuales (mas nuevo primero).
list_snapshots() {
  local db_id="$1"
  aws rds describe-db-snapshots \
    --snapshot-type manual \
    --query "reverse(sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}'], &SnapshotCreateTime))[].[DBSnapshotIdentifier,Status,SnapshotCreateTime,AllocatedStorage]" \
    --output table
}

# prune_snapshots <db-instance-id> <keep-n>
#   Borra los snapshots manuales mas viejos, conservando los <keep-n> ultimos.
#   Idempotente. Con menos de keep-n snapshots no hace nada.
prune_snapshots() {
  local db_id="$1"
  local keep="${2:-4}"
  local all count victims

  all=$(aws rds describe-db-snapshots \
    --snapshot-type manual \
    --query "reverse(sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}'], &SnapshotCreateTime))[].DBSnapshotIdentifier" \
    --output text 2>/dev/null || echo "")
  if [ -z "$all" ] || [ "$all" = "None" ]; then
    echo "  No hay snapshots manuales de $db_id -> nada que podar."
    return 0
  fi

  # `tr` porque --output text separa por tabs en una sola linea.
  count=$(echo "$all" | tr '\t' '\n' | grep -c . || true)
  if [ "$count" -le "$keep" ]; then
    echo "  $count snapshot(s) <= retencion ($keep) -> nada que podar."
    return 0
  fi

  victims=$(echo "$all" | tr '\t' '\n' | tail -n +$((keep + 1)))
  echo "  $count snapshots, retencion $keep -> borrando $((count - keep)):"
  for s in $victims; do
    echo "    - $s"
    aws rds delete-db-snapshot --db-snapshot-identifier "$s" >/dev/null
  done
}
