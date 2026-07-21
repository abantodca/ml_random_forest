#!/usr/bin/env bash
# =============================================================================
# tasks/lib/snapshot.sh  -  Backups del RDS: crear, verificar, restaurar, podar
# =============================================================================
# Sourceado (no ejecutado) desde tasks/ops.yml, tasks/infra.yml y Taskfile.yml.
# Asume el CWD en la raiz del repo (Task siempre corre desde ahi).
#
# VOCABULARIO UNICO (mismas palabras en tasks, docs y mensajes en pantalla):
#   backup    = un snapshot MANUAL del RDS. Unica copia de MLflow tracking +
#               Model Registry + la tabla `forecasts`.
#   restaurar = crear el RDS a partir de un backup (rds_snapshot_identifier).
#   artifacts = modelos .joblib + reports HTML. Viven en S3, NO en el RDS, y no
#               participan de este ciclo: sobreviven al teardown solos.
#
# CICLO — ver docs/02-produccion-aws.md #8.5 "Ciclo backup -> restauracion".
#   apagar:   ensure_backup -> destroy (skip_final_snapshot=true)
#             -> assert_backup_exists -> prune_snapshots
#   levantar: resolve_restore_snapshot -> apply [-var rds_snapshot_identifier]
#
# POR QUE EL BACKUP VA ANTES DEL DESTROY (y no como final_snapshot de Terraform):
#   El `final_snapshot` de aws_db_instance se toma DURANTE el destroy. Si algo
#   falla ahi —el caso real es el RDS en `stopped` -> InvalidDBInstanceState— el
#   destroy aborta a la mitad y quedas con la infra rota Y sin backup. Tomandolo
#   antes, verificado y `available`, el destroy ya no puede perder datos.
#
# Solo se miran snapshots `manual`: los `automated` (backup_retention_period = 7)
# se borran junto con la instancia y no sirven como fuente de
# `aws_db_instance.snapshot_identifier`.
#
# Uso:  source tasks/lib/snapshot.sh
# =============================================================================

# ─── Consulta ────────────────────────────────────────────────────────────────

# latest_snapshot <db-instance-id>
#   Imprime en stdout el identifier del backup mas reciente (por
#   SnapshotCreateTime) que este `available`. Si no hay ninguno, imprime vacio
#   y retorna 0 (NO es un error: es el caso de un stand-up desde cero).
#   Todo el ruido va a stderr para no contaminar la sustitucion de comandos.
latest_snapshot() {
  local db_id="$1"
  local snap
  snap=$(aws rds describe-db-snapshots \
    --snapshot-type manual \
    --query "sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}' && Status=='available'], &SnapshotCreateTime)[-1].DBSnapshotIdentifier" \
    --output text 2>/dev/null || echo "")

  # La CLI devuelve el string "None" cuando el query no matchea nada.
  if [ -z "$snap" ] || [ "$snap" = "None" ]; then
    echo "  No hay backups de $db_id -> el RDS se creara VACIO." >&2
    echo ""
    return 0
  fi
  echo "  Backup mas reciente de $db_id: $snap" >&2
  echo "$snap"
}

# list_snapshots <db-instance-id>
#   Tabla legible de los backups (mas nuevo primero).
list_snapshots() {
  local db_id="$1"
  aws rds describe-db-snapshots \
    --snapshot-type manual \
    --query "reverse(sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}'], &SnapshotCreateTime))[].[DBSnapshotIdentifier,Status,SnapshotCreateTime,AllocatedStorage]" \
    --output table
}

# ─── Creacion ────────────────────────────────────────────────────────────────

# backup_now <db-instance-id> [etiqueta]
#   Crea un backup y ESPERA a que quede `available`. Imprime el identifier en
#   stdout; todo lo demas va a stderr para poder capturarlo con $(...).
#   Retorna 1 si el RDS no existe: quien llama decide si eso es fatal.
backup_now() {
  local db_id="$1"
  local label="${2:-backup}"
  local snap

  if ! aws rds describe-db-instances --db-instance-identifier "$db_id" >/dev/null 2>&1; then
    echo "  RDS $db_id no existe -> no hay nada que respaldar." >&2
    return 1
  fi

  # AWS rechaza snapshotear una instancia detenida (InvalidDBInstanceState) y
  # `ops:down` la deja parada. ensure_rds_available vive en nuke.sh; se sourcea
  # aca si el caller no lo hizo, para que backup_now sea autosuficiente.
  if ! command -v ensure_rds_available >/dev/null 2>&1; then
    # shellcheck source=tasks/lib/nuke.sh
    source tasks/lib/nuke.sh
  fi
  ensure_rds_available "$db_id" >&2

  snap="${db_id}-${label}-$(date +%Y%m%d%H%M%S)"
  echo "  Creando backup $snap..." >&2
  aws rds create-db-snapshot \
    --db-instance-identifier "$db_id" \
    --db-snapshot-identifier "$snap" >/dev/null

  echo "  Esperando a que $snap quede available (~3-8 min)..." >&2
  aws rds wait db-snapshot-available --db-snapshot-identifier "$snap"
  echo "  OK backup $snap verificado y restaurable." >&2
  echo "$snap"
}

# ensure_backup <db-instance-id> [etiqueta] [max-edad-minutos]
#   "Si no hay backup, lo hace; si ya hay uno fresco, trabaja con ese."
#   Garantiza que exista un backup `available` ANTES de una operacion
#   destructiva. Imprime en stdout el identifier del backup vigente.
#
#   max-edad-minutos (default 0 = siempre crea uno nuevo). Con un valor > 0
#   reutiliza el ultimo backup si es mas nuevo que esa edad: sirve para
#   reintentar un teardown que fallo DESPUES de haber respaldado, sin volver a
#   pagar los ~8 min de espera.
#
#   Retorna 1 si el RDS no existe (nada que respaldar): hacer teardown de una
#   infra ya destruida es legitimo, y el caller lo trata como no-fatal.
ensure_backup() {
  local db_id="$1"
  local label="${2:-backup}"
  local max_age_min="${3:-0}"
  local last created age_min

  if ! aws rds describe-db-instances --db-instance-identifier "$db_id" >/dev/null 2>&1; then
    echo "  RDS $db_id no existe -> no hay nada que respaldar (skip)." >&2
    return 1
  fi

  if [ "$max_age_min" -gt 0 ]; then
    last=$(aws rds describe-db-snapshots \
      --snapshot-type manual \
      --query "sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}' && Status=='available'], &SnapshotCreateTime)[-1].[DBSnapshotIdentifier,SnapshotCreateTime]" \
      --output text 2>/dev/null || echo "")
    if [ -n "$last" ] && [ "$last" != "None" ]; then
      created=$(echo "$last" | awk '{print $2}')
      age_min=$(( ( $(date -u +%s) - $(date -u -d "$created" +%s) ) / 60 ))
      if [ "$age_min" -le "$max_age_min" ]; then
        echo "  Ya hay un backup de hace ${age_min} min (<= ${max_age_min}) -> se reutiliza." >&2
        echo "$last" | awk '{print $1}'
        return 0
      fi
    fi
  fi

  backup_now "$db_id" "$label"
}

# ─── Verificacion ────────────────────────────────────────────────────────────

# assert_backup_exists <db-instance-id>
#   Falla ruidosamente si NO quedo ningun backup restaurable. Se corre DESPUES
#   del teardown/destroy: convierte una perdida silenciosa de datos (el bug
#   historico de este repo) en un error visible mientras todavia se puede
#   reaccionar.
assert_backup_exists() {
  local db_id="$1"
  local snap
  snap=$(aws rds describe-db-snapshots \
    --snapshot-type manual \
    --query "sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}' && Status=='available'], &SnapshotCreateTime)[-1].DBSnapshotIdentifier" \
    --output text 2>/dev/null || echo "")

  if [ -z "$snap" ] || [ "$snap" = "None" ]; then
    echo ""
    echo "ERROR No quedo NINGUN backup de $db_id."
    echo "      MLflow Registry y la tabla forecasts NO son recuperables."
    echo "      Si el RDS todavia existe: task ops:backup-now"
    return 1
  fi
  echo "  OK backup vigente: $snap (lo consumira el proximo deploy/rebuild)."
}

# ─── Restauracion ────────────────────────────────────────────────────────────

# resolve_restore_snapshot <db-instance-id> [preferencia]
#   Fuente UNICA de la decision "restaurar o arrancar limpio". La comparten
#   `task deploy` y `task rebuild` para que el resultado sea identico por
#   cualquiera de los dos caminos.
#
#   Imprime en stdout el identifier a restaurar, o vacio si corresponde crear un
#   RDS nuevo. Los mensajes van a stderr.
#
#   preferencia:
#     ""      (default) -> el backup mas reciente, si existe
#     "none"            -> forzar RDS vacio (ignora los backups)
#     "<id>"            -> ese backup exacto (se valida que exista y este available)
#
#   Precedencia deliberada: si el RDS YA existe, nunca se restaura. Restaurar es
#   una operacion de CREACION (snapshot_identifier es ForceNew); pasarlo sobre
#   una instancia viva la recrearia y destruiria los datos actuales.
resolve_restore_snapshot() {
  local db_id="$1"
  local pref="${2:-}"
  local st

  if aws rds describe-db-instances --db-instance-identifier "$db_id" >/dev/null 2>&1; then
    echo "  RDS $db_id ya existe -> no se restaura nada (apply normal)." >&2
    echo ""
    return 0
  fi

  if [ "$pref" = "none" ]; then
    echo "  SNAPSHOT=none -> RDS nuevo y VACIO (no se restaura)." >&2
    echo ""
    return 0
  fi

  if [ -n "$pref" ]; then
    st=$(aws rds describe-db-snapshots --db-snapshot-identifier "$pref" \
      --query 'DBSnapshots[0].Status' --output text 2>/dev/null || echo "")
    if [ "$st" != "available" ]; then
      echo "ERROR El backup '$pref' no existe o no esta available (estado: ${st:-inexistente})." >&2
      echo "      Ver los disponibles con: task backups" >&2
      return 1
    fi
    echo "  Backup fijado por el usuario: $pref" >&2
    echo "$pref"
    return 0
  fi

  latest_snapshot "$db_id"
}

# ─── Retencion ───────────────────────────────────────────────────────────────

# prune_snapshots <db-instance-id> <keep-n>
#   Borra los backups mas viejos, conservando los <keep-n> ultimos.
#   Idempotente. Con menos de keep-n backups no hace nada.
prune_snapshots() {
  local db_id="$1"
  local keep="${2:-6}"
  local all count victims

  all=$(aws rds describe-db-snapshots \
    --snapshot-type manual \
    --query "reverse(sort_by(DBSnapshots[?DBInstanceIdentifier=='${db_id}'], &SnapshotCreateTime))[].DBSnapshotIdentifier" \
    --output text 2>/dev/null || echo "")
  if [ -z "$all" ] || [ "$all" = "None" ]; then
    echo "  No hay backups de $db_id -> nada que podar."
    return 0
  fi

  # `tr` porque --output text separa por tabs en una sola linea.
  count=$(echo "$all" | tr '\t' '\n' | grep -c . || true)
  if [ "$count" -le "$keep" ]; then
    echo "  $count backup(s) <= retencion ($keep) -> nada que podar."
    return 0
  fi

  victims=$(echo "$all" | tr '\t' '\n' | tail -n +$((keep + 1)))
  echo "  $count backups, retencion $keep -> borrando $((count - keep)):"
  for s in $victims; do
    echo "    - $s"
    aws rds delete-db-snapshot --db-snapshot-identifier "$s" >/dev/null
  done
}
