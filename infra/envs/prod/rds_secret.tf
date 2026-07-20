# infra/envs/prod/rds_secret.tf
# =============================================================================
# Credencial master del RDS — vive en la RAIZ, NO dentro de module.mlflow.
# =============================================================================
# Por que aca (ADR del ciclo teardown/rebuild):
#
#   `task ops:teardown` destruye los VOLATILE_MODULES, y module.mlflow contiene
#   el RDS -> la instancia se BORRA (con snapshot final timestamped). Si el
#   random_password y su secret vivieran dentro del modulo, se irian con el, y
#   `task ops:rebuild` generaria una password NUEVA. Al restaurar el snapshot,
#   la base recuperada conserva la password VIEJA -> MLflow y la API no
#   autentican y el backup es irrecuperable en la practica.
#
#   Al vivir en la raiz sobreviven al loop de `terraform destroy -target=$mod`
#   (que solo apunta a modulos), de modo que la credencial sigue siendo valida
#   para cualquier snapshot tomado antes del teardown.
#
# `task nuke` SI los borra a proposito (purge_secret en Taskfile.yml): esa es la
# ruta irreversible y asume que ya no queda nada que recuperar.
#
# MIGRACION desde el layout anterior (recursos dentro de module.mlflow):
#   usar `task infra:migrate-rds-secret`, que hace `terraform state mv` de los
#   tres recursos SIN recrearlos (un destroy/create aqui rotaria la password y
#   romperia el RDS vivo). Ver GUIA #8.5.
# =============================================================================

resource "random_password" "rds" {
  length  = 32
  special = false # algunos chars rompen connection strings -> evitar
}

resource "aws_secretsmanager_secret" "rds" {
  name = "${var.project}-rds-password"
}

resource "aws_secretsmanager_secret_version" "rds" {
  secret_id     = aws_secretsmanager_secret.rds.id
  secret_string = random_password.rds.result
}
