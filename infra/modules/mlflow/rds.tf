# infra/modules/mlflow/rds.tf
data "aws_region" "current" {}

# La credencial master (random_password + secret) NO vive aca a proposito:
# teardown destruye este modulo y la password debe sobrevivir para poder
# restaurar los snapshots. Ver infra/envs/prod/rds_secret.tf.

resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.project}-rds-subnets"
  subnet_ids = var.private_subnet_ids
}

resource "aws_db_instance" "mlflow" {
  identifier             = "${var.project}-mlflow"
  engine                 = "postgres"
  engine_version         = "15"
  instance_class         = var.rds_instance_class
  allocated_storage      = var.rds_allocated_storage_gb
  storage_type           = "gp3"
  storage_encrypted      = true
  db_name                = "mlflow"
  username               = "mlflow"
  password               = var.rds_password
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids = [var.sg_rds_id]
  publicly_accessible    = false
  apply_immediately      = true

  # Proteccion de datos (default true / snapshot final). Las tareas de
  # destroy/teardown levantan deletion_protection via AWS CLI antes del destroy.
  deletion_protection       = var.rds_deletion_protection
  skip_final_snapshot       = var.rds_skip_final_snapshot
  final_snapshot_identifier = var.rds_skip_final_snapshot ? null : (var.rds_final_snapshot_identifier != "" ? var.rds_final_snapshot_identifier : "${var.project}-mlflow-final")

  # Restore-from-snapshot: vacio (default) = instancia NUEVA y VACIA. Con valor,
  # la instancia se crea a partir de ese snapshot. Lo inyecta `task ops:rebuild`
  # resolviendo el snapshot final mas reciente (tasks/lib/snapshot.sh), cerrando
  # el ciclo teardown -> snapshot -> rebuild. Ver GUIA #8.6.
  #
  # Al restaurar, AWS conserva db_name/username/password DEL SNAPSHOT y los
  # argumentos de arriba se ignoran; por eso la credencial master vive en la
  # raiz (infra/envs/prod/rds_secret.tf) y sigue siendo la correcta.
  snapshot_identifier = var.rds_snapshot_identifier != "" ? var.rds_snapshot_identifier : null

  backup_retention_period = 7
  backup_window           = "06:00-07:00"
  maintenance_window      = "Mon:07:00-Mon:08:00"

  tags = { Name = "${var.project}-mlflow" }

  # OBLIGATORIO, no cosmetico: snapshot_identifier es ForceNew. Sin este
  # ignore_changes, cualquier apply posterior que no repita el mismo -var
  # (p.ej. el `terraform apply` completo que hace `ops:up` cuando el ALB no
  # existe, tasks/ops.yml) veria "" contra el valor en state y DESTRUIRIA Y
  # RECREARIA el RDS, perdiendo todo lo restaurado. El snapshot solo debe
  # influir en la creacion inicial de la instancia.
  lifecycle {
    ignore_changes = [snapshot_identifier]
  }
}
