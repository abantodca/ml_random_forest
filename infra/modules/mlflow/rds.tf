# infra/modules/mlflow/rds.tf
data "aws_region" "current" {}

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

resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.project}-rds-subnets"
  subnet_ids = var.private_subnet_ids
}

resource "aws_db_instance" "mlflow" {
  identifier              = "${var.project}-mlflow"
  engine                  = "postgres"
  engine_version          = "15"
  instance_class          = var.rds_instance_class
  allocated_storage       = var.rds_allocated_storage_gb
  storage_type            = "gp3"
  storage_encrypted       = true
  db_name                 = "mlflow"
  username                = "mlflow"
  password                = random_password.rds.result
  db_subnet_group_name    = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids  = [var.sg_rds_id]
  publicly_accessible     = false
  skip_final_snapshot     = true # OK para prod single-AZ; ajustar en Parte 10.4
  apply_immediately       = true
  deletion_protection     = false # cambiar a true cuando hay datos productivos
  backup_retention_period = 7
  backup_window           = "06:00-07:00"
  maintenance_window      = "Mon:07:00-Mon:08:00"

  tags = { Name = "${var.project}-mlflow" }
}
