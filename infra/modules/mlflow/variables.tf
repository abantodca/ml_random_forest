variable "project" { type = string }
variable "vpc_id" { type = string }
variable "public_subnet_ids" { type = list(string) }
variable "private_subnet_ids" { type = list(string) }
variable "sg_alb_id" { type = string }
variable "sg_mlflow_id" { type = string }
variable "sg_rds_id" { type = string }
variable "rds_instance_class" { type = string }
variable "rds_allocated_storage_gb" {
  type    = number
  default = 20
}
variable "mlflow_image" { type = string }
variable "artifacts_bucket" { type = string }
variable "artifacts_bucket_arn" { type = string }
variable "log_retention_days" { type = number }

# ── Proteccion del RDS (datos de MLflow + forecasts) ─────────────────────────
# Defaults protectivos para prod. Las tareas de destroy/teardown levantan la
# proteccion via AWS CLI antes del terraform destroy (ver tasks/infra.yml,
# tasks/ops.yml), asi el flujo de fresh-start sigue funcionando.
variable "rds_deletion_protection" {
  type    = bool
  default = true
}
variable "rds_skip_final_snapshot" {
  type    = bool
  default = false
}
variable "rds_final_snapshot_identifier" {
  description = "Nombre del snapshot final. Las tareas de destroy pasan uno timestamped para evitar colisiones en destroys repetidos. Vacio = nombre estatico <project>-mlflow-final."
  type        = string
  default     = ""
}
