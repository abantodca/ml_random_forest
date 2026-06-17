variable "project" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "sg_api_id" { type = string }

variable "ecs_cluster_id" { type = string }
variable "alb_listener_arn" { type = string }
variable "service_discovery_namespace_id" { type = string }

variable "api_image" {
  description = "URI completa de la imagen ECR de la API (repo:tag)."
  type        = string
}

# --- MLflow / modelos ---
variable "mlflow_tracking_uri" {
  description = "URI interna del MLflow (service discovery): http://mlflow.<project>.local:5000."
  type        = string
}
variable "model_registry_prefix" {
  description = "Prefijo del registered model. Debe coincidir con el trainer."
  type        = string
  default     = "rnd-forest-"
}
variable "mlflow_preload_models" {
  description = "Precargar modelos al boot (true) o lazy (false). Lazy acota RAM."
  type        = bool
  default     = false
}

# --- Base de datos (reusa el RDS de MLflow, base `forecasts`) ---
variable "rds_address" { type = string }
variable "rds_password_secret_arn" { type = string }

# --- S3 artifacts (boto3 vía cliente MLflow) ---
variable "artifacts_bucket" { type = string }
variable "artifacts_bucket_arn" { type = string }

# --- CORS (origen del ALB para Swagger / llamadas browser) ---
variable "cors_origins" {
  description = "Origenes CORS permitidos (coma-separados)."
  type        = string
  default     = "http://localhost:8501"
}

# --- Capacidad / costo (ver analisis en GUIA) ---
# Combos validos Fargate: 1 vCPU admite 2-8 GB. La API carga modelos en RAM;
# con lazy-load (preload=false) y ~6 variedades, 1 vCPU / 2 GB sobra. Subir
# memory a 4096 si se activa preload de muchas variedades.
variable "cpu" {
  type    = number
  default = 1024 # 1 vCPU
}
variable "memory" {
  type    = number
  default = 2048 # 2 GB
}
variable "desired_count" {
  description = "Replicas. El scheduler lo maneja (0 fuera de horario)."
  type        = number
  default     = 1
}

variable "log_retention_days" { type = number }
