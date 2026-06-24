variable "project" {
  description = "Slug del proyecto (prefijo de todos los recursos)."
  type        = string
  default     = "ml-training"
}

variable "region" {
  description = "Region AWS para todo el deployment."
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR de la VPC. /16 da espacio para 65k IPs."
  type        = string
  default     = "10.20.0.0/16"
}

variable "enable_nat" {
  description = "true (default): NAT gateway + EIP activos. `task teardown` lo pone en false para liberar el NAT (~$33/mes) cuando el stack queda idle; deploy/rebuild lo recrean."
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "Email que recibe notificaciones SNS (job FAILED, MAPE high)."
  type        = string
}

variable "enable_cicd" {
  description = "Crea la capa CI/CD (roles IAM GHA + data source OIDC). DEFAULT false: el stand-up completo (storage->...->api/ui) NO necesita CI/CD ni el bootstrap OIDC. Poner true DESPUES de `task infra:bootstrap-oidc` para sumar CI/CD sin tocar el resto. Ver main.tf #Capa 9."
  type        = bool
  default     = false
}

variable "github_org" {
  description = "Organizacion / usuario GitHub que aloja el repo (para OIDC trust). Solo se usa si enable_cicd=true."
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "Nombre del repo (sin la org). Para trust policy OIDC. Solo se usa si enable_cicd=true."
  type        = string
  default     = ""
}

variable "varieties_allowed" {
  description = "Allow-list defensivo para el Lambda dispatcher (rechaza submits con variety no listada). NO define las variedades del modelo: la verdad esta en las hojas del Excel (data/BD_HISTORICO_ACUMULADO.xlsx) y se descubre dinamicamente con src/step_01_load/data_loader.py::list_varieties(). Esta lista solo previene typos en `aws lambda invoke`."
  type        = list(string)
  default     = ["POP", "JUPITER", "VENTURA", "SEKOYA", "ALLISON", "STELLA"]
}

variable "spot_max_vcpus" {
  description = "Maximo de vCPUs simultaneas en la queue Spot."
  type        = number
  default     = 16
}

variable "ondemand_max_vcpus" {
  description = "Maximo de vCPUs simultaneas en la queue On-Demand (solo prod_xl)."
  type        = number
  default     = 16
}

variable "batch_instance_type" {
  description = "Tipo de instancia EC2 que arranca Batch."
  type        = string
  default     = "c6i.2xlarge"
}

variable "rds_instance_class" {
  description = "Clase RDS. Hostea DOS bases: MLflow backend + `forecasts` de la API (Capa 4.5). db.t4g.small da holgura de RAM/conexiones para el stack completo; db.t4g.micro alcanza a muy bajo trafico."
  type        = string
  default     = "db.t4g.small"
}

# ── Proteccion del RDS ───────────────────────────────────────────────────────
# Defaults protectivos. Las tareas destroy/teardown las sobreescriben con -var
# (y levantan deletion_protection via AWS CLI) para permitir el fresh-start.
variable "rds_deletion_protection" {
  description = "true (default): el RDS no se puede borrar accidentalmente. Las tareas de teardown lo levantan automaticamente."
  type        = bool
  default     = true
}
variable "rds_skip_final_snapshot" {
  description = "false (default): cada destroy del RDS toma un snapshot final. Las tareas de teardown pasan un nombre timestamped."
  type        = bool
  default     = false
}
variable "rds_final_snapshot_identifier" {
  description = "Nombre del snapshot final (lo inyectan las tareas de destroy, timestamped). Vacio = <project>-mlflow-final."
  type        = string
  default     = ""
}

variable "mlflow_image_tag" {
  description = "Tag de la imagen MLflow en ECR (build manual una vez)."
  type        = string
  default     = "v3.12.0"
}

variable "reports_image_tag" {
  description = "Tag de la imagen reports (nginx + s3-sync) en ECR."
  type        = string
  default     = "stable"
}

variable "trainer_image_tag" {
  description = "Tag de la imagen del trainer. CI/CD lo sobreescribe por commit SHA."
  type        = string
  default     = "latest"
}

variable "mape_alarm_threshold" {
  description = "Umbral de MAPE (%) para disparar alarma CloudWatch."
  type        = number
  default     = 25
}

variable "log_retention_days" {
  description = "Dias que CloudWatch retiene logs."
  type        = number
  default     = 14
}

variable "work_start_hour_local" {
  description = "Hora local de arranque del scheduler (PET, UTC-5)."
  type        = number
  default     = 8
}

variable "work_end_hour_local" {
  description = "Hora local de apagado del scheduler."
  type        = number
  default     = 12
}

# ── App stack: API (FastAPI) + UI (Streamlit) ──────────────────────────────
variable "api_image_tag" {
  description = "Tag de la imagen de la API en ECR. CI/CD lo sobreescribe por commit SHA."
  type        = string
  default     = "latest"
}

variable "ui_image_tag" {
  description = "Tag de la imagen de la UI en ECR. CI/CD lo sobreescribe por commit SHA."
  type        = string
  default     = "latest"
}

variable "model_registry_prefix" {
  description = "Prefijo del registered model en MLflow. Debe coincidir con MODEL_REGISTRY_PREFIX del trainer (src/config.py)."
  type        = string
  default     = "rnd-forest-"
}

variable "api_preload_models" {
  description = "Precargar TODOS los modelos al boot de la API. false = lazy (recomendado en prod: arranque rapido y menos RAM)."
  type        = bool
  default     = false
}

# --- Capacidad: dimensionar segun necesidad (ver analisis de costo en GUIA) ---
# Combos Fargate validos: 1 vCPU (1024) admite 2-8 GB; 2 vCPU (2048) admite 4-16 GB.
# Default API 1 vCPU / 2 GB cubre lazy-load de ~6 variedades. Subir memory a 4096
# si se activa api_preload_models con muchas variedades, o cpu a 2048 para mas
# concurrencia. Cambiar aqui en tfvars NO requiere tocar codigo.
variable "api_cpu" {
  type    = number
  default = 1024
}
variable "api_memory" {
  type    = number
  default = 2048
}
variable "ui_cpu" {
  type    = number
  default = 512
}
variable "ui_memory" {
  type    = number
  default = 1024
}
