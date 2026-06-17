variable "project" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "sg_ui_id" { type = string }

variable "ecs_cluster_id" { type = string }
variable "alb_listener_arn" { type = string }

variable "ui_image" {
  description = "URI completa de la imagen ECR de la UI (repo:tag)."
  type        = string
}

variable "api_internal_url" {
  description = "URL interna de la API (service discovery) que consume la UI."
  type        = string
}

variable "base_url_path" {
  description = "Sub-path del ALB donde sirve Streamlit (STREAMLIT_SERVER_BASE_URL_PATH)."
  type        = string
  default     = "app"
}

# --- Capacidad / costo: Streamlit es liviano ---
variable "cpu" {
  type    = number
  default = 512 # 0.5 vCPU
}
variable "memory" {
  type    = number
  default = 1024 # 1 GB
}
variable "desired_count" {
  type    = number
  default = 1
}

variable "log_retention_days" { type = number }
