variable "project" { type = string }
variable "alert_email" { type = string }
variable "batch_job_queue_names" {
  type        = map(string)
  description = "Colas Batch a vigilar: clave estatica (spot/ondemand) -> nombre real de la cola (apply-time). El mapa permite que for_each tenga claves conocidas en plan aunque los nombres no existan aun (deploy desde cero)."
}
variable "alb_arn_suffix" {
  type        = string
  description = "Suffix del ALB ARN (formato 'app/<name>/<id>'). Usado por CloudWatch metrics."
}
variable "varieties" { type = list(string) }
variable "mape_alarm_threshold" { type = number }
