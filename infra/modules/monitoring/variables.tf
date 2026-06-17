variable "project" { type = string }
variable "alert_email" { type = string }
variable "batch_job_queue_name" { type = string }
variable "alb_arn_suffix" {
  type        = string
  description = "Suffix del ALB ARN (formato 'app/<name>/<id>'). Usado por CloudWatch metrics."
}
variable "varieties" { type = list(string) }
variable "mape_alarm_threshold" { type = number }
