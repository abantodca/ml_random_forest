output "service_name" {
  description = "Nombre del servicio ECS (para el scheduler on/off)."
  value       = aws_ecs_service.api.name
}
output "target_group_arn" { value = aws_lb_target_group.api.arn }
output "internal_url" {
  description = "URL interna (service discovery) que usa la UI."
  value       = "http://api.${var.project}.local:8000"
}
