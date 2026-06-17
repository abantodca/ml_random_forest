output "service_name" {
  description = "Nombre del servicio ECS (para el scheduler on/off)."
  value       = aws_ecs_service.ui.name
}
output "target_group_arn" { value = aws_lb_target_group.ui.arn }
output "app_path" {
  description = "Sub-path publico del ALB donde vive la UI."
  value       = "/${var.base_url_path}/"
}
