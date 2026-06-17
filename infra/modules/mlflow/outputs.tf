output "tracking_uri" { value = "http://${aws_lb.main.dns_name}" }
output "alb_dns" { value = aws_lb.main.dns_name }
output "alb_arn_suffix" { value = aws_lb.main.arn_suffix } # para CloudWatch dimensions
output "alb_listener_arn" { value = aws_lb_listener.http.arn }
output "cluster_id" { value = aws_ecs_cluster.main.id }
output "cluster_name" { value = aws_ecs_cluster.main.name }
output "service_name" { value = aws_ecs_service.mlflow.name }
output "rds_instance_id" { value = aws_db_instance.mlflow.id }

# --- Wiring para los modulos api / ui (Capa 4.5) ---
output "service_discovery_namespace_id" {
  description = "ID del namespace privado <project>.local para registrar la API."
  value       = aws_service_discovery_private_dns_namespace.main.id
}
output "rds_address" {
  description = "Host del RDS (la API monta su DATABASE_URL hacia la base forecasts)."
  value       = aws_db_instance.mlflow.address
}
output "rds_password_secret_arn" {
  description = "ARN del secret con el password del RDS (lo inyecta la API)."
  value       = aws_secretsmanager_secret.rds.arn
}
