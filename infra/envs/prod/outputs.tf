output "alb_dns" {
  description = "DNS publico del ALB (MLflow + Reports + API + UI)."
  value       = module.mlflow.alb_dns
}

output "tracking_uri" {
  description = "URL completa para MLFLOW_TRACKING_URI."
  value       = module.mlflow.tracking_uri
}

output "ecr_trainer_url" {
  description = "URL del repo ECR del trainer (para docker push)."
  value       = module.storage.ecr_trainer_url
}

output "ecr_mlflow_url" {
  description = "URL del repo ECR del MLflow custom."
  value       = module.storage.ecr_mlflow_url
}

output "ecr_reports_url" {
  description = "URL del repo ECR del reports nginx."
  value       = module.storage.ecr_reports_url
}

output "ecr_api_url" {
  description = "URL del repo ECR de la API (FastAPI)."
  value       = module.storage.ecr_api_url
}

output "ecr_ui_url" {
  description = "URL del repo ECR de la UI (Streamlit)."
  value       = module.storage.ecr_ui_url
}

output "ui_url" {
  description = "URL publica de la UI (Streamlit) detras del ALB."
  value       = "http://${module.mlflow.alb_dns}${module.ui.app_path}"
}

output "api_docs_url" {
  description = "URL publica del Swagger de la API."
  value       = "http://${module.mlflow.alb_dns}/docs"
}

output "data_bucket" {
  value = module.storage.data_bucket
}

output "artifacts_bucket" {
  value = module.storage.artifacts_bucket
}

output "job_queue_spot" {
  value = module.batch.job_queue_spot
}

output "job_queue_ondemand" {
  value = module.batch.job_queue_ondemand
}

output "job_definition_name" {
  value = module.batch.job_definition_name
}

output "dispatcher_function_name" {
  value = module.lambdas.dispatcher_function_name
}

output "sns_topic_arn" {
  value = module.monitoring.sns_topic_arn
}

output "gha_deploy_role_arn" {
  description = "Role que asume GitHub Actions para `terraform apply`."
  value       = module.cicd.gha_deploy_role_arn
}

output "gha_train_role_arn" {
  description = "Role que asume GitHub Actions para invocar Lambda dispatcher."
  value       = module.cicd.gha_train_role_arn
}
