output "data_bucket" { value = aws_s3_bucket.data.bucket }
output "data_bucket_arn" { value = aws_s3_bucket.data.arn }
output "artifacts_bucket" { value = aws_s3_bucket.artifacts.bucket }
output "artifacts_bucket_arn" { value = aws_s3_bucket.artifacts.arn }

# Los nombres de output NO cambian (consumidores externos intactos); solo la
# referencia interna pasa a la instancia for_each `this[<key>]`.
output "ecr_trainer_url" { value = aws_ecr_repository.this["trainer"].repository_url }
output "ecr_mlflow_url" { value = aws_ecr_repository.this["mlflow"].repository_url }
output "ecr_reports_url" { value = aws_ecr_repository.this["reports"].repository_url }
output "ecr_api_url" { value = aws_ecr_repository.this["api"].repository_url }
output "ecr_ui_url" { value = aws_ecr_repository.this["ui"].repository_url }
