output "data_bucket" { value = aws_s3_bucket.data.bucket }
output "data_bucket_arn" { value = aws_s3_bucket.data.arn }
output "artifacts_bucket" { value = aws_s3_bucket.artifacts.bucket }
output "artifacts_bucket_arn" { value = aws_s3_bucket.artifacts.arn }

output "ecr_trainer_url" { value = aws_ecr_repository.trainer.repository_url }
output "ecr_mlflow_url" { value = aws_ecr_repository.mlflow.repository_url }
output "ecr_reports_url" { value = aws_ecr_repository.reports.repository_url }
output "ecr_api_url" { value = aws_ecr_repository.api.repository_url }
output "ecr_ui_url" { value = aws_ecr_repository.ui.repository_url }
