output "job_queue_spot" { value = aws_batch_job_queue.spot.name }
output "job_queue_spot_arn" { value = aws_batch_job_queue.spot.arn }
output "job_queue_ondemand" { value = aws_batch_job_queue.ondemand.name }
output "job_queue_ondemand_arn" { value = aws_batch_job_queue.ondemand.arn }
output "job_definition_name" { value = aws_batch_job_definition.trainer.name }
output "log_group_name" { value = aws_cloudwatch_log_group.batch.name }
