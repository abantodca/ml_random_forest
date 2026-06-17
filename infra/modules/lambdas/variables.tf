variable "project" { type = string }
variable "job_queue_spot_arn" { type = string }
variable "job_queue_ondemand_arn" { type = string }
# *_name vars: el dispatcher.py / notifier.py usan los NAMES (no ARNs)
# para `batch.submit_job` / `batch.describe_jobs`. Antes el .tf construia
# `"${var.project}-job-queue-spot"` inline; ahora se reciben como input
# del envs/prod (wireado desde module.batch.job_queue_spot/ondemand).
variable "job_queue_spot_name" { type = string }
variable "job_queue_ondemand_name" { type = string }
variable "job_definition_name" { type = string }
variable "data_bucket" { type = string }
variable "varieties_allowed" { type = list(string) }
variable "sns_topic_arn" { type = string }
variable "batch_log_group_name" { type = string }
variable "log_retention_days" { type = number }
variable "lambdas_src_dir" { type = string }
