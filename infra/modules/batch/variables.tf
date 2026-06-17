variable "project" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "sg_batch_id" { type = string }
variable "ecr_trainer_url" { type = string }
variable "trainer_image_tag" { type = string }

variable "spot_max_vcpus" { type = number }
variable "ondemand_max_vcpus" { type = number }
variable "spot_bid_percentage" {
  type    = number
  default = 70
}
variable "instance_type" { type = string }

variable "tracking_uri" { type = string }
variable "artifacts_bucket" { type = string }
variable "artifacts_bucket_arn" { type = string }
variable "data_bucket_arn" { type = string }

variable "job_attempt_seconds" {
  type    = number
  default = 28800 # 8h hard ceiling (incluye prod_xl)
}

variable "log_retention_days" { type = number }
