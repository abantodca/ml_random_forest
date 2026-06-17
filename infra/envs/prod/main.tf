# OIDC provider de GitHub (creado en Parte 2.5, NO creado por Terraform).
# Si saltaste #2.5, este `data` falla con "no resource found" en plan.
# Pre-check antes de `terraform plan`:
#   aws iam list-open-id-connect-providers --query 'OpenIDConnectProviderList[?contains(Arn,`token.actions.githubusercontent.com`)]'
# Si devuelve [], correr `bash infra/bootstrap-oidc.sh` (#2.5).
data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

# -------------------------------------------------------------------------
# Capa 1: Red (VPC + subnets + NAT + SGs)
# -------------------------------------------------------------------------
module "network" {
  source   = "../../modules/network"
  project  = var.project
  vpc_cidr = var.vpc_cidr
}

# -------------------------------------------------------------------------
# Capa 2: Storage (S3 buckets + ECR repos)
# -------------------------------------------------------------------------
module "storage" {
  source  = "../../modules/storage"
  project = var.project
}

# -------------------------------------------------------------------------
# Capa 3: MLflow (RDS + ECS Fargate + ALB)
# -------------------------------------------------------------------------
module "mlflow" {
  source = "../../modules/mlflow"

  project              = var.project
  vpc_id               = module.network.vpc_id
  public_subnet_ids    = module.network.public_subnet_ids
  private_subnet_ids   = module.network.private_subnet_ids
  sg_alb_id            = module.network.sg_alb_id
  sg_mlflow_id         = module.network.sg_mlflow_id
  sg_rds_id            = module.network.sg_rds_id
  rds_instance_class   = var.rds_instance_class
  mlflow_image         = "${module.storage.ecr_mlflow_url}:${var.mlflow_image_tag}"
  artifacts_bucket     = module.storage.artifacts_bucket
  artifacts_bucket_arn = module.storage.artifacts_bucket_arn
  log_retention_days   = var.log_retention_days
}

# -------------------------------------------------------------------------
# Capa 4: Reports (Fargate nginx, mismo cluster + ALB que MLflow)
# -------------------------------------------------------------------------
module "reports" {
  source = "../../modules/reports"

  project              = var.project
  vpc_id               = module.network.vpc_id
  private_subnet_ids   = module.network.private_subnet_ids
  sg_mlflow_id         = module.network.sg_mlflow_id
  ecs_cluster_id       = module.mlflow.cluster_id
  alb_listener_arn     = module.mlflow.alb_listener_arn
  artifacts_bucket     = module.storage.artifacts_bucket
  artifacts_bucket_arn = module.storage.artifacts_bucket_arn
  reports_image        = "${module.storage.ecr_reports_url}:${var.reports_image_tag}"
  log_retention_days   = var.log_retention_days
}

# -------------------------------------------------------------------------
# Capa 5: Batch (Spot + OD queues, job-def, IAM)
# -------------------------------------------------------------------------
module "batch" {
  source = "../../modules/batch"

  project              = var.project
  private_subnet_ids   = module.network.private_subnet_ids
  sg_batch_id          = module.network.sg_batch_id
  ecr_trainer_url      = module.storage.ecr_trainer_url
  trainer_image_tag    = var.trainer_image_tag
  spot_max_vcpus       = var.spot_max_vcpus
  ondemand_max_vcpus   = var.ondemand_max_vcpus
  instance_type        = var.batch_instance_type
  tracking_uri         = module.mlflow.tracking_uri
  artifacts_bucket     = module.storage.artifacts_bucket
  artifacts_bucket_arn = module.storage.artifacts_bucket_arn
  data_bucket_arn      = module.storage.data_bucket_arn
  log_retention_days   = var.log_retention_days
}

# -------------------------------------------------------------------------
# Capa 6: Monitoring (SNS + alarmas CloudWatch)
# -------------------------------------------------------------------------
module "monitoring" {
  source = "../../modules/monitoring"

  project              = var.project
  alert_email          = var.alert_email
  batch_job_queue_name = module.batch.job_queue_spot
  alb_arn_suffix       = module.mlflow.alb_arn_suffix
  varieties            = var.varieties_allowed
  mape_alarm_threshold = var.mape_alarm_threshold
}

# -------------------------------------------------------------------------
# Capa 7: Lambdas (dispatcher + notifier)
# -------------------------------------------------------------------------
module "lambdas" {
  source = "../../modules/lambdas"

  project                 = var.project
  job_queue_spot_arn      = module.batch.job_queue_spot_arn
  job_queue_ondemand_arn  = module.batch.job_queue_ondemand_arn
  job_queue_spot_name     = module.batch.job_queue_spot
  job_queue_ondemand_name = module.batch.job_queue_ondemand
  job_definition_name     = module.batch.job_definition_name
  data_bucket             = module.storage.data_bucket
  varieties_allowed       = var.varieties_allowed
  sns_topic_arn           = module.monitoring.sns_topic_arn
  batch_log_group_name    = module.batch.log_group_name
  log_retention_days      = var.log_retention_days
  lambdas_src_dir         = "${path.module}/../../lambdas"
}

# -------------------------------------------------------------------------
# Capa 8: Scheduler (auto on/off RDS + Fargate)
# -------------------------------------------------------------------------
module "scheduler" {
  source = "../../modules/scheduler"

  project                  = var.project
  ecs_cluster_name         = module.mlflow.cluster_name
  ecs_service_name_mlflow  = module.mlflow.service_name
  ecs_service_name_reports = module.reports.service_name
  ecs_service_name_api     = module.api.service_name
  ecs_service_name_ui      = module.ui.service_name
  rds_instance_id          = module.mlflow.rds_instance_id
  job_queue_spot_name      = module.batch.job_queue_spot
  job_queue_ondemand_name  = module.batch.job_queue_ondemand
  work_start_hour_local    = var.work_start_hour_local
  work_end_hour_local      = var.work_end_hour_local
  log_retention_days       = var.log_retention_days
  lambdas_src_dir          = "${path.module}/../../lambdas"
}

# -------------------------------------------------------------------------
# Capa 9: CI/CD (GHA IAM roles confiando en OIDC)
# -------------------------------------------------------------------------
module "cicd" {
  source = "../../modules/cicd"

  project           = var.project
  github_org        = var.github_org
  github_repo       = var.github_repo
  oidc_provider_arn = data.aws_iam_openid_connect_provider.github.arn
}

# -------------------------------------------------------------------------
# Capa 4.5: App stack — API (FastAPI) + UI (Streamlit)
# Mismo cluster + ALB que MLflow. La API reusa el RDS de MLflow (base
# `forecasts`, auto-creada al boot) y carga modelos rnd-forest-* via S3.
# -------------------------------------------------------------------------
module "api" {
  source = "../../modules/api"

  project                        = var.project
  vpc_id                         = module.network.vpc_id
  private_subnet_ids             = module.network.private_subnet_ids
  sg_api_id                      = module.network.sg_api_id
  ecs_cluster_id                 = module.mlflow.cluster_id
  alb_listener_arn               = module.mlflow.alb_listener_arn
  service_discovery_namespace_id = module.mlflow.service_discovery_namespace_id
  api_image                      = "${module.storage.ecr_api_url}:${var.api_image_tag}"
  # MLflow interno via service discovery (no pasa por el ALB).
  mlflow_tracking_uri     = "http://mlflow.${var.project}.local:5000"
  model_registry_prefix   = var.model_registry_prefix
  mlflow_preload_models   = var.api_preload_models
  rds_address             = module.mlflow.rds_address
  rds_password_secret_arn = module.mlflow.rds_password_secret_arn
  artifacts_bucket        = module.storage.artifacts_bucket
  artifacts_bucket_arn    = module.storage.artifacts_bucket_arn
  cors_origins            = "http://${module.mlflow.alb_dns}"
  cpu                     = var.api_cpu
  memory                  = var.api_memory
  log_retention_days      = var.log_retention_days
}

module "ui" {
  source = "../../modules/ui"

  project            = var.project
  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  sg_ui_id           = module.network.sg_ui_id
  ecs_cluster_id     = module.mlflow.cluster_id
  alb_listener_arn   = module.mlflow.alb_listener_arn
  ui_image           = "${module.storage.ecr_ui_url}:${var.ui_image_tag}"
  api_internal_url   = module.api.internal_url
  cpu                = var.ui_cpu
  memory             = var.ui_memory
  log_retention_days = var.log_retention_days
}
