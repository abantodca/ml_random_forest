# infra/modules/batch/main.tf  (parte 1/4 — data + log group)
data "aws_region" "current" {}

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/${var.project}"
  retention_in_days = var.log_retention_days
}

# infra/modules/batch/main.tf  (parte 2/4 — compute environments)
resource "aws_batch_compute_environment" "spot" {
  # AWS provider v6+ usa `name`. El atributo `compute_environment_name`
  # fue deprecado en v5 y eliminado en v6 -> con `aws ~> 6.0` lockeado
  # en #3.2.1, `terraform validate` falla si se usa el nombre viejo.
  name         = "${var.project}-ce-spot"
  service_role = aws_iam_role.batch_service.arn
  type         = "MANAGED"
  state        = "ENABLED"

  # En destroy, Terraform borra los dependientes antes que sus dependencias.
  # Sin este depends_on, `aws_iam_role_policy_attachment.batch_service` (que
  # solo cuelga del role, no del CE) puede desadjuntarse ANTES de que el CE
  # termine de borrarse -> Batch pierde ecs:ListClusters, no desmonta el
  # cluster ECS y el CE queda DISABLED/INVALID, trabando el destroy entero.
  depends_on = [aws_iam_role_policy_attachment.batch_service]

  compute_resources {
    type                = "SPOT"
    bid_percentage      = var.spot_bid_percentage
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"
    min_vcpus           = 0
    max_vcpus           = var.spot_max_vcpus
    desired_vcpus       = 0
    instance_type       = [var.instance_type]
    subnets             = var.private_subnet_ids
    security_group_ids  = [var.sg_batch_id]
    instance_role       = aws_iam_instance_profile.batch.arn
    tags                = { Name = "${var.project}-batch-spot" }
  }

  lifecycle {
    create_before_destroy = true
    ignore_changes        = [compute_resources[0].desired_vcpus]
  }
}

resource "aws_batch_compute_environment" "ondemand" {
  name         = "${var.project}-ce-ondemand" # ver nota en bloque "spot" sobre v6
  service_role = aws_iam_role.batch_service.arn
  type         = "MANAGED"
  state        = "ENABLED"

  # Mismo motivo que en el CE spot: el attachment del AWSBatchServiceRole debe
  # sobrevivir hasta que el CE termine de borrarse (evita el estado INVALID).
  depends_on = [aws_iam_role_policy_attachment.batch_service]

  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    min_vcpus           = 0
    max_vcpus           = var.ondemand_max_vcpus
    desired_vcpus       = 0
    instance_type       = [var.instance_type]
    subnets             = var.private_subnet_ids
    security_group_ids  = [var.sg_batch_id]
    instance_role       = aws_iam_instance_profile.batch.arn
    tags                = { Name = "${var.project}-batch-od" }
  }

  lifecycle {
    create_before_destroy = true
    ignore_changes        = [compute_resources[0].desired_vcpus]
  }
}

# infra/modules/batch/main.tf  (parte 3/4 — job queues)
resource "aws_batch_job_queue" "spot" {
  name     = "${var.project}-job-queue-spot"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.spot.arn
  }
}

resource "aws_batch_job_queue" "ondemand" {
  name     = "${var.project}-job-queue-ondemand"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.ondemand.arn
  }
}

# infra/modules/batch/main.tf  (parte 4/4 — job definition)
resource "aws_batch_job_definition" "trainer" {
  name = "${var.project}-trainer"
  type = "container"

  retry_strategy {
    attempts = 2
    # Auto-retry solo cuando Spot interrumpe el host (preserva exit codes
    # del trainer; un error real no se reintenta)
    evaluate_on_exit {
      action           = "RETRY"
      on_status_reason = "Host EC2*"
    }
    evaluate_on_exit {
      action    = "EXIT"
      on_reason = "*"
    }
  }

  timeout {
    attempt_duration_seconds = var.job_attempt_seconds
  }

  container_properties = jsonencode({
    image            = "${var.ecr_trainer_url}:${var.trainer_image_tag}"
    vcpus            = 8     # c6i.2xlarge tiene 8 vCPU
    memory           = 14000 # de los 16 GB, dejamos ~2 GB para kernel + Batch agent
    jobRoleArn       = aws_iam_role.job.arn
    executionRoleArn = aws_iam_role.exec.arn
    # networkConfiguration (assignPublicIp) es solo Fargate; en EC2 la IP
    # publica se define en la subnet/compute environment, no aqui.
    # Sobreescrito por Lambda dispatcher (Sec 3.9.5) en cada submit.
    command = ["--varieties", "POP", "--tuning", "smoke"]
    environment = [
      { name = "MLFLOW_TRACKING_URI", value = var.tracking_uri },
      { name = "S3_ARTIFACTS_BUCKET", value = var.artifacts_bucket },
      { name = "S3_ARTIFACTS_PREFIX", value = "artifacts" },
      { name = "S3_REPORTS_PREFIX", value = "reports" },
      # S3_DATA_BUCKET / S3_DATA_KEY se inyectan por job (varia por submit)
      { name = "AWS_DEFAULT_REGION", value = data.aws_region.current.region },
      { name = "PYTHONUNBUFFERED", value = "1" }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.batch.name
        awslogs-region        = data.aws_region.current.region
        awslogs-stream-prefix = "trainer"
      }
    }
  })

  propagate_tags = true
  tags = {
    Project = var.project
  }
}
