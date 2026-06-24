data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

locals {
  start_hour_utc = (var.work_start_hour_local - var.tz_offset_hours + 24) % 24
  stop_hour_utc  = (var.work_end_hour_local - var.tz_offset_hours + 24) % 24

  # ARNs concretos para acotar las acciones IAM destructivas (least-privilege):
  # antes el rol usaba Resource="*" -> podia parar CUALQUIER RDS / tocar CUALQUIER
  # ECS service de la cuenta. Ahora se limita a los recursos de este proyecto.
  acct    = data.aws_caller_identity.current.account_id
  region  = data.aws_region.current.region
  rds_arn = "arn:aws:rds:${local.region}:${local.acct}:db:${var.rds_instance_id}"
  ecs_svc_arns = [
    for s in [
      var.ecs_service_name_mlflow, var.ecs_service_name_reports,
      var.ecs_service_name_api, var.ecs_service_name_ui
    ] : "arn:aws:ecs:${local.region}:${local.acct}:service/${var.ecs_cluster_name}/${s}"
  ]
}

data "archive_file" "scheduler" {
  type        = "zip"
  source_file = "${var.lambdas_src_dir}/scheduler.py"
  output_path = "${path.module}/scheduler.zip"
}

# Trust policy compartida en infra/modules/_shared/assume-lambda.json
resource "aws_iam_role" "scheduler" {
  name               = "${var.project}-scheduler"
  assume_role_policy = file("${path.module}/../_shared/assume-lambda.json")
}

resource "aws_iam_role_policy" "scheduler" {
  role = aws_iam_role.scheduler.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecs:UpdateService", "ecs:DescribeServices"]
        Resource = local.ecs_svc_arns
      },
      {
        Effect   = "Allow"
        Action   = ["rds:StartDBInstance", "rds:StopDBInstance"]
        Resource = local.rds_arn
      },
      {
        # DescribeDBInstances NO soporta permisos a nivel de recurso -> requiere "*".
        Effect   = "Allow"
        Action   = ["rds:DescribeDBInstances"]
        Resource = "*"
      },
      {
        # ListJobs tampoco soporta resource-level. DescribeJobs se removio: el
        # scheduler solo llama list_jobs (ver infra/lambdas/scheduler.py).
        Effect   = "Allow"
        Action   = ["batch:ListJobs"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "scheduler" {
  name              = "/aws/lambda/${var.project}-scheduler"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "scheduler" {
  function_name    = "${var.project}-scheduler"
  role             = aws_iam_role.scheduler.arn
  runtime          = "python3.12"
  handler          = "scheduler.handler"
  filename         = data.archive_file.scheduler.output_path
  source_code_hash = data.archive_file.scheduler.output_base64sha256
  timeout          = 900 # Patch 13.3: 15 min (antes 300). Cubre RDS cold start (~5-8 min) + wait MLflow.
  memory_size      = 256

  environment {
    variables = {
      PROJECT         = var.project
      ECS_CLUSTER     = var.ecs_cluster_name
      ECS_SVC_MLFLOW  = var.ecs_service_name_mlflow
      ECS_SVC_REPORTS = var.ecs_service_name_reports
      ECS_SVC_API     = var.ecs_service_name_api
      ECS_SVC_UI      = var.ecs_service_name_ui
      RDS_INSTANCE    = var.rds_instance_id
      # Antes los names se construian inline (`"${var.project}-job-queue-spot"`);
      # ahora vienen como input wireado desde module.batch en envs/prod.
      JOB_QUEUE_SPOT     = var.job_queue_spot_name
      JOB_QUEUE_ONDEMAND = var.job_queue_ondemand_name
      # Patch 13.1: propagar workdays + ventana al _keepstop (sino el
      # martes/jueves queda "dentro de ventana" y nunca re-para el RDS).
      WORKDAYS_CRON  = var.workdays_cron
      WORK_START_UTC = tostring(local.start_hour_utc)
      WORK_END_UTC   = tostring(local.stop_hour_utc)
    }
  }

  depends_on = [aws_cloudwatch_log_group.scheduler]
}

resource "aws_cloudwatch_event_rule" "start" {
  name                = "${var.project}-start"
  description         = "L-V ${var.work_start_hour_local}:00 PET start RDS+Fargate"
  schedule_expression = "cron(0 ${local.start_hour_utc} ? * ${var.workdays_cron} *)"
}

resource "aws_cloudwatch_event_target" "start" {
  rule      = aws_cloudwatch_event_rule.start.name
  target_id = "scheduler-start"
  arn       = aws_lambda_function.scheduler.arn
  input     = jsonencode({ action = "start" })
}

# ----- EventBridge: cron STOP L-V <stop_hour_utc>:00 -----------------
resource "aws_cloudwatch_event_rule" "stop" {
  name                = "${var.project}-stop"
  description         = "L-V ${var.work_end_hour_local}:00 PET stop RDS+Fargate"
  schedule_expression = "cron(0 ${local.stop_hour_utc} ? * ${var.workdays_cron} *)"
}

resource "aws_cloudwatch_event_target" "stop" {
  rule      = aws_cloudwatch_event_rule.stop.name
  target_id = "scheduler-stop"
  arn       = aws_lambda_function.scheduler.arn
  input     = jsonencode({ action = "stop" })
}

# ----- Cron extra: cada 6h chequea RDS y lo re-stop si quedo RUNNING --
# (necesario porque RDS auto-arranca despues de 7 dias stopped)
resource "aws_cloudwatch_event_rule" "rds_keepstop" {
  name                = "${var.project}-rds-keepstop"
  description         = "Cada 6h: re-stop RDS si quedo RUNNING fuera de ventana"
  schedule_expression = "rate(6 hours)"
}

resource "aws_cloudwatch_event_target" "rds_keepstop" {
  rule      = aws_cloudwatch_event_rule.rds_keepstop.name
  target_id = "scheduler-keepstop"
  arn       = aws_lambda_function.scheduler.arn
  input     = jsonencode({ action = "keepstop" })
}

resource "aws_lambda_permission" "start" {
  statement_id  = "AllowStart"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.start.arn
}

resource "aws_lambda_permission" "stop" {
  statement_id  = "AllowStop"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.stop.arn
}

resource "aws_lambda_permission" "keepstop" {
  statement_id  = "AllowKeepstop"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.rds_keepstop.arn
}
