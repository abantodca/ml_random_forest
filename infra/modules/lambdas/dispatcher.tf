# infra/modules/lambdas/dispatcher.tf
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Empaca el codigo Python en zip
data "archive_file" "dispatcher" {
  type        = "zip"
  source_file = "${var.lambdas_src_dir}/dispatcher.py"
  output_path = "${path.module}/dispatcher.zip"
}

# IAM (trust policy compartida en infra/modules/_shared/assume-lambda.json)
resource "aws_iam_role" "dispatcher" {
  name               = "${var.project}-dispatcher"
  assume_role_policy = file("${path.module}/../_shared/assume-lambda.json")
}

resource "aws_iam_role_policy" "dispatcher" {
  role = aws_iam_role.dispatcher.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        # TagResource: SubmitJob pasa tags={...}, lo cual requiere batch:TagResource.
        # DescribeJobs removido: el dispatcher solo hace submit_job (ver dispatcher.py).
        Action = ["batch:SubmitJob", "batch:TagResource"]
        Resource = [
          var.job_queue_spot_arn,
          var.job_queue_ondemand_arn,
          # SubmitJob por NAME autoriza contra el ARN SIN revision; SubmitJob
          # por ARN con revision matchea el patron `:*`. Hacen falta ambos.
          "arn:aws:batch:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:job-definition/${var.job_definition_name}",
          "arn:aws:batch:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:job-definition/${var.job_definition_name}:*"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "dispatcher" {
  name              = "/aws/lambda/${var.project}-dispatcher"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "dispatcher" {
  function_name    = "${var.project}-dispatcher"
  role             = aws_iam_role.dispatcher.arn
  runtime          = "python3.12"
  handler          = "dispatcher.handler"
  filename         = data.archive_file.dispatcher.output_path
  source_code_hash = data.archive_file.dispatcher.output_base64sha256
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      PROJECT = var.project
      # AWS Batch SubmitJob/ListJobs aceptan ARN o name; usamos NAME.
      # Antes el .tf construia `"${var.project}-job-queue-spot"` inline;
      # ahora se recibe como input (var.job_queue_spot_name) wireado
      # desde module.batch.job_queue_spot — single source of truth.
      JOB_QUEUE_SPOT     = var.job_queue_spot_name
      JOB_QUEUE_ONDEMAND = var.job_queue_ondemand_name
      JOB_DEFINITION     = var.job_definition_name
      DATA_BUCKET        = var.data_bucket
      VARIETIES_ALLOWED  = join(",", var.varieties_allowed)
    }
  }

  depends_on = [aws_cloudwatch_log_group.dispatcher]
}
