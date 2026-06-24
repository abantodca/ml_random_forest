# infra/modules/lambdas/notifier.tf
data "archive_file" "notifier" {
  type        = "zip"
  source_file = "${var.lambdas_src_dir}/notifier.py"
  output_path = "${path.module}/notifier.zip"
}

resource "aws_iam_role" "notifier" {
  name               = "${var.project}-notifier"
  assume_role_policy = file("${path.module}/../_shared/assume-lambda.json")
}

resource "aws_iam_role_policy" "notifier" {
  role = aws_iam_role.notifier.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = var.sns_topic_arn
      },
      # (batch:DescribeJobs removido: el notifier lee todo del evento EventBridge,
      #  nunca llama DescribeJobs — ver infra/lambdas/notifier.py)
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_cloudwatch_log_group" "notifier" {
  name              = "/aws/lambda/${var.project}-notifier"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "notifier" {
  function_name    = "${var.project}-notifier"
  role             = aws_iam_role.notifier.arn
  runtime          = "python3.12"
  handler          = "notifier.handler"
  filename         = data.archive_file.notifier.output_path
  source_code_hash = data.archive_file.notifier.output_base64sha256
  timeout          = 30
  memory_size      = 128

  environment {
    variables = {
      SNS_TOPIC_ARN   = var.sns_topic_arn
      BATCH_LOG_GROUP = var.batch_log_group_name
    }
  }

  depends_on = [aws_cloudwatch_log_group.notifier]
}

# EventBridge rule: Batch Job State Change FAILED/SUCCEEDED -> notifier.
# Captura ambos estados terminales: FAILED para alertar errores y SUCCEEDED
# para avisar que un entrenamiento (ej. POP) termino OK. El notifier.py
# distingue el estado via detail.status y arma el asunto/cuerpo segun el caso.
resource "aws_cloudwatch_event_rule" "batch_terminal" {
  name        = "${var.project}-batch-terminal"
  description = "Captura Batch jobs en estado FAILED o SUCCEEDED"
  event_pattern = jsonencode({
    source        = ["aws.batch"]
    "detail-type" = ["Batch Job State Change"]
    detail = {
      status = ["FAILED", "SUCCEEDED"]
    }
  })
}

resource "aws_cloudwatch_event_target" "notifier" {
  rule      = aws_cloudwatch_event_rule.batch_terminal.name
  target_id = "notifier"
  arn       = aws_lambda_function.notifier.arn
}

resource "aws_lambda_permission" "notifier_eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.notifier.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.batch_terminal.arn
}
