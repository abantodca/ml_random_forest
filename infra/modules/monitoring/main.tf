# ----- SNS topic + suscripcion email ----------------------------------
resource "aws_sns_topic" "alerts" {
  name = "${var.project}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ----- Alarma 1: Batch job FAILED -------------------------------------
# CloudWatch publica metricas de Batch (FailedJobs por queue) cada 5 min.
resource "aws_cloudwatch_metric_alarm" "batch_failed" {
  alarm_name          = "${var.project}-batch-job-failed"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "FailedJobs"
  namespace           = "AWS/Batch"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Al menos un Batch job fallo (no por Spot interrupt)"
  treat_missing_data  = "notBreaching"
  dimensions = {
    JobQueue = var.batch_job_queue_name
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
}

# ----- Alarma 2: MAPE alto, por variedad ------------------------------
# Custom metric "MAPE" en namespace "${project}/Training", dimension
# `variety`. Emitida por el trainer (Parte 5).
resource "aws_cloudwatch_metric_alarm" "mape_high" {
  for_each = toset(var.varieties)

  alarm_name          = "${var.project}-mape-${lower(each.value)}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "MAPE"
  namespace           = "${var.project}/Training"
  period              = 3600 # 1h (MAPE se publica al final del run)
  statistic           = "Maximum"
  threshold           = var.mape_alarm_threshold
  alarm_description   = "MAPE de ${each.value} supero ${var.mape_alarm_threshold}%"
  treat_missing_data  = "notBreaching"
  dimensions = {
    variety = each.value
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

# ----- Alarma 3: ALB 5xx -----------------------------------------------
resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${var.project}-alb-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  treat_missing_data  = "notBreaching"
  dimensions = {
    LoadBalancer = var.alb_arn_suffix
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
}
