# ============================================================================
# Modulo api — FastAPI en ECS Fargate (mismo cluster + ALB que MLflow).
#
# Sirve los modelos rnd-forest-* registrados en MLflow y persiste pronosticos
# en la base `forecasts` del RDS de MLflow (la API la auto-crea al boot).
# Patron espejo del modulo reports + service discovery + secret RDS.
# ============================================================================
data "aws_region" "current" {}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project}/api"
  retention_in_days = var.log_retention_days
}

# ── IAM ────────────────────────────────────────────────────────────────────
resource "aws_iam_role" "api_exec" {
  name               = "${var.project}-api-exec"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

resource "aws_iam_role_policy_attachment" "api_exec" {
  role       = aws_iam_role.api_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# El exec role inyecta el secret del RDS password al arrancar la task.
resource "aws_iam_role_policy" "api_exec_secret" {
  role = aws_iam_role.api_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = var.rds_password_secret_arn
    }]
  })
}

resource "aws_iam_role" "api_task" {
  name               = "${var.project}-api-task"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

# Task role: leer artifacts de S3 (boto3 vía cliente MLflow al cargar modelos).
resource "aws_iam_role_policy" "api_task_s3" {
  role = aws_iam_role.api_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [var.artifacts_bucket_arn, "${var.artifacts_bucket_arn}/*"]
    }]
  })
}

# ── Service discovery (la UI llama a api.<project>.local:8000) ───────────────
resource "aws_service_discovery_service" "api" {
  name = "api"
  dns_config {
    namespace_id = var.service_discovery_namespace_id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }
}

# ── ALB target group + reglas ────────────────────────────────────────────────
resource "aws_lb_target_group" "api" {
  name        = "${var.project}-tg-api"
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  health_check {
    path                = "/api/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
    matcher             = "200"
  }
  deregistration_delay = 30
}

# Ruteo por PREFIJOS ESPECIFICOS (no `/api/*` a secas): MLflow es el default del
# ALB y con --serve-artifacts expone /api/2.0/mlflow-artifacts/*. Un `/api/*`
# generico robaria esa ruta y rompe el preview de artifacts del MLflow UI.
# Listamos solo los prefijos reales del FastAPI.
resource "aws_lb_listener_rule" "api_functional" {
  listener_arn = var.alb_listener_arn
  priority     = 88

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
  condition {
    path_pattern {
      values = ["/api/health*", "/api/forecasts*", "/api/varieties*", "/api/history*"]
    }
  }
}

# Swagger / OpenAPI publico (showcase). Mantiene la doc accesible en /docs.
resource "aws_lb_listener_rule" "api_docs" {
  listener_arn = var.alb_listener_arn
  priority     = 89

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
  condition {
    path_pattern { values = ["/docs", "/openapi.json", "/redoc"] }
  }
}

# ── ECS task definition + service ────────────────────────────────────────────
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.cpu)
  memory                   = tostring(var.memory)
  execution_role_arn       = aws_iam_role.api_exec.arn
  task_role_arn            = aws_iam_role.api_task.arn

  container_definitions = jsonencode([
    {
      name         = "api"
      image        = var.api_image
      essential    = true
      portMappings = [{ containerPort = 8000, protocol = "tcp" }]
      # Componemos DATABASE_URL en runtime para inyectar el password del RDS
      # (secret) sin persistirlo. Single `$` a proposito (igual que MLflow):
      # Terraform solo escapa `$${`; `$RDS_PASSWORD` queda literal y lo expande
      # el shell con la env var inyectada desde Secrets Manager.
      command = [
        "sh", "-c",
        "export DATABASE_URL=postgresql://mlflow:$RDS_PASSWORD@${var.rds_address}:5432/forecasts; exec uvicorn app.main:app --host 0.0.0.0 --port 8000"
      ]
      secrets = [{
        name      = "RDS_PASSWORD"
        valueFrom = var.rds_password_secret_arn
      }]
      environment = [
        { name = "MLFLOW_TRACKING_URI", value = var.mlflow_tracking_uri },
        { name = "EXPERIMENT_PREFIX", value = var.model_registry_prefix },
        { name = "MLFLOW_PRELOAD_MODELS", value = tostring(var.mlflow_preload_models) },
        { name = "CORS_ORIGINS", value = var.cors_origins },
        { name = "AWS_DEFAULT_REGION", value = data.aws_region.current.region }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = data.aws_region.current.region
          awslogs-stream-prefix = "api"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -fsS http://localhost:8000/api/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 40
      }
    }
  ])
}

resource "aws_ecs_service" "api" {
  name            = "api"
  cluster         = var.ecs_cluster_id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
  propagate_tags  = "SERVICE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.sg_api_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  service_registries {
    registry_arn = aws_service_discovery_service.api.arn
  }

  # El scheduler maneja desired_count (0 fuera de horario) -> ignore drift.
  lifecycle {
    ignore_changes = [desired_count]
  }
}
