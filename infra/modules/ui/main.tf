# ============================================================================
# Modulo ui — Streamlit en ECS Fargate (mismo cluster + ALB que MLflow).
# Sirve detras del ALB en /app/* (STREAMLIT_SERVER_BASE_URL_PATH=app) y consume
# la API por service discovery interno. Patron espejo del modulo reports.
# ============================================================================
data "aws_region" "current" {}

resource "aws_cloudwatch_log_group" "ui" {
  name              = "/ecs/${var.project}/ui"
  retention_in_days = var.log_retention_days
}

resource "aws_iam_role" "ui_exec" {
  name               = "${var.project}-ui-exec"
  assume_role_policy = file("${path.module}/../_shared/assume-ecs-tasks.json")
}

resource "aws_iam_role_policy_attachment" "ui_exec" {
  role       = aws_iam_role.ui_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
# Sin task role: la UI no accede a AWS (solo HTTP a la API).

# ── ALB target group + regla /app/* ─────────────────────────────────────────
resource "aws_lb_target_group" "ui" {
  name        = "${var.project}-tg-ui"
  port        = 8501
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  health_check {
    # Con base-path, Streamlit expone el health bajo el prefijo.
    path                = "/${var.base_url_path}/_stcore/health"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
    matcher             = "200"
  }
  deregistration_delay = 30
}

resource "aws_lb_listener_rule" "ui" {
  listener_arn = var.alb_listener_arn
  priority     = 70

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ui.arn
  }
  condition {
    path_pattern { values = ["/${var.base_url_path}", "/${var.base_url_path}/*"] }
  }
}

# ── ECS task definition + service ────────────────────────────────────────────
resource "aws_ecs_task_definition" "ui" {
  family                   = "${var.project}-ui"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = tostring(var.cpu)
  memory                   = tostring(var.memory)
  execution_role_arn       = aws_iam_role.ui_exec.arn

  container_definitions = jsonencode([
    {
      name         = "ui"
      image        = var.ui_image
      essential    = true
      portMappings = [{ containerPort = 8501, protocol = "tcp" }]
      environment = [
        { name = "API_URL", value = var.api_internal_url },
        { name = "STREAMLIT_SERVER_BASE_URL_PATH", value = var.base_url_path },
        { name = "LOG_LEVEL", value = "info" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ui.name
          awslogs-region        = data.aws_region.current.region
          awslogs-stream-prefix = "ui"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -fsS http://localhost:8501/${var.base_url_path}/_stcore/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 30
      }
    }
  ])
}

resource "aws_ecs_service" "ui" {
  name            = "ui"
  cluster         = var.ecs_cluster_id
  task_definition = aws_ecs_task_definition.ui.arn
  desired_count   = var.desired_count
  propagate_tags  = "SERVICE"

  # Fargate Spot: ~70% mas barato. La UI (Streamlit) es stateless; si Spot la
  # reclama, el usuario recarga. El capacity provider se habilita en el cluster.
  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
  }

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.sg_ui_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ui.arn
    container_name   = "ui"
    container_port   = 8501
  }

  lifecycle {
    ignore_changes = [desired_count]
  }
}
