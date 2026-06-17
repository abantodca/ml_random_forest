data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  # Trust policy GHA-OIDC compartido entre gha-deploy y gha-train.
  # El template vive en infra/modules/_shared/assume-github-oidc.json.tftpl
  # — provider_arn / org / repo se inyectan via templatefile().
  gha_oidc_trust = templatefile("${path.module}/../_shared/assume-github-oidc.json.tftpl", {
    provider_arn = var.oidc_provider_arn
    org          = var.github_org
    repo         = var.github_repo
  })
}

# ----- Role 1: gha-deploy (CI workflows que aplican terraform + push ECR)
resource "aws_iam_role" "deploy" {
  name               = "${var.project}-gha-deploy"
  assume_role_policy = local.gha_oidc_trust
}

resource "aws_iam_role_policy" "deploy" {
  role = aws_iam_role.deploy.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Terraform: state remoto + lock nativo S3 (use_lockfile en backend "s3").
      # PutObject + DeleteObject cubren tanto el .tfstate como el .tfstate.tflock
      # que Terraform escribe junto al state. NO requiere DynamoDB.
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
        Resource = [
          "arn:aws:s3:::${var.project}-tfstate-*",
          "arn:aws:s3:::${var.project}-tfstate-*/*"
        ]
      },
      # ECR: push de las 5 imagenes (trainer + mlflow + reports + api + ui)
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage"
        ]
        Resource = "*" # ECR scoping necesita el endpoint para auth, * es estandar
      },
      # Terraform: leer/escribir resources (scope intencionalmente amplio para que
      # `terraform apply` funcione sobre TODOS los modulos. En produccion mas
      # estricta, dividir en roles plan-only + apply-only).
      #
      # BLAST RADIUS de este statement: un atacante que comprometa el OIDC
      # trust (ej. fork con write a `main`, o GitHub actions de un usuario
      # con permisos en el repo) puede:
      #   - destruir TODA la infra del proyecto (terraform destroy desde CI).
      #   - crear nuevos IAM roles (iam:*) y escalar a admin de la cuenta.
      #   - leer Secrets Manager (incluido el RDS password).
      # MITIGACIONES en uso:
      #   - trust policy con `sub = "repo:org/repo:*"` (solo este repo).
      #   - branch protection en main (#6.6) + required reviewers.
      #   - GitHub Environment "production" con manual approval (#6.5).
      # Refinable en #10 (hardening): partir en deploy-plan-only + apply
      # con CODEOWNERS, o restringir Resource por modulo via tags.
      {
        Effect = "Allow"
        Action = [
          "ec2:*", "vpc:*", "iam:*", "rds:*", "logs:*",
          "ecs:*", "elasticloadbalancing:*", "servicediscovery:*",
          "batch:*", "lambda:*", "events:*", "sns:*",
          "cloudwatch:*", "secretsmanager:*", "kms:*",
          "s3:GetBucketLocation", "s3:ListAllMyBuckets",
          "s3:CreateBucket", "s3:DeleteBucket", "s3:PutBucket*", "s3:GetBucket*",
          "ecr:*"
        ]
        Resource = "*"
      }
    ]
  })
}

# ----- Role 2: gha-train (solo invocar Lambda dispatcher) -------------
resource "aws_iam_role" "train" {
  name               = "${var.project}-gha-train"
  assume_role_policy = local.gha_oidc_trust
}

resource "aws_iam_role_policy" "train" {
  role = aws_iam_role.train.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["lambda:InvokeFunction"]
        # Patch 13.2: gha-train tambien invoca scheduler para wake/stop
        # en el workflow auto-train-on-push.yml.
        Resource = [
          "arn:aws:lambda:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:function:${var.project}-dispatcher",
          "arn:aws:lambda:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:function:${var.project}-scheduler"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["batch:DescribeJobs", "batch:ListJobs"]
        Resource = "*"
      },
      {
        # Patch 13.2: chequear estado RDS antes de wake
        Effect   = "Allow"
        Action   = ["rds:DescribeDBInstances"]
        Resource = "*"
      },
      {
        # Patch 13.2: chequear estado de los services Fargate
        Effect   = "Allow"
        Action   = ["ecs:DescribeServices"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:GetLogEvents", "logs:DescribeLogStreams"]
        Resource = "arn:aws:logs:${data.aws_region.current.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/batch/${var.project}*"
      }
    ]
  })
}
