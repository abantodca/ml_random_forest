data "aws_caller_identity" "current" {}

locals {
  # ${ACCOUNT: -7} en bash. substr(...,5,7) toma chars 5-11 (indices 0-based)
  # = los ultimos 7 chars de un account_id estandar de 12 digitos.
  # Coincide con scripts/aws-suffix.sh (POSIX `${acct#?????}`).
  account_suffix = substr(data.aws_caller_identity.current.account_id, 5, 7)
}

resource "aws_s3_bucket" "data" {
  bucket = "${var.project}-data-${local.account_suffix}"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project}-artifacts-${local.account_suffix}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-noncurrent"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration { noncurrent_days = 90 }
    abort_incomplete_multipart_upload { days_after_initiation = 7 }
  }
}

# ── ECR (5 repos) ────────────────────────────────────────────────────────────
# Tabla unica fuente de verdad. Antes eran 5 bloques aws_ecr_repository +
# 3 aws_ecr_lifecycle_policy casi identicos (~109 lineas). `lifecycle` marca
# que repos tienen politica de retencion (trainer/api/ui SI; mlflow/reports NO).
# NOTA: refactor con cambio de direccion en el state -> requiere `terraform
# state mv` antes del apply (ver header del PR / runbook). Sin migracion,
# terraform destruiria y recrearia los repos.
locals {
  ecr_repos = {
    trainer = { name = var.project, tag_mutability = "MUTABLE", lifecycle = true }                # CI/CD reusa "latest" + sha
    mlflow  = { name = "${var.project}-mlflow", tag_mutability = "IMMUTABLE", lifecycle = false } # v3.12.0 nunca cambia
    reports = { name = "${var.project}-reports", tag_mutability = "MUTABLE", lifecycle = false }  # iteramos nginx.conf seguido
    api     = { name = "${var.project}-api", tag_mutability = "MUTABLE", lifecycle = true }       # CI/CD reusa "latest" + sha
    ui      = { name = "${var.project}-ui", tag_mutability = "MUTABLE", lifecycle = true }
  }
  ecr_lifecycle_repos = { for k, v in local.ecr_repos : k => v if v.lifecycle }
}

resource "aws_ecr_repository" "this" {
  for_each             = local.ecr_repos
  name                 = each.value.name
  image_tag_mutability = each.value.tag_mutability
  force_delete         = true # destroy borra el repo aunque tenga imagenes
  image_scanning_configuration { scan_on_push = true }
  encryption_configuration { encryption_type = "AES256" }
}

resource "aws_ecr_lifecycle_policy" "this" {
  for_each   = local.ecr_lifecycle_repos
  repository = aws_ecr_repository.this[each.key].name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 tagged"
        selection    = { tagStatus = "tagged", tagPrefixList = ["v", "sha-"], countType = "imageCountMoreThan", countNumber = 10 }
        action       = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Expire untagged > 7 days"
        selection    = { tagStatus = "untagged", countType = "sinceImagePushed", countUnit = "days", countNumber = 7 }
        action       = { type = "expire" }
      }
    ]
  })
}
