terraform {
  # >= 1.10.0 es obligatorio: el backend usa `use_lockfile=true` (locking nativo
  # S3, sin DynamoDB). En 1.6-1.9 ese flag se ignora SILENCIOSAMENTE -> applies
  # concurrentes podrian corromper el state sin aviso. Ver tasks/infra.yml::_init.
  required_version = ">= 1.10.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "Terraform"
      Env       = "prod"
    }
  }
}
