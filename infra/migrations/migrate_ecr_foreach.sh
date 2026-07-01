#!/usr/bin/env bash
# Migracion de state: ECR 5 repos + 3 lifecycle policies de direcciones
# nombradas -> instancias for_each `this[<key>]`.
#
# CONTEXTO: el refactor DRY de infra/modules/storage/main.tf cambio las
# direcciones de los recursos. Sin esta migracion, `terraform apply` veria los
# recursos viejos como "a destruir" y los nuevos como "a crear" -> destruiria y
# recrearia los 5 repos ECR (perderia las imagenes). Con `state mv` Terraform
# entiende que es el MISMO recurso con otra direccion -> 0 cambios.
#
# USO:
#   1) cd infra/envs/prod   (con backend inicializado: `task infra:_init` o terraform init)
#   2) bash <ruta>/migrate_ecr_foreach.sh
#   3) terraform plan -target=module.storage   -> debe decir "No changes"
#   4) recien ahi: task deploy / task infra:apply
#
# Idempotente: si un `state mv` ya se hizo, terraform avisa "Invalid source
# address: ... does not exist" y el script sigue (|| true) con el resto.
set -uo pipefail

TF="terraform"

mv_one() {
  local from="$1" to="$2"
  echo ">>> $from  ->  $to"
  $TF state mv "$from" "$to" || echo "    (omitido: origen inexistente o ya migrado)"
}

echo "=== Pre-check: direcciones ECR actuales en el state ==="
$TF state list | grep -E 'module\.storage\.aws_ecr_(repository|lifecycle_policy)\.' || true
echo

# Repositorios (los 5)
for key in trainer mlflow reports api ui; do
  mv_one "module.storage.aws_ecr_repository.${key}" "module.storage.aws_ecr_repository.this[\"${key}\"]"
done

# Lifecycle policies (solo trainer/api/ui tienen)
for key in trainer api ui; do
  mv_one "module.storage.aws_ecr_lifecycle_policy.${key}" "module.storage.aws_ecr_lifecycle_policy.this[\"${key}\"]"
done

echo
echo "=== Post-check: nuevas direcciones en el state ==="
$TF state list | grep -E 'module\.storage\.aws_ecr_(repository|lifecycle_policy)\.this' || true
echo
echo "Ahora corre:  terraform plan -target=module.storage   (esperado: No changes / 0 to destroy)"
