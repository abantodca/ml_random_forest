variable "project" { type = string }
variable "vpc_cidr" { type = string }

# enable_nat=false libera el NAT gateway + EIP (el costo idle dominante, ~$33/mes)
# sin tocar VPC/subnets/SGs. Lo usa `task teardown` para apagar el NAT cuando el
# stack queda idle por un rato largo; `task rebuild`/deploy lo recrea (default true).
# Mientras este false, las subnets privadas no tienen salida a internet -> solo
# bajalo cuando no haya Fargate/Batch corriendo (teardown ya los destruyo).
variable "enable_nat" {
  type    = bool
  default = true
}
