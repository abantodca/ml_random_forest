# ADR-009 — El secreto del RDS vive fuera del módulo que se destruye

**Estado:** accepted
**Número asignado:** 2026-07-20. `infra/envs/prod/rds_secret.tf` se autodenomina ADR en su propio
comentario (*"Por que aca (ADR del ciclo teardown/rebuild)"*) pero nunca recibió número.

## Contexto

El lifecycle del stack tiene cuatro modos, y dos de ellos destruyen infraestructura:
`task ops:teardown` elimina los `VOLATILE_MODULES` para dejar de pagar, y `task ops:rebuild` la
vuelve a levantar. `module.mlflow` contiene el RDS, así que en un teardown **la instancia se borra**
(tras tomarle un backup verificado — ver `02-produccion-aws.md` #8.5).

El problema, literal de `infra/envs/prod/rds_secret.tf:1-26`:

> "`task ops:teardown` destruye los VOLATILE_MODULES, y `module.mlflow` contiene el RDS -> la
> instancia se BORRA (con snapshot final timestamped). Si el `random_password` y su secret vivieran
> dentro del modulo, se irian con el, y `task ops:rebuild` generaria una password NUEVA. Al restaurar
> el snapshot, la base recuperada conserva la password VIEJA -> MLflow y la API no autentican y **el
> backup es irrecuperable en la practica.**"

Es un fallo particularmente cruel: el backup existe, está íntegro, y es inservible. Y no se descubre
en el teardown —se descubre en el rebuild, cuando ya necesitás los datos.

## Decisión

`random_password` y su `aws_secretsmanager_secret` se declaran en **`infra/envs/prod/`**, no dentro
de `modules/mlflow/`. El módulo recibe la password como input. Así el secreto **sobrevive** al ciclo
teardown → rebuild y sigue coincidiendo con el que quedó grabado en el backup.

## Consecuencias

**Se gana:** el backup del RDS es efectivamente restaurable. El ciclo de ahorro —que es el punto
del diseño— no destruye la recuperabilidad.

**Se pierde:** la composición es menos limpia. El módulo `mlflow` no es autocontenido: depende de que
el env le pase la credencial. Es una asimetría deliberada, y por eso está comentada en el archivo.

**Regla de migración**, del mismo archivo: mover este recurso entre módulos en un stack ya
desplegado *"rotaria la password y romperia el RDS vivo"*. Si hiciera falta moverlo, va con
`terraform state mv`, nunca con destroy/create.

## Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| `random_password` dentro de `modules/mlflow/` | Se destruye con el módulo; el rebuild genera una password nueva que no coincide con la del backup → backup irrecuperable |
| Password fija en `tfvars` | Secreto en texto plano y versionado |
| Rotar la password en cada rebuild y actualizar la base restaurada | Requiere levantar la base con la password vieja para poder cambiarla — el huevo y la gallina |

## Dónde vive en el código

- `infra/envs/prod/rds_secret.tf` — la declaración y el comentario que originó este ADR
- `infra/modules/mlflow/` — recibe la password como variable
- `tasks/ops.yml` — `ops:teardown`, `ops:rebuild`, y la lista `VOLATILE_MODULES`
