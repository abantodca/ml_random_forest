# ADR-006 — El ALB rutea por prefijos específicos, nunca `/api/*` genérico

**Estado:** accepted
**Número asignado:** 2026-07-20, en la reorganización documental. El rationale ya estaba escrito dos
veces —en prosa y como comentario dentro del HCL— pero sin número de ADR.

## Contexto

Un solo ALB sirve cuatro cosas en producción: MLflow (default), la API FastAPI, la UI Streamlit y el
nginx de reports. La forma obvia de rutear la API sería una regla `/api/*` → target group de la API.

Eso rompe MLflow, y el motivo está documentado literalmente dos veces. En prosa:

> "**Por que ruteo por prefijos especificos (no `/api/*` a secas)**: MLflow es el default del ALB y
> con `--serve-artifacts` expone `/api/2.0/mlflow-artifacts/*`. Un `/api/*` generico le robaria esa
> ruta y rompe el preview de artifacts del MLflow UI. Por eso listamos solo los prefijos reales del
> FastAPI."

Y como comentario dentro del `aws_lb_listener_rule.api_functional`, para que nadie lo "simplifique"
al editar el Terraform sin leer la guía.

## Decisión

Las listener rules enumeran **los prefijos reales de la API**, nunca un comodín:

| Priority | Patrones | Destino |
|---|---|---|
| 88 | `/api/health*`, `/api/forecasts*`, `/api/varieties*`, `/api/history*` | API FastAPI |
| 89 | `/docs`, `/openapi.json`, `/redoc` | API (Swagger público, showcase) |
| 100 | `/reports/*`, `/artifacts/*` | nginx Fargate Spot |
| — | `/app/*` | UI Streamlit |
| default | todo lo demás, incluido `/api/2.0/mlflow-artifacts/*` | **MLflow** |

Regla de orden: *"el ALB evalúa de menor a mayor; `priority=100` corre antes del default action
(siempre último)"*.

Complemento: **la UI no llama a la API por el ALB.** Usa service discovery interno
(`http://api.ml-training.local:8000`), así que el tráfico UI→API ni siquiera entra en este árbol de
decisión.

## Consecuencias

**Se gana:**

- El preview de artifacts del MLflow UI funciona. Es la funcionalidad que un `/api/*` genérico rompe
  de forma silenciosa: el UI carga, la lista de runs carga, y solo revienta al abrir un artifact.
- Las rutas expuestas son explícitas y auditables: se ve de un vistazo qué hay publicado.

**Se pierde:**

- **Agregar un router nuevo a la API exige tocar Terraform.** Un `POST /api/reportes` que funcione en
  local va a dar 404 en producción hasta que se agregue el prefijo a la regla 88. Es el costo directo
  de esta decisión y la trampa número uno para quien agrega endpoints.

## Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| `/api/*` genérico a la API | Roba `/api/2.0/mlflow-artifacts/*` y rompe el preview de artifacts de MLflow |
| Montar la API bajo otro prefijo (`/svc/*`) | Cambiaría el contrato con la UI y con los clientes; además `/docs` y `/openapi.json` seguirían fuera |
| Un ALB por servicio | ~$16/mes cada uno, para un beneficio nulo a esta escala |
| Sacar `--serve-artifacts` de MLflow | Obligaría a que cada cliente tenga credenciales de S3 directas |

## Dónde vive en el código

- `infra/modules/api/` — `aws_lb_listener_rule.api_functional` (priority 88) y la de Swagger (89)
- `infra/modules/reports/` — priority 100
- `infra/modules/ui/` — regla de `/app/*`
- `infra/modules/mlflow/` — default action del listener
- `ui/app/client/` — usa service discovery interno, no el ALB
