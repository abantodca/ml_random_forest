# Architecture Decision Records

Decisiones estructurales del sistema, con su contexto y sus consecuencias. Cada una está
referenciada desde el código (`src/config.py`, `champion.py`, `mlflow_registry.py`), desde
`CLAUDE.md` y desde las guías de despliegue.

**Cambiar una decisión ratificada acá implica un ADR nuevo que la supersede, no un parche local.**

| # | Decisión | Estado |
|---|---|---|
| [001](ADR-001-mlflow-backend-postgres-s3.md) | El backend de MLflow es siempre Postgres + S3 | accepted |
| [002](ADR-002-campeon-automatico.md) | El sistema elige el campeón; no hay flag para forzar un backend | accepted |
| [003](ADR-003-s3-real-sin-localstack.md) | S3 real también en local; nunca LocalStack | accepted |
| [004](ADR-004-diagnostics-standalone.md) | Diagnostics/EDA/drift son standalone y no bloquean el training | accepted |
| [005](ADR-005-nombres-step-xx-contrato-serializacion.md) | Los nombres `step_XX_verbo/` son un contrato de serialización | accepted † |
| [006](ADR-006-ruteo-alb-por-prefijos.md) | El ALB rutea por prefijos específicos, nunca `/api/*` genérico | accepted † |
| [007](ADR-007-lags-dentro-del-pipeline.md) | Los lags se computan dentro del sklearn Pipeline | accepted † |
| [008](ADR-008-ci-sin-tests-todavia.md) | CI sin job `test` mientras no exista `tests/` | accepted |
| [009](ADR-009-rds-secret-fuera-del-modulo.md) | El secreto del RDS vive fuera del módulo que se destruye | accepted |

† **Número asignado en la reorganización documental de 2026-07-20.** La decisión y su rationale ya
existían y estaban citados en el repo (`CLAUDE.md`, `CONTRIBUTING.md`, `README.md`, comentarios de
código); lo que no existía era el número de ADR. Antes de esta fecha el repo solo mencionaba
ADR-001..004 y ADR-008, y `docs/adr/` no existía — todos los links eran placeholders rotos.

## Formato

Cada ADR tiene: **Contexto** (qué problema resolvía, con citas del repo), **Decisión** (qué se
decidió exactamente), **Consecuencias** (qué se gana, qué se pierde, qué queda prohibido),
**Alternativas descartadas** (con el porqué del descarte) y **Dónde vive en el código**.

Las citas entre comillas son texto literal del repo con su `archivo:línea`. Donde hubo que inferir,
está marcado explícitamente.
