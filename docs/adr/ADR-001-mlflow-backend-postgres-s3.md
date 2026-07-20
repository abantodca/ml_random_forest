# ADR-001 — El backend de MLflow es siempre Postgres + S3

**Estado:** accepted
**Relacionado:** [ADR-003](ADR-003-s3-real-sin-localstack.md) (casi siempre se citan juntos)

## Contexto

MLflow admite varios backends: `file://mlruns` en disco, sqlite, o un tracking server con base SQL y
un artifact store. Las tres primeras opciones son cómodas para arrancar y **no soportan el Model
Registry** ni concurrencia real.

El repo lo fija desde el archivo de configuración, `src/config.py:6-13`:

> "Backend MLflow: El proyecto SIEMPRE usa un MLflow server (Postgres + S3 detras). En local lo sirve
> `docker compose up` (servicio mlflow en :5000, backend Postgres + S3 real parametrizado via
> S3_MLFLOW_BUCKET). En produccion apuntas la misma env var `MLFLOW_TRACKING_URI` a tu server real
> (ECS Fargate detras de ALB). No hay backend file://mlruns ni sqlite local ni LocalStack."

Y `src/step_06_track/mlflow_registry.py:3-6`:

> "Model Registry siempre habilitado (Postgres lo soporta nativamente; **el viejo modo file://mlruns
> no**)."

## Decisión

Una sola forma de backend —**MLflow server con metadata en Postgres y artifacts en S3**— idéntica en
local y en producción. Lo único que cambia entre los dos entornos es a dónde apunta
`MLFLOW_TRACKING_URI`.

## Consecuencias

**Se gana:**

- El Model Registry está disponible **incondicionalmente**, sin ramas de código que lo comprueben.
  `src/config.py:710-712`: *"ADR-001 garantiza que SIEMPRE corremos contra un MLflow server con
  backend SQL (Postgres en local + AWS), por lo que el Registry esta disponible incondicionalmente."*
- Concurrencia: varias variedades entrenando en paralelo escriben al mismo tracking server sin
  corromper el estado.
- Paridad local↔prod. Es lo que hace que el smoke run local sea evidencia sobre producción y no una
  aproximación.

**Se pierde / queda prohibido:**

- `file://mlruns`, sqlite y cualquier backend de archivos. El directorio `mrun`/`mlruns/` que pueda
  quedar en el árbol es **legacy muerto**: el `.dockerignore` lo excluye bajo el header
  `# ── Legacy del modo file:// (ADR-001) ──`.
- Levantar el stack requiere Postgres arriba. No hay modo "solo un `python main.py` sin Docker" que
  registre modelos.

**Excepción deliberada:** sobrevive un guard vestigial en
`src/step_06_track/mlflow_registry.py:257-259`, que devuelve `None` si detecta un backend `file://`:

> "caso de emergencia: ADR-001 lo prohibe en operacion normal pero el guard se mantiene como red de
> seguridad"

No es una contradicción del ADR: es defensa en profundidad para que un misconfig degrade a "no
registro el modelo" en vez de a un error opaco.

## Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| `file://mlruns` | No soporta Model Registry. La tabla de decisiones fijas lo admite solo como *"Filesystem sólo en dev"*, y ni siquiera eso se adoptó |
| sqlite | Soporta Registry pero no concurrencia; rompe con variedades en paralelo |
| Backend distinto en local que en prod | Rompe la paridad: el smoke run dejaría de ser evidencia sobre producción |

## Dónde vive en el código

- `src/config.py:6-13`, `:62`, `:710-714`
- `src/step_06_track/mlflow_registry.py:3-6`, `:257-259`, `:284` (`is_file_backend`), `:337-341`
- `infra/modules/mlflow/` — RDS Postgres + ECS Fargate + ALB
- `docker-compose.yml` — servicio `mlflow` en `:5000` con `--serve-artifacts`
