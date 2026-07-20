# ADR-003 — S3 real también en local; nunca LocalStack

**Estado:** accepted
**Relacionado:** [ADR-001](ADR-001-mlflow-backend-postgres-s3.md)

## Contexto

El stack local necesita un artifact store para MLflow y un origen de datos. La opción habitual es
emular S3 (LocalStack, MinIO) para no depender de la nube durante el desarrollo.

Acá se decidió lo contrario, y el motivo está en el argumento de fidelidad de la guía de producción:

> "El origen del dataset es la única ruta donde 'mismo binario' se rompe sutilmente: en local lee un
> bind-mount, en Batch baja desde S3. **Si nunca probaste el path S3 en local, lo descubrís recién en
> el primer `task batch:smoke`.**"

Es decir: emular S3 mueve el descubrimiento del fallo al peor momento posible —el primer despliegue—
en vez de al desarrollo.

`src/config.py:62-68` lo fija:

> "Apunta SIEMPRE a un bucket S3 real (ADR-003: no usamos LocalStack). En local lo configurás vía
> `.env` (`S3_ARTIFACTS_BUCKET=<tu-bucket>`). En AWS Batch lo inyecta la job-def. El upload ocurre al
> final de `main.py` si el bucket esta configurado; `scripts/s3_sync.py` es defensivo: si S3 falla, el
> training termina OK igual y los artefactos quedan en disco local del container."

## Decisión

El entorno local usa **buckets S3 reales** con sufijo de cuenta, creados por
`task local:ensure-buckets`:

> "Crea los buckets con el mismo `ACCOUNT_SUFFIX` pero **sin versioning**, asi local y prod comparten
> nombres sin colisionar."

## Consecuencias

**Se gana:**

- El path de S3 —credenciales, permisos, nombres de bucket, serialización— se ejercita en cada
  corrida local. Los fallos de integración aparecen en tu máquina, no en el primer job de Batch.
- Coste despreciable: la guía estima el tramo local en **~$0.05/mes**, solo almacenamiento.

**Se pierde:**

- Hace falta una cuenta AWS y credenciales válidas para desarrollar. No hay modo offline.
- El `~/.aws` se monta `:ro` dentro de los contenedores; eso es una dependencia del host que hay que
  documentar en el onboarding.

**Mitigación:** `scripts/s3_sync.py` es deliberadamente **defensivo** — si S3 falla, el entrenamiento
termina OK y los artefactos quedan en el disco del contenedor. Un problema de red no te cuesta una
corrida de dos horas.

## Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| LocalStack | Emula la API pero no los permisos ni el comportamiento real; mueve el descubrimiento del fallo al primer deploy |
| MinIO | Mismo problema que LocalStack, más un servicio extra que mantener en el compose |
| Bind-mount en local y S3 en prod | Es exactamente la divergencia que el argumento de fidelidad busca evitar |

## Dónde vive en el código

- `src/config.py:62-74`
- `scripts/s3_sync.py` (comportamiento defensivo), `scripts/ensure-s3-bucket.sh`
- `tasks/local.yml` — `local:ensure-buckets`
- `main.py::_hydrate_data_from_s3` — el path que Batch ejercita y que local también debe ejercitar
