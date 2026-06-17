# syntax=docker/dockerfile:1.7
# ============================================================================
# Imagen de la API (FastAPI) — servicio `api` del monorepo ml_training.
#
# IMPORTANTE: el contexto de build es la RAÍZ del repo (no api/), porque la
# imagen necesita el paquete `src/` raíz para des-picklear los modelos de
# MLflow (LagFeatureTransformer, FeatureGenerator, target_transform, ...).
# Una única fuente de verdad: el mismo `src/` que entrena el trainer.
#   docker build -f api/Dockerfile -t ml-training-api .
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder: instala dependencias en un venv aislado
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Toolchain mínimo para compilar wheels nativos (lightgbm/xgboost/asyncpg).
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY api/requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime: imagen final mínima
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ARG GIT_SHA=unknown
ARG VERSION=dev
LABEL org.opencontainers.image.title="ml-training-api" \
      org.opencontainers.image.description="FastAPI de pronosticos (sirve modelos rnd-forest-* de MLflow)" \
      org.opencontainers.image.source="https://github.com/abantodca/ml_training" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# libgomp1 = runtime de OpenMP (lightgbm/xgboost lo necesitan en inferencia).
# tini propaga SIGTERM correctamente cuando ECS/compose detienen la task.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libgomp1 curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 appuser \
    && useradd  --system --uid 1001 --gid appuser --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Código de la API + `src/` raíz (única fuente de verdad para unpickle).
# Orden de COPY: de mejor cache (cambia poco) a peor cache (cambia más).
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser api/app ./app

USER appuser
STOPSIGNAL SIGTERM

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
