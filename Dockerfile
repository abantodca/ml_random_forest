# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.13.1-slim-bookworm

FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./

# Cache mount de BuildKit: el pip cache persiste entre builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --wheel-dir /wheels -r requirements.txt


FROM python:${PYTHON_VERSION} AS runtime

ARG GIT_SHA=unknown
ARG VERSION=dev
# created/timestamp NO va aqui: se inyecta como --label en `docker build` para
# no invalidar la cache ni cambiar el digest de capas en cada rebuild del commit.
LABEL org.opencontainers.image.title="ml-training" \
      org.opencontainers.image.description="Random Forest training pipeline" \
      org.opencontainers.image.source="https://github.com/abantodca/ml_training" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates tini git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 mluser \
    && useradd  --system --uid 1001 --gid mluser --home ${APP_HOME} mluser

WORKDIR ${APP_HOME}

COPY --from=builder /wheels /wheels
COPY requirements.txt ./
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Orden de COPY: de mejor cache (cambia poco) a peor cache (cambia más)
COPY --chown=mluser:mluser src/    ./src/
COPY --chown=mluser:mluser scripts/ ./scripts/
COPY --chown=mluser:mluser main.py  ./

# Directorios que init_dirs() asume (idempotente)
RUN mkdir -p data/training logs artifacts reports \
    && chown -R mluser:mluser ${APP_HOME}

# MLflow llama `getpass.getuser()` para taggear `mlflow.user` en cada run; sin
# USER/LOGNAME en el environment (el `USER mluser` de abajo NO setea la env var)
# emite "No username set in the environment" varias veces por modelo. Fijarlo
# explicito silencia el warning y deja un tag legible en la UI de MLflow. Vale
# tanto en compose como en AWS Batch (misma imagen).
ENV USER=mluser
USER mluser
STOPSIGNAL SIGTERM

# tini propaga SIGTERM correctamente cuando Batch mata el job
ENTRYPOINT ["/usr/bin/tini", "--", "python", "main.py"]
CMD ["--varieties", "POP", "--tuning", "smoke"]
