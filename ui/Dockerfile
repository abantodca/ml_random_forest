# syntax=docker/dockerfile:1.7
# ============================================================================
# Imagen de la UI (Streamlit) — servicio `ui` del monorepo ml_training.
# Contexto de build = carpeta ui/.  docker build -f ui/Dockerfile -t ml-training-ui ui
#
# Ruteo: en local sirve en la raíz (:8501). En prod, detrás del ALB en /app/*,
# se setea STREAMLIT_SERVER_BASE_URL_PATH=app (env nativa de Streamlit) en la
# task de ECS — no requiere rebuild.
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ARG GIT_SHA=unknown
ARG VERSION=dev
LABEL org.opencontainers.image.title="ml-training-ui" \
      org.opencontainers.image.description="Streamlit dashboard de pronosticos (consume la API)" \
      org.opencontainers.image.source="https://github.com/abantodca/ml_training" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 appuser \
    && useradd  --system --uid 1001 --gid appuser --create-home appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser . .

USER appuser
STOPSIGNAL SIGTERM

EXPOSE 8501

# Healthcheck sensible al base-path: /_stcore/health en local, /app/_stcore/health en prod.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -fsS "http://localhost:8501/${STREAMLIT_SERVER_BASE_URL_PATH:+${STREAMLIT_SERVER_BASE_URL_PATH}/}_stcore/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
