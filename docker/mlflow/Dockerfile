FROM ghcr.io/mlflow/mlflow:v3.12.0

RUN pip install --no-cache-dir \
        psycopg2-binary==2.9.9 \
        boto3==1.38.0

LABEL org.opencontainers.image.title="mlflow-with-pg-s3" \
      org.opencontainers.image.description="MLflow 3.12.0 + psycopg2-binary + boto3" \
      org.opencontainers.image.source="https://github.com/abantodca/ml_training" \
      org.opencontainers.image.base.name="ghcr.io/mlflow/mlflow:v3.12.0"
