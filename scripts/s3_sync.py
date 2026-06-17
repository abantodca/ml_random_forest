"""Sube artifacts/ y reports/ a S3 al terminar el entrenamiento.

Se invoca desde main.py si S3_ARTIFACTS_BUCKET esta configurado.
No rompe el pipeline si falla: loguea el error y continua.

Variables de entorno requeridas en EC2/CI:
    S3_ARTIFACTS_BUCKET   — nombre del bucket (sin s3://)
    S3_ARTIFACTS_PREFIX   — prefijo para artifacts/ (default: ml-training)
    S3_REPORTS_PREFIX     — prefijo para reports/  (default: ml-training/reports)
    AWS_DEFAULT_REGION    — region del bucket

Estructura en S3 resultante:
    s3://bucket/ml-training/artifacts/run_summary_AGGREGATE.json
    s3://bucket/ml-training/artifacts/champion_POP.json
    s3://bucket/ml-training/artifacts/final_pipeline_POP_xgb.joblib
    s3://bucket/ml-training/reports/Winner_POP.html
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("ml_pipeline")


def _upload_dir(
    s3_client,
    local_dir: Path,
    bucket: str,
    prefix: str,
    *,
    extensions: tuple[str, ...] | None = None,
) -> int:
    """Sube todos los archivos de `local_dir` a s3://bucket/prefix/.

    Returns numero de archivos subidos.
    """
    if not local_dir.exists():
        return 0

    uploaded = 0
    for file_path in local_dir.iterdir():
        if not file_path.is_file():
            continue
        if extensions and file_path.suffix.lower() not in extensions:
            continue
        key = f"{prefix}/{file_path.name}".lstrip("/")
        try:
            s3_client.upload_file(str(file_path), bucket, key)
            uploaded += 1
        except Exception as exc:
            logger.warning(f"S3 upload fallido: {file_path.name} -> {key} | {exc}")

    return uploaded


def sync_to_s3(
    artifacts_dir: Path,
    reports_dir: Path,
    bucket: str,
    artifacts_prefix: str,
    reports_prefix: str,
) -> bool:
    """Sube artifacts/ y reports/ a S3.

    Retorna True si todo fue bien, False si hubo errores parciales.
    Nunca levanta excepcion — el pipeline siempre termina aunque S3 falle.
    """
    try:
        import boto3
    except ImportError:
        logger.error("boto3 no instalado. Instala con: pip install boto3")
        return False

    try:
        s3 = boto3.client("s3")

        # artifacts/: JSONs + joblibsn
        n_artifacts = _upload_dir(
            s3,
            artifacts_dir,
            bucket,
            artifacts_prefix,
            extensions=(".json", ".joblib"),
        )
        # reports/: HTMLs y Excels del dashboard ejecutivo
        n_reports = _upload_dir(
            s3,
            reports_dir,
            bucket,
            reports_prefix,
            extensions=(".html", ".xlsx"),
        )

        logger.info(
            f"S3 sync OK | bucket={bucket} | "
            f"artifacts={n_artifacts} | reports={n_reports}"
        )
        logger.info(
            f"  artifacts -> s3://{bucket}/{artifacts_prefix}/\n"
            f"  reports   -> s3://{bucket}/{reports_prefix}/"
        )
        return True

    except Exception as exc:
        logger.error(
            f"S3 sync FALLO (el training se guardo en disco igualmente): {exc}"
        )
        return False
