"""Emite custom metrics a CloudWatch.

Solo se activa cuando el trainer corre dentro de AWS Batch (detectado via
`AWS_BATCH_JOB_ID`, env var que el servicio Batch inyecta automaticamente y
NO existe en local). En docker compose local es no-op silencioso, evitando
contaminar el namespace prod con datos de smoke tests.
"""
from __future__ import annotations

import logging
import os
from typing import Final

log = logging.getLogger(__name__)

NAMESPACE: Final[str] = "ml-training/Training"


def emit_mape_metric(variety: str, mape_value: float) -> None:
    """Publica MAPE a CloudWatch con dimension `variety`.

    No falla el training si la publicacion falla (best-effort).
    """
    if not os.environ.get("AWS_BATCH_JOB_ID"):
        # Local (docker compose): skip silencioso.
        # AWS_BATCH_JOB_ID lo inyecta el servicio Batch automaticamente.
        return

    try:
        import boto3
    except ImportError:
        log.warning("boto3 no instalado, skip CloudWatch metric")
        return

    try:
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[{
                "MetricName": "MAPE",
                "Dimensions": [{"Name": "variety", "Value": variety}],
                "Value":      float(mape_value),
                "Unit":       "Percent",
            }],
        )
        log.info("CloudWatch MAPE=%.4f emitido (variety=%s)", mape_value, variety)
    except Exception as exc:
        log.warning("CloudWatch put_metric_data fallo: %s", exc)
