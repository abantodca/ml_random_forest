"""Lambda dispatcher: submit jobs a AWS Batch.

Payload aceptado:
{
  "varieties": "POP,JUPITER",      # CSV o "all"
  "tuning":    "prod_xl",          # smoke|dev|prod|prod_xl (default prod_xl, igual que local)
  "parallel":  1,                  # opcional, N variedades en paralelo (default 1 = secuencial)
  "s3_data_key": "BD_HISTORICO_ACUMULADO.xlsx"   # opcional, default = ese mismo
}

Contrato del trainer (main.py):
- CMD ["--varieties","POP,JUPITER","--tuning","prod_xl"]  (+ "--parallel-varieties","N" si parallel>1)
- ENV S3_DATA_BUCKET, S3_DATA_KEY (para _hydrate_data_from_s3)
- ENV MLFLOW_TRACKING_URI, S3_ARTIFACTS_BUCKET, ... (ya en job-def)
"""

from __future__ import annotations

import json
import logging
import os
import re

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

batch = boto3.client("batch")

PROJECT            = os.environ["PROJECT"]
JOB_QUEUE_SPOT     = os.environ["JOB_QUEUE_SPOT"]
JOB_QUEUE_ONDEMAND = os.environ["JOB_QUEUE_ONDEMAND"]
JOB_DEFINITION     = os.environ["JOB_DEFINITION"]
DATA_BUCKET        = os.environ["DATA_BUCKET"]
VARIETIES_ALLOWED  = set(os.environ["VARIETIES_ALLOWED"].split(","))

TUNINGS = {"smoke", "dev", "prod", "prod_xl"}


def _normalize_varieties(raw: str) -> list[str]:
    if not raw:
        raise ValueError("varieties vacio")
    raw = raw.strip()
    if raw.lower() == "all":
        return sorted(VARIETIES_ALLOWED)
    items = [v.strip().upper() for v in raw.split(",") if v.strip()]
    bad = [v for v in items if v not in VARIETIES_ALLOWED]
    if bad:
        raise ValueError(f"variedades no permitidas: {bad}. Validas: {sorted(VARIETIES_ALLOWED)}")
    return items


def _validate_tuning(tuning: str) -> str:
    if tuning not in TUNINGS:
        raise ValueError(f"tuning invalido: {tuning}. Validos: {sorted(TUNINGS)}")
    return tuning


def _validate_key(key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._/\-]+\.xlsx", key):
        raise ValueError(f"s3_data_key invalido: {key}")
    return key


def _validate_mode(mode: str) -> str:
    # train (default) entrena; eda corre el analisis exploratorio standalone.
    if mode not in ("train", "eda"):
        raise ValueError(f"mode invalido: {mode}. Validos: eda, train")
    return mode


# Tope defensivo: main.py reparte cores=inner_n_jobs=cpu//parallel, asi que un
# parallel mayor que los vCPU del contenedor (16 en c6i.4xlarge) no aporta y solo
# fragmenta. 16 deja margen sin permitir valores absurdos por payload.
_MAX_PARALLEL = 16


def _validate_parallel(raw) -> int:
    try:
        p = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"parallel invalido: {raw!r} (entero >= 1)")
    if not (1 <= p <= _MAX_PARALLEL):
        raise ValueError(f"parallel invalido: {p} (rango 1..{_MAX_PARALLEL})")
    return p


def handler(event, _context):
    log.info("event: %s", json.dumps(event)[:1000])

    # EventBridge envuelve el payload en `detail`; manual invoke lo pasa raw.
    payload = event.get("detail", event) or {}

    try:
        varieties = _normalize_varieties(payload.get("varieties", ""))
        tuning    = _validate_tuning(payload.get("tuning", "prod_xl"))
        parallel  = _validate_parallel(payload.get("parallel", 1))
        s3_key    = _validate_key(payload.get("s3_data_key", "BD_HISTORICO_ACUMULADO.xlsx"))
        mode      = _validate_mode(payload.get("mode", "train"))
    except ValueError as exc:
        log.error("validacion fallo: %s", exc)
        return {"statusCode": 400, "body": str(exc)}

    queue = JOB_QUEUE_ONDEMAND if tuning == "prod_xl" else JOB_QUEUE_SPOT
    job_name = f"{PROJECT}-{'eda' if mode == 'eda' else tuning}-{'-'.join(varieties)[:50]}"
    # sanitize: Batch acepta [a-zA-Z0-9_-], max 128
    job_name = re.sub(r"[^a-zA-Z0-9_-]", "-", job_name)[:128]

    # EDA: standalone, no entrena -> ignora tuning/modelo (y parallel). Training:
    # parallel>1 reparte variedades en procesos (main.py::run_parallel); con una
    # sola variedad main.py lo ignora, asi que solo lo anexamos cuando aporta.
    if mode == "eda":
        command = ["--eda", "--varieties", ",".join(varieties)]
    else:
        command = ["--varieties", ",".join(varieties), "--tuning", tuning]
        if parallel > 1:
            command += ["--parallel-varieties", str(parallel)]

    response = batch.submit_job(
        jobName=job_name,
        jobQueue=queue,
        jobDefinition=JOB_DEFINITION,
        containerOverrides={
            "command": command,
            "environment": [
                {"name": "S3_DATA_BUCKET", "value": DATA_BUCKET},
                {"name": "S3_DATA_KEY",    "value": s3_key},
            ],
        },
        tags={
            "variety": ",".join(varieties),
            "tuning": tuning,
            "mode": mode,
            "parallel": str(parallel),
        },
    )

    log.info(
        "submit OK: jobId=%s queue=%s mode=%s parallel=%s",
        response["jobId"], queue, mode, parallel,
    )
    return {
        "statusCode": 200,
        "body": {
            "jobId":    response["jobId"],
            "jobName":  response["jobName"],
            "queue":    queue,
            "varieties": varieties,
            "tuning":   tuning,
            "parallel": parallel,
            "mode":     mode,
        },
    }
