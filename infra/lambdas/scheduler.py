"""Lambda scheduler: start/stop RDS + Fargate.

Acciones:
- start:    arranca RDS + ECS services secuencialmente (RDS -> MLflow -> Reports)
- stop:     baja ECS services a 0 + para RDS. Antes chequea Batch jobs RUNNING.
- keepstop: cada 6h. Si RDS quedo RUNNING fuera de ventana, lo re-para.
            Ventana parametrizada via WORKDAYS_CRON + WORK_START_UTC + WORK_END_UTC.
- autostop: disparado por EventBridge al terminar un job Batch (SUCCEEDED/FAILED).
            Apaga el stack completo (= stop) solo fuera de la ventana laboral y
            si no quedan otros jobs activos. Para que un entrenamiento que termina
            de noche no deje el cluster encendido hasta el cron stop del dia
            siguiente, sin tumbarlo tras un smoke/eda diurno.
"""

from __future__ import annotations

import logging
import os
import time

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

ecs   = boto3.client("ecs")
rds   = boto3.client("rds")
batch = boto3.client("batch")

ECS_CLUSTER        = os.environ["ECS_CLUSTER"]
ECS_SVC_MLFLOW     = os.environ["ECS_SVC_MLFLOW"]
ECS_SVC_REPORTS    = os.environ["ECS_SVC_REPORTS"]
# App stack (defaults tolerantes: los servicios se llaman literalmente api/ui).
ECS_SVC_API        = os.environ.get("ECS_SVC_API", "api")
ECS_SVC_UI         = os.environ.get("ECS_SVC_UI", "ui")
RDS_INSTANCE       = os.environ["RDS_INSTANCE"]
JOB_QUEUE_SPOT     = os.environ["JOB_QUEUE_SPOT"]
JOB_QUEUE_ONDEMAND = os.environ["JOB_QUEUE_ONDEMAND"]

# Patch 13.1: workdays + horas configurables via env (default = comportamiento
# original MON-FRI 13-17 UTC). EventBridge cron usa los mismos tokens (MON,WED,FRI).
_WEEKDAY_MAP = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}


def _parse_workdays(cron_token: str) -> set[int]:
    """Parsea 'MON,WED,FRI' o 'MON-FRI' a un set de tm_wday (0=lunes)."""
    cron_token = cron_token.strip().upper()
    if "-" in cron_token:
        a, b = cron_token.split("-", 1)
        ia, ib = _WEEKDAY_MAP[a.strip()], _WEEKDAY_MAP[b.strip()]
        return set(range(ia, ib + 1))
    return {_WEEKDAY_MAP[tok.strip()] for tok in cron_token.split(",") if tok.strip()}


def _in_work_window() -> bool:
    """True si el instante actual (UTC) cae en la ventana laboral configurada.

    Ventana = WORKDAYS_CRON (dias) x [WORK_START_UTC, WORK_END_UTC). Dentro de
    ella el cron start/stop gobierna el on/off; fuera de ella el cluster debe
    estar abajo, asi que keepstop (cron 6h) y autostop (fin de job) solo actuan
    cuando esto devuelve False.
    """
    workdays = _parse_workdays(os.environ.get("WORKDAYS_CRON", "MON-FRI"))
    start_utc = int(os.environ.get("WORK_START_UTC", "13"))
    end_utc = int(os.environ.get("WORK_END_UTC", "17"))
    utc_hour = time.gmtime().tm_hour
    weekday = time.gmtime().tm_wday  # 0=lunes
    return (weekday in workdays) and (start_utc <= utc_hour < end_utc)


def _running_jobs() -> list[str]:
    """IDs de jobs en estado RUNNING o RUNNABLE en cualquiera de las queues."""
    ids: list[str] = []
    for queue in (JOB_QUEUE_SPOT, JOB_QUEUE_ONDEMAND):
        for status in ("RUNNING", "RUNNABLE", "STARTING"):
            resp = batch.list_jobs(jobQueue=queue, jobStatus=status)
            ids.extend(j["jobId"] for j in resp.get("jobSummaryList", []))
    return ids


def _start():
    """Wake secuencial: RDS -> MLflow -> Reports (Patch 13.3).

    Por que serializar y no lanzar todo en paralelo:
    1. El container MLflow intenta conectar a RDS al arrancar. Si RDS
       no esta available, falla healthcheck startPeriod -> ECS lo
       reinicia. Costoso en tiempo.
    2. Reports depende de S3 (no de RDS o MLflow) pero su UI vacia es
       confusa si MLflow todavia esta cargando. Mejor secuencial.
    """
    log.info("=== START (secuencial: RDS -> MLflow -> Reports) ===")

    # Etapa 1: RDS
    db = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE)["DBInstances"][0]
    if db["DBInstanceStatus"] == "stopped":
        rds.start_db_instance(DBInstanceIdentifier=RDS_INSTANCE)
        log.info("rds start_db_instance ack")

    # Esperar hasta available (max ~8 min)
    state = db["DBInstanceStatus"]
    for i in range(48):
        db = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE)["DBInstances"][0]
        state = db["DBInstanceStatus"]
        log.info("rds[%d]=%s", i, state)
        if state == "available":
            break
        time.sleep(10)
    else:
        raise RuntimeError(f"RDS no available tras 8 min (estado={state})")

    log.info("rds OK -> arrancando MLflow")

    # Etapa 2: MLflow Fargate
    ecs.update_service(cluster=ECS_CLUSTER, service=ECS_SVC_MLFLOW, desiredCount=1)
    log.info("ecs %s -> desiredCount=1", ECS_SVC_MLFLOW)

    # Esperar hasta running (max ~5 min). Si no llega, igual arrancamos reports.
    for i in range(30):
        svc = ecs.describe_services(cluster=ECS_CLUSTER, services=[ECS_SVC_MLFLOW])["services"][0]
        running = svc.get("runningCount", 0)
        log.info("mlflow[%d]: running=%d desired=%d", i, running, svc.get("desiredCount", 0))
        if running >= 1:
            break
        time.sleep(10)
    else:
        log.warning("MLflow no esta running tras 5 min, arrancamos reports igual")

    # Etapa 3: Reports + API + UI Fargate (no esperan, son no-bloqueantes).
    # La API tolera que MLflow aun no este listo (lazy-load); RDS ya esta up.
    for svc in (ECS_SVC_REPORTS, ECS_SVC_API, ECS_SVC_UI):
        ecs.update_service(cluster=ECS_CLUSTER, service=svc, desiredCount=1)
        log.info("ecs %s -> desiredCount=1", svc)
    log.info("=== START OK ===")


def _stop():
    log.info("=== STOP ===")
    running = _running_jobs()
    if running:
        log.warning(
            "Batch jobs activos (%d): %s. Postponiendo stop hasta proximo cron.",
            len(running), running[:5]
        )
        return

    # ECS: desired_count = 0 (incluye app stack api + ui)
    for svc in (ECS_SVC_MLFLOW, ECS_SVC_REPORTS, ECS_SVC_API, ECS_SVC_UI):
        ecs.update_service(cluster=ECS_CLUSTER, service=svc, desiredCount=0)
        log.info("ecs %s -> desiredCount=0", svc)

    # RDS: stop si esta RUNNING
    db = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE)["DBInstances"][0]
    state = db["DBInstanceStatus"]
    if state == "available":
        rds.stop_db_instance(DBInstanceIdentifier=RDS_INSTANCE)
        log.info("rds stop_db_instance ack")
    else:
        log.info("rds en estado %s (skip stop)", state)


def _keepstop():
    """Defense: si RDS quedo RUNNING fuera de ventana, re-pararlo (Patch 13.1)."""
    log.info("=== KEEPSTOP ===")
    if _in_work_window():
        log.info("dentro de ventana laboral, skip keepstop")
        return

    db = rds.describe_db_instances(DBInstanceIdentifier=RDS_INSTANCE)["DBInstances"][0]
    state = db["DBInstanceStatus"]
    if state == "available":
        running = _running_jobs()
        if running:
            log.warning("Batch jobs activos, skip keepstop")
            return
        rds.stop_db_instance(DBInstanceIdentifier=RDS_INSTANCE)
        log.info("rds re-stopped por keepstop")
    else:
        log.info("rds en estado %s (skip)", state)


def _autostop():
    """Disparado al terminar un job Batch (SUCCEEDED/FAILED) via EventBridge.

    Apaga el stack completo (delega en _stop: ECS a 0 + RDS stop) para no
    dejarlo encendido toda la noche tras un entrenamiento. Dos guardas evitan
    apagados indeseados:
    - solo fuera de la ventana laboral: dentro de ella el cron stop ya gobierna
      el apagado; no queremos tumbar el cluster tras un smoke/eda diurno.
    - _stop() ademas pospone si quedan otros jobs activos (otra variedad aun
      entrenando) -> no apaga a media corrida multi-variedad.
    """
    log.info("=== AUTOSTOP (trigger: Batch job terminal) ===")
    if _in_work_window():
        log.info("dentro de ventana laboral, skip autostop (probable job diurno)")
        return
    _stop()


def handler(event, _context):
    action = (event or {}).get("action", "stop")
    if action == "start":
        _start()
    elif action == "stop":
        _stop()
    elif action == "keepstop":
        _keepstop()
    elif action == "autostop":
        _autostop()
    else:
        raise ValueError(f"action desconocida: {action}")
    return {"statusCode": 200, "body": action}
