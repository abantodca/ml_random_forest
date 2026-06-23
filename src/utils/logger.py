"""Setup del logger con salida a archivo y consola."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import LOGS_DIR

_DEFAULT_NAME = "ml_pipeline"
_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d:%(funcName)s | %(message)s"
)

_BUSINESS_AUDIT_FILE = LOGS_DIR / "business_audit.jsonl"


class PrefixAdapter(logging.LoggerAdapter):
    """LoggerAdapter que antepone un prefijo fijo a cada mensaje.

    Uso tipico (single_run.py):
        log = PrefixAdapter(logger, prefix=f"[{variety}/{model_type}]")
        log.info("[1/6] Cargando datos...")
        # -> "[POP/xgb] [1/6] Cargando datos..."

    Centraliza el prefijo en una sola variable -- antes aparecia repetido
    en ~12 f-strings dentro de train_model.
    """

    def __init__(self, logger: logging.Logger, prefix: str):
        super().__init__(logger, extra={"prefix": prefix})

    def process(self, msg, kwargs):
        return f"{self.extra['prefix']} {msg}", kwargs


def setup_logging(
    name: str = _DEFAULT_NAME,
    level: int = logging.INFO,
    log_file: str | Path = "pipeline_run.log",
) -> logging.Logger:
    """Configura logging y devuelve un logger.

    - Si `name` == default (ml_pipeline) configuramos el ROOT logger con
      handlers de archivo + consola, de modo que cualquier modulo que use
      `logging.getLogger(__name__)` herede los handlers automaticamente
      (zero side-effects en imports).
    - Si `name` != default (e.g. 'variety.POP') creamos un logger NOMBRADO
      con propagate=False y handler dedicado al archivo correspondiente.
      Util para los workers paralelos donde cada variedad tiene su log.

    En ambos casos la funcion es idempotente: llamadas repetidas con el
    mismo target NO duplican handlers.
    """
    formatter = logging.Formatter(_FORMAT)
    file_path = Path(log_file)
    if not file_path.is_absolute():
        file_path = LOGS_DIR / file_path

    target_root = name == _DEFAULT_NAME
    # Cuando es el target principal, los handlers viven en el ROOT para que
    # cualquier `logging.getLogger(__name__)` los herede sin tocar nada.
    target = logging.getLogger() if target_root else logging.getLogger(name)
    target.setLevel(level)

    # Idempotencia: si ya hay un handler de archivo con la misma ruta, no
    # agregamos otro. Esto deja al usuario re-llamar setup_logging() sin
    # duplicar lineas.
    have_file = any(
        isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == file_path
        for h in target.handlers
    )
    if not have_file:
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        target.addHandler(fh)

    have_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in target.handlers
    )
    if not have_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(level)
        target.addHandler(sh)

    if not target_root:
        # logger nombrado para workers paralelos: NO propagar al root para
        # evitar que sus eventos se escriban tambien en pipeline_run.log.
        target.propagate = False
        return target

    # Para el caso default devolvemos el logger nombrado (no el root) para
    # que los logs aparezcan con `name=ml_pipeline` y no `name=root`. Los
    # handlers viven en root y se heredan por propagacion.
    return logging.getLogger(_DEFAULT_NAME)


def log_business_audit(
    logger: logging.Logger,
    *,
    variety: str,
    model_type: str,
    tuning: str,
    business_validation,
    nested_metrics: dict[str, float],
    best_params: dict[str, Any],
    mlflow_run_id: str | None = None,
) -> Path:
    """Registra una linea de auditoria comparable entre entrenamientos.

    Escribe DOS cosas:
      1. Mensaje INFO al logger (visible en consola y en pipeline_run.log).
      2. Linea JSON en logs/business_audit.jsonl (un append-only diario para
         comparar runs y detectar regresiones entre entrenamientos sucesivos).

    Why: el negocio quiere saber 'el modelo de hoy es mejor o peor que el de
    ayer' en KG/JR (no en KG/JR_H). Este JSONL es la fuente de verdad para
    esa comparacion (y futuro grafico de evolucion en el HTML).
    """
    if business_validation.is_empty():
        logger.warning(
            f"[{variety}/{model_type}] business_audit: sin metricas (faltan KG/JR o H-EF en data)"
        )
        return _BUSINESS_AUDIT_FILE

    moof = business_validation.metrics_oof
    mins = business_validation.metrics_insample

    logger.info(
        f"[{variety}/{model_type}] BUSINESS (KG/JR) | "
        f"OOF: R2={moof.get('r2', float('nan')):.4f} "
        f"MAE={moof.get('mae', float('nan')):.4f} "
        f"MAPE={moof.get('mape', float('nan')):.2f}% | "
        f"InSample: R2={mins.get('r2', float('nan')):.4f} "
        f"MAE={mins.get('mae', float('nan')):.4f} | "
        f"n_oof={business_validation.n_oof} n_drop={business_validation.n_dropped_business}"
    )

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "variety": variety,
        "model_type": model_type,
        "tuning": tuning,
        "mlflow_run_id": mlflow_run_id,
        "n_oof": business_validation.n_oof,
        "n_insample": business_validation.n_insample,
        "n_dropped_business": business_validation.n_dropped_business,
        "metrics_kg_jr_h_oof": {
            "mae": float(nested_metrics.get("nested_cv_mae_mean", float("nan"))),
            "r2": float(nested_metrics.get("nested_cv_r2_mean", float("nan"))),
        },
        "metrics_kg_jr_oof": {k: float(v) for k, v in moof.items()},
        "metrics_kg_jr_insample": {k: float(v) for k, v in mins.items()},
        "best_params": {
            k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in best_params.items()
        },
    }

    _BUSINESS_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _BUSINESS_AUDIT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return _BUSINESS_AUDIT_FILE
