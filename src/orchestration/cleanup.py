"""Cleanup defensivo entre runs: figuras matplotlib, MLflow runs huerfanos, GC.

Tambien centraliza la limpieza de archivos residuales en `reports/` que
sobreviven a re-corridas de la misma variedad (HTMLs/Excels viejos sin el
campeon actual).
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow

from src.config import REPORTS_DIR

_logger = logging.getLogger(__name__)

# matplotlib backend ya esta forzado a 'Agg' en src/__init__.py (side-effect
# del paquete raiz). Importar pyplot aqui es seguro porque cualquier
# `import src.*` carga primero el __init__ del paquete.


def cleanup_state(logger, label: str) -> None:
    """Cierra figuras, termina runs MLflow huerfanos y forza GC.

    Llamar entre modelos y entre variedades para evitar:
      - Acumulacion de figuras matplotlib en memoria (RAM creciente).
      - Runs MLflow abiertos por errores no capturados (corrompen tags).
      - Memoria sin liberar de XGB/LGB que mantienen pools internos.
    """
    plt.close("all")
    if mlflow.active_run() is not None:
        mlflow.end_run()
    collected = gc.collect()
    logger.info(f"Cache limpiado ({label}) | gc liberados={collected}")


def cleanup_residual_reports(
    *,
    variety: str,
    keep: Iterable[Path | str],
    reports_dir: Path | None = None,
) -> list[Path]:
    """Borra artefactos OBSOLETOS de la variedad que no esten en `keep`.

    Solo limpia patrones legacy que ya no se generan (`reporte_*`,
    `business_export_*`). Los `Winner_{variety}_*.html` y `.xlsx` se
    ACUMULAN intencionalmente (uno por run) para que el dashboard global
    `reports/index.html` muestre el historial completo de trainings; el
    sidebar agrupado por variedad mantiene la lista navegable.

    Si en el futuro hace falta retencion por antiguedad, agregar un
    parametro `max_age_days` o `keep_last_n` aqui en vez de re-incluir
    Winner en los patterns.
    """
    rdir = Path(reports_dir) if reports_dir else REPORTS_DIR
    keep_set = {Path(p).resolve() for p in keep}
    deleted: list[Path] = []
    patterns = [
        f"reporte_{variety}_*.html",
        f"business_export_{variety}_*.xlsx",
    ]
    for pat in patterns:
        for f in rdir.glob(pat):
            if f.resolve() in keep_set:
                continue
            try:
                f.unlink()
                deleted.append(f)
            except OSError as exc:
                # Limpieza best-effort: un residuo que no se pudo borrar no
                # debe abortar el run, pero conviene dejar rastro.
                _logger.warning("No se pudo borrar el residuo %s: %s", f, exc)
    return deleted
