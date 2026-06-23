"""Helpers de cálculo y persistencia local de un run (sin MLflow).

Extraídos de `single_run.py` para separar el "qué se calcula/guarda en disco"
del "qué se loguea a MLflow" (`_run_mlflow_logging.py`) y del orquestador
(`train_model`). Todos son privados del paquete: los consume solo
`single_run.train_model`.
"""

from __future__ import annotations

from datetime import datetime

import joblib
import numpy as np

from src.config import ARTIFACTS_DIR
from src.step_05_evaluate.metrics import calculate_regression_metrics
from src.step_06_track.business_validation import BusinessValidation


def full_dataset_metrics(
    final_pipeline,
    X,
    y,
    business_validation: BusinessValidation,
    logger=None,
) -> tuple[dict[str, float], dict[str, float], np.ndarray | None]:
    """Computa metricas sobre el DATASET COMPLETO (refit + predict all).

    Returns
    -------
    full_metrics_business : KG/JR (unidad de negocio). Vacio si faltan KG/JR/H-EF.
    full_metrics_h        : KG/JR_H (unidad del modelo).
    pred_h_full           : array de predicciones en KG/JR_H sobre todo X.
                            None si la prediccion fallo.

    Nota: in-sample es OPTIMISTA (modelo predice lo que entreno). Se usa
    como sanity check del modelo de produccion y para el panel de
    "Aplicacion Total" del dashboard, NO para decidir despliegue.
    """
    try:
        pred_h_full = np.asarray(final_pipeline.predict(X), dtype=float)
    except Exception:
        # Predict puede fallar por dtype mismatch (sklearn), feature-name
        # mismatch (xgboost/lightgbm), o ValueError numerico. La jerarquia
        # exacta varia por libreria; mantenemos Exception y logueamos
        # traceback para diagnostico.
        if logger is not None:
            logger.warning(
                "full_metrics: final_pipeline.predict(X) fallo; "
                "se omite tarjeta 'Aplicacion Total'",
                exc_info=True,
            )
        pred_h_full = None

    full_metrics_h: dict[str, float] = {}
    if pred_h_full is not None:
        full_metrics_h = calculate_regression_metrics(
            np.asarray(y, dtype=float),
            pred_h_full,
        )

    # KG/JR (business): reusamos las metricas in-sample que ya calcula
    # validate_against_business_unit (refit + predict all + multiplicar por H-EF).
    full_metrics_business = dict(business_validation.metrics_insample or {})

    return full_metrics_business, full_metrics_h, pred_h_full


def persist_pipeline_and_oof_locally(
    *,
    final_pipeline,
    oof: dict,
    variety: str,
    run_name: str,
    log,
):
    """Persiste pipeline (.joblib) y OOF (.npz) a disco ANTES de tocar MLflow.

    Devuelve (local_pipeline_path, oof_arr_path).

    Si MLflow falla a mitad del logging (e.g. run marcado deleted
    externamente), el modelo entrenado ya esta a salvo en disco -- 1+h de
    tuning no se pierden por un fallo de tracking. OOF arrays se preservan
    para GAMM Phase 0 (corrector de residuos) y post-mortems sin
    re-entrenamiento.
    """
    # unlink primero (2026-06-13): los nombres `_vN` pueden COLISIONAR con
    # archivos de corridas previas (la version se recalcula tras borrar runs
    # perdedores) y el bind-mount puede tener archivos de OTRO uid (task
    # train --user vs docker compose run directo como mluser). Sobrescribir
    # un archivo ajeno da PermissionError; desvincularlo del directorio
    # (escribible) y crear uno propio no.
    local_pipeline = ARTIFACTS_DIR / f"final_pipeline_{variety}_{run_name}.joblib"
    local_pipeline.unlink(missing_ok=True)
    joblib.dump(final_pipeline, local_pipeline)
    log.info(f"Pipeline persistido localmente: {local_pipeline.name}")

    oof_arr_path = ARTIFACTS_DIR / f"oof_{variety}_{run_name}.npz"
    oof_arr_path.unlink(missing_ok=True)
    np.savez(oof_arr_path, y_true=oof["y_true"], y_pred=oof["y_pred"])
    log.info(f"OOF persistido localmente: {oof_arr_path.name}")

    return local_pipeline, oof_arr_path


def build_run_summary(
    *,
    variety: str,
    model_type: str,
    run_id: str,
    nested_metrics: dict[str, float],
    bv_oof_dump: dict[str, float],
    full_metrics_business: dict[str, float],
    full_metrics_h: dict[str, float],
    best_params: dict[str, object],
    local_pipeline,
    elapsed: float,
) -> dict:
    """Pure data construction: dict serializable del summary del run."""
    return {
        "variety": variety,
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mlflow_run_id": run_id,
        "metrics": {k: float(v) for k, v in nested_metrics.items()},
        "business_metrics_oof": bv_oof_dump,
        "full_metrics_business": {k: float(v) for k, v in full_metrics_business.items()},
        "full_metrics_model": {k: float(v) for k, v in full_metrics_h.items()},
        "best_params": {
            k: (float(v) if isinstance(v, (int, float)) else v) for k, v in best_params.items()
        },
        "artifacts": {"pipeline": str(local_pipeline)},
        "elapsed_seconds": round(elapsed, 2),
    }
