"""Helpers que envuelven el logging a MLflow de un run individual.

Extraídos de `single_run.py` para separar el "qué se loguea a MLflow" del
cálculo/persistencia local (`_run_outputs.py`) y del orquestador
(`train_model`). Todos son privados del paquete: los consume solo
`single_run.train_model`. Cada uno asume que hay un run MLflow activo
(se invocan DENTRO del `safe_start_run`).
"""

from __future__ import annotations

import argparse
from datetime import datetime

from src.config import REPORTS_DIR, TRAINING_FILE
from src.diagnostics.residuals import write_residual_report
from src.diagnostics.run_metadata import collect_run_metadata
from src.step_06_track.business_validation import (
    BusinessValidation,
    validate_against_business_unit,
)
from src.step_06_track.mlflow_registry import (
    log_artifact,
    log_business_metrics,
    log_metrics,
    log_params,
    log_pipeline,
    set_tags,
)
from src.utils.logger import log_business_audit


def set_initial_run_tags(variety: str, model_type: str, version: int, args) -> None:
    """Tags MLflow basicos al abrir el run. Se invoca DENTRO del start_run."""
    set_tags(
        {
            "variety": variety,
            "tuning": args.tuning,
            "model_type": model_type,
            "version": f"v{version}",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
    )


def log_full_metrics(
    full_metrics_business: dict[str, float],
    full_metrics_h: dict[str, float],
) -> None:
    """Loguea metricas full a MLflow con prefijos `full_business_` / `full_model_`.

    Tags resumen `full_business_mape` / `full_business_r2` filtrables en UI.
    """
    if full_metrics_business:
        log_metrics({f"full_business_{k}": v for k, v in full_metrics_business.items()})
        set_tags(
            {
                "full_business_mape": f"{full_metrics_business.get('mape', float('nan')):.2f}",
                "full_business_r2": f"{full_metrics_business.get('r2', float('nan')):.4f}",
            }
        )
    if full_metrics_h:
        log_metrics({f"full_model_{k}": v for k, v in full_metrics_h.items()})


def log_pipeline_with_signature(final_pipeline, X) -> str | None:
    """log_pipeline con signature inferida desde un sample de X.

    Castea int columns -> float64 SOLO en el sample (no en train data) para
    que la firma sea NaN-safe: el runtime de MLflow promueve int->float si
    encuentra NaN en inferencia y, sin este cast, rompe schema enforcement.

    Devuelve `model_uri` (formato `models:/m-<id>` en MLflow 3.x) para que el
    caller lo propague a `register_model` via `ModelResult.model_uri` — evita
    el warning "no artifacts at artifact path 'model_pipeline'". Devuelve
    None si el run quedo inactivo (log_pipeline absorbio la excepcion).
    """
    X_sample = X.head(min(50, len(X))).copy()
    int_cols = X_sample.select_dtypes(include=["integer"]).columns
    if len(int_cols) > 0:
        X_sample[int_cols] = X_sample[int_cols].astype("float64")
    try:
        y_sample = final_pipeline.predict(X_sample)
    except Exception:
        y_sample = None
    model_info = log_pipeline(
        final_pipeline,
        name="model_pipeline",
        X_sample=X_sample,
        y_sample=y_sample,
    )
    return model_info.model_uri if model_info is not None else None


def log_run_metadata_and_params(
    *,
    variety: str,
    model_type: str,
    args: argparse.Namespace,
    settings: dict,
    X,
    log,
) -> None:
    """Run metadata (git/dataset hash) + params iniciales a MLflow.

    Side effects: set_tags(metadata) y log_params(...). Fallos de
    `collect_run_metadata` no abortan training (trazabilidad rota es un
    finding de auditoria, no bloqueante).
    """
    try:
        metadata_tags = collect_run_metadata(
            training_file=TRAINING_FILE,
            n_rows=int(X.shape[0]),
            n_cols=int(X.shape[1]),
        )
        set_tags(metadata_tags)
    except (OSError, ValueError):
        # OSError: training file / git dir inaccesible.
        # ValueError: hash o parsing fallido.
        # Otros errores (e.g. ImportError) deberian propagar -> NO los
        # tragamos. Trazabilidad rota es un finding de auditoria; loggear
        # con traceback para que el siguiente sysadmin sepa que arreglar.
        log.warning("collect_run_metadata fallo (no aborta training)", exc_info=True)
    log_params(
        {
            "variety": variety,
            "tuning": args.tuning,
            "model_type": model_type,
            "n_trials": settings["n_trials"],
            "final_trials": settings["final_trials"],
            "outer_folds": settings["outer_folds"],
            "inner_folds": settings["inner_folds"],
            "skip_final_tuning": settings["skip_final_tuning"],
            "n_rows": int(X.shape[0]),
            "n_features_input": int(X.shape[1]),
        }
    )


def log_nested_cv_summary(nested_metrics: dict[str, float]) -> None:
    """Loguea nested CV metrics + tags resumen filtrables en MLflow UI."""
    log_metrics(nested_metrics)
    set_tags(
        {
            "r2_mean": f"{nested_metrics['nested_cv_r2_mean']:.4f}",
            "mae_test_mean": f"{nested_metrics['nested_cv_mae_mean']:.4f}",
            "mae_train_mean": f"{nested_metrics.get('nested_cv_mae_train_mean', 0):.4f}",
            "overfit_gap": f"{nested_metrics.get('nested_cv_gap_mean', 0):+.4f}",
        }
    )


def run_business_validation(
    *,
    oof: dict,
    final_pipeline,
    X,
    business_cols,
    variety: str,
    model_type: str,
    args: argparse.Namespace,
    nested_metrics: dict[str, float],
    best_params: dict[str, object],
    run_id: str,
    logger,
    log,
) -> BusinessValidation:
    """Valida en unidad de negocio (KG/JR = KG/JR_H * H-EF), loguea y audita.

    Devuelve el `BusinessValidation` para reuso aguas abajo (full metrics +
    summary + ModelResult).
    """
    log.info("Validando en unidad de negocio (KG/JR)...")
    business_validation = validate_against_business_unit(
        oof=oof,
        final_pipeline=final_pipeline,
        X_full=X,
        business_cols=business_cols,
    )
    log_business_metrics(business_validation)
    log_business_audit(
        logger,
        variety=variety,
        model_type=model_type,
        tuning=args.tuning,
        business_validation=business_validation,
        nested_metrics=nested_metrics,
        best_params=best_params,
        mlflow_run_id=run_id,
    )
    return business_validation


def write_residual_diagnostics(
    *,
    variety: str,
    model_type: str,
    run_name: str,
    run_id: str,
    oof: dict,
    log,
) -> None:
    """Residual diagnostics post-fit: DW + Ljung-Box, BP + White, Shapiro/AD/JB + plots.

    Si rechaza autocorrelacion residual -> el modelo dejo patron temporal ->
    revisar lag features. Si rechaza heteroscedasticidad -> considerar
    log-target o regresion Gamma. Fallo aqui NO aborta training.
    """
    try:
        residual_html = REPORTS_DIR / f"residuals_{variety}_{run_name}.html"
        write_residual_report(
            variety=variety,
            model_type=model_type,
            y_true=oof["y_true"],
            y_pred=oof["y_pred"],
            out_path=residual_html,
            run_id=run_id,
        )
        log_artifact(str(residual_html), artifact_path="residuals")
        log.info(f"Residual diagnostics: {residual_html.name}")
    except Exception:
        # write_residual_report engloba IO (HTML), statsmodels (tests
        # estadisticos) y matplotlib (plots). Cualquier libreria puede
        # lanzar errores propios (LinAlgError, ConvergenceWarning como
        # error, etc.). Mantenemos Exception broad: el diagnostico es
        # opcional y nunca debe bloquear el modelo de produccion.
        log.warning("Residual diagnostics fallo (no aborta training)", exc_info=True)
