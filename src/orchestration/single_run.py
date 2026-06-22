"""Entrenamiento de UN modelo (xgb|lgb) para UNA variedad.

Devuelve un `ModelResult` listo para `select_champion`. Cada llamada:
  - Abre su propio MLflow run dentro del experimento de la variedad.
  - Corre nested CV + Optuna independiente.
  - Loguea metricas Train/Test/Full (model y business units), params, tags.
  - Persiste pipeline + summary JSON.

NO genera HTML por modelo: el dashboard ejecutivo se construye una sola vez
en `variety_runner` despues de elegir campeon (`Winner_{variedad}.html`).
NO selecciona ni registra el campeon: eso vive en `variety_runner` para
poder elegir entre todos los modelos al final.
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime

import joblib
import numpy as np

from src.config import (
    ARTIFACTS_DIR,
    BACKEND_BUDGET_FRACTION,
    REPORTS_DIR,
    TRAINING_FILE,
)
from src.diagnostics.residuals import write_residual_report
from src.diagnostics.run_metadata import collect_run_metadata
from src.pipeline.build_pipeline import create_preprocessing_pipeline
from src.step_01_load.data_loader import load_business_columns, load_data
from src.step_04_train.tuning import perform_nested_cv
from src.step_05_evaluate.champion import ModelResult
from src.step_05_evaluate.metrics import calculate_regression_metrics
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
    next_run_version,
    safe_start_run,
    set_tags,
)
from src.utils.logger import PrefixAdapter, log_business_audit
from src.utils.sklearn_helpers import dump_json_artifact
from src.variety_config import for_variety


def _full_dataset_metrics(
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
                "se omite tarjeta 'Aplicacion Total'", exc_info=True,
            )
        pred_h_full = None

    full_metrics_h: dict[str, float] = {}
    if pred_h_full is not None:
        full_metrics_h = calculate_regression_metrics(
            np.asarray(y, dtype=float), pred_h_full,
        )

    # KG/JR (business): reusamos las metricas in-sample que ya calcula
    # validate_against_business_unit (refit + predict all + multiplicar por H-EF).
    full_metrics_business = dict(business_validation.metrics_insample or {})

    return full_metrics_business, full_metrics_h, pred_h_full


def _set_initial_run_tags(variety: str, model_type: str, version: int, args) -> None:
    """Tags MLflow basicos al abrir el run. Se invoca DENTRO del start_run."""
    set_tags({
        "variety": variety,
        "tuning": args.tuning,
        "model_type": model_type,
        "version": f"v{version}",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    })


def _log_full_metrics(
    full_metrics_business: dict[str, float],
    full_metrics_h: dict[str, float],
) -> None:
    """Loguea metricas full a MLflow con prefijos `full_business_` / `full_model_`.

    Tags resumen `full_business_mape` / `full_business_r2` filtrables en UI.
    """
    if full_metrics_business:
        log_metrics({f"full_business_{k}": v for k, v in full_metrics_business.items()})
        set_tags({
            "full_business_mape": f"{full_metrics_business.get('mape', float('nan')):.2f}",
            "full_business_r2":   f"{full_metrics_business.get('r2',   float('nan')):.4f}",
        })
    if full_metrics_h:
        log_metrics({f"full_model_{k}": v for k, v in full_metrics_h.items()})


def _log_pipeline_with_signature(final_pipeline, X) -> str | None:
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
        final_pipeline, name="model_pipeline",
        X_sample=X_sample, y_sample=y_sample,
    )
    return model_info.model_uri if model_info is not None else None


def _build_run_summary(
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
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in best_params.items()
        },
        "artifacts": {"pipeline": str(local_pipeline)},
        "elapsed_seconds": round(elapsed, 2),
    }


def _log_run_metadata_and_params(
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
        log.warning("collect_run_metadata fallo (no aborta training)",
                    exc_info=True)
    log_params({
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
    })


def _persist_pipeline_and_oof_locally(
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


def _log_nested_cv_summary(nested_metrics: dict[str, float]) -> None:
    """Loguea nested CV metrics + tags resumen filtrables en MLflow UI."""
    log_metrics(nested_metrics)
    set_tags({
        "r2_mean": f"{nested_metrics['nested_cv_r2_mean']:.4f}",
        "mae_test_mean": f"{nested_metrics['nested_cv_mae_mean']:.4f}",
        "mae_train_mean": f"{nested_metrics.get('nested_cv_mae_train_mean', 0):.4f}",
        "overfit_gap": f"{nested_metrics.get('nested_cv_gap_mean', 0):+.4f}",
    })


def _run_business_validation(
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
        oof=oof, final_pipeline=final_pipeline,
        X_full=X, business_cols=business_cols,
    )
    log_business_metrics(business_validation)
    log_business_audit(
        logger,
        variety=variety, model_type=model_type, tuning=args.tuning,
        business_validation=business_validation,
        nested_metrics=nested_metrics, best_params=best_params,
        mlflow_run_id=run_id,
    )
    return business_validation


def _write_residual_diagnostics(
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
            variety=variety, model_type=model_type,
            y_true=oof["y_true"], y_pred=oof["y_pred"],
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


def _build_bv_oof_dump(business_validation: BusinessValidation) -> dict[str, float]:
    """Filtra metrics_oof a floats finitos serializables en JSON.

    `math.isfinite` excluye NaN/Inf que romperian `json.dump` (default
    `allow_nan=False` en utils) y dashboards downstream.
    """
    if business_validation and not business_validation.is_empty():
        return {
            k: float(v) for k, v in business_validation.metrics_oof.items()
            if isinstance(v, (int, float)) and math.isfinite(v)
        }
    return {}


def train_model(
    variety: str,
    model_type: str,
    args: argparse.Namespace,
    settings: dict,
    logger,
) -> ModelResult:
    """Entrena UN modelo (xgb|lgb) para UNA variedad. Devuelve ModelResult."""
    log = PrefixAdapter(logger, prefix=f"[{variety}/{model_type}]")
    logger.info("-" * 78)
    logger.info(f"# {variety} / {model_type}")
    logger.info("-" * 78)

    t0 = time.perf_counter()

    # Config POR VARIEDAD (P0.2): overrides explicitos (meses de temporada,
    # umbral KNN, rare_min_count). Para variedades sin overrides es
    # passthrough de los defaults globales (POP queda identico).
    variety_cfg = for_variety(variety)

    # Presupuesto efectivo: override POR BACKEND (config.BACKEND_BUDGET_FRACTION).
    # Por defecto el dict esta vacio -> todos los backends corren al perfil
    # completo (frac=1.0). inner_folds intacto (barato; reducirlo degrada el
    # tuning). Escala correctamente en smoke/dev/prod/prod_xl.
    frac = BACKEND_BUDGET_FRACTION.get(model_type, 1.0)
    eff = dict(settings)
    if frac < 1.0:
        eff["n_trials"] = max(5, int(settings["n_trials"] * frac))
        eff["final_trials"] = max(3, int(settings["final_trials"] * frac))
        eff["outer_folds"] = max(2, math.ceil(settings["outer_folds"] * frac))
        log.info(
            f"Presupuesto reducido x{frac} (backend de referencia): "
            f"outer={eff['outer_folds']} trials={eff['n_trials']} "
            f"final_trials={eff['final_trials']} (inner={eff['inner_folds']} intacto)"
        )

    log.info(f"[1/6] Cargando datos | hoja={variety}")
    X, y = load_data(sheet=variety, rare_min_count=variety_cfg.rare_min_count)
    business_cols = load_business_columns(sheet=variety)  # KG/JR + H-EF alineadas con (X,y)

    log.info("[2/6] Construyendo preprocesador...")
    preprocessor = create_preprocessing_pipeline(variety_cfg)

    # Run name versionado: el experimento ya identifica la variedad, asi que
    # el run solo necesita decir el modelo y su version (xgb_v1, xgb_v2, ...).
    # `experiment_prefix` viene vacio por default desde config.py -> el
    # experimento es el nombre de la variedad (e.g. "POP").
    experiment_name = f"{args.experiment_prefix}{variety}"
    version = next_run_version(experiment_name, model_type)
    run_name = f"{model_type}_v{version}"

    with safe_start_run(run_name=run_name) as run:
        _set_initial_run_tags(variety, model_type, version, args)
        # Trazabilidad: git commit + dataset hash + n_rows. Hace cada run
        # reproducible y permite detectar drift automaticamente cuando
        # dataset_sha256 cambia.
        _log_run_metadata_and_params(
            variety=variety, model_type=model_type, args=args,
            settings=eff, X=X, log=log,
        )

        log.info("[3/6] Nested CV con Optuna...")
        final_pipeline, best_params, nested_metrics, oof = perform_nested_cv(
            X=X, y=y, preprocessor=preprocessor,
            n_trials=eff["n_trials"],
            final_trials=eff["final_trials"],
            model_type=model_type,
            outer_folds=eff["outer_folds"],
            inner_folds=eff["inner_folds"],
            skip_final_tuning=eff["skip_final_tuning"],
            inner_cv_n_jobs=eff.get("inner_cv_n_jobs", -1),
            logger=logger,
            variety_cfg=variety_cfg,
        )

        # Bandas conformal por fundo + cold-start (2026-06-11): calibradas
        # con los residuos OOF de ESTE nested CV y adjuntas al pipeline como
        # atributo pickle-safe. La API las usa para kghora_lo/hi en lugar de
        # la heuristica ±1.96·std del ensemble (cobertura real << nominal).
        try:
            from src.step_05_evaluate.conformal_bands import build_conformal_metadata
            final_pipeline.conformal_ = build_conformal_metadata(
                y_true=oof["y_true"], y_pred_oof=oof["y_pred"],
                fundo=X["FUNDO"], formato=X["FORMATO"],
            )
            if final_pipeline.conformal_:
                qbf = final_pipeline.conformal_["q_by_fundo"]
                log.info(
                    f"Bandas conformal adjuntas | q90_global="
                    f"{final_pipeline.conformal_['q_global']:.3f} | "
                    f"por_fundo={ {k: round(v, 2) for k, v in qbf.items()} } | "
                    f"known_ff={len(final_pipeline.conformal_['known_ff'])}"
                )
        except Exception as exc:  # noqa: BLE001 — metadata opcional, jamas rompe el run
            logger.warning(f"Conformal metadata fallo (se omite): {exc}")
            final_pipeline.conformal_ = None

        local_pipeline, oof_arr_path = _persist_pipeline_and_oof_locally(
            final_pipeline=final_pipeline, oof=oof,
            variety=variety, run_name=run_name, log=log,
        )

        log.info("[4/6] MLflow logging...")
        _log_nested_cv_summary(nested_metrics)
        log_params(best_params)

        # ---- Validacion en unidad de negocio (KG/JR = KG/JR_H * H-EF) ----
        business_validation = _run_business_validation(
            oof=oof, final_pipeline=final_pipeline,
            X=X, business_cols=business_cols,
            variety=variety, model_type=model_type, args=args,
            nested_metrics=nested_metrics, best_params=best_params,
            run_id=run.info.run_id, logger=logger, log=log,
        )

        # ---- Metricas en DATASET COMPLETO (refit + predict all) ----
        # "Aplicacion Total": tarjeta del dashboard ejecutivo. Es la perspectiva
        # del modelo de produccion aplicado a toda la historia disponible.
        full_metrics_business, full_metrics_h, _pred_h_full = _full_dataset_metrics(
            final_pipeline, X, y, business_validation, logger=logger,
        )
        _log_full_metrics(full_metrics_business, full_metrics_h)

        # NOTE: el Excel multi-hoja YA NO se genera aqui. Se genera UNA SOLA
        # vez en `variety_runner` para el modelo CAMPEON, en
        # `reports/Winner_{variety}.xlsx` (junto al HTML del dashboard).
        # Razon: evitar archivos residuales de modelos perdedores.

        # best_params como artifact JSON (precision sin truncado de MLflow params).
        # Path versionado por run_name (xgb_v20, lgb_v3, ...) para que el archivo
        # local de v19 NO sea sobrescrito por v20. MLflow ya tiene historial
        # versionado por run_id, esto agrega trazabilidad fuera de MLflow.
        params_path = dump_json_artifact(
            ARTIFACTS_DIR / f"best_params_{variety}_{run_name}.json",
            best_params,
        )
        log_artifact(params_path, artifact_path="hyperparameters")

        log.info("[5/6] Persistiendo pipeline en MLflow...")
        model_uri = _log_pipeline_with_signature(final_pipeline, X)
        # MLflow 3.x guarda log_model en LoggedModel separado (visible en
        # tab "Models" del experimento). Subimos tambien el .joblib y el OOF
        # como artifacts tradicionales para que sean visibles bajo la
        # pestaña "Artifacts" del run, tanto en MLflow local como en
        # produccion (Fargate). Sin esto la pestaña aparece "No Artifacts
        # Recorded" aunque el modelo si este registrado.
        log_artifact(str(local_pipeline), artifact_path="pipeline")
        log_artifact(str(oof_arr_path), artifact_path="oof")

        _write_residual_diagnostics(
            variety=variety, model_type=model_type, run_name=run_name,
            run_id=run.info.run_id, oof=oof, log=log,
        )

        elapsed = time.perf_counter() - t0
        bv_oof_dump = _build_bv_oof_dump(business_validation)
        # Summary local versionado por run_name. Cada run histórico (xgb_v19,
        # xgb_v20, ...) conserva su propio JSON sin sobrescribirse.
        summary = _build_run_summary(
            variety=variety, model_type=model_type, run_id=run.info.run_id,
            nested_metrics=nested_metrics, bv_oof_dump=bv_oof_dump,
            full_metrics_business=full_metrics_business,
            full_metrics_h=full_metrics_h,
            best_params=best_params, local_pipeline=local_pipeline,
            elapsed=elapsed,
        )
        summary_path = dump_json_artifact(
            ARTIFACTS_DIR / f"run_summary_{variety}_{run_name}.json",
            summary,
        )
        log_artifact(summary_path)

        log.info(
            f"[6/6] DONE | "
            f"MAE_test={nested_metrics['nested_cv_mae_mean']:.4f} | "
            f"MAE_train={nested_metrics.get('nested_cv_mae_train_mean', 0):.4f} | "
            f"gap={nested_metrics.get('nested_cv_gap_mean', 0):+.4f} | "
            f"R2={nested_metrics['nested_cv_r2_mean']:.4f} | "
            f"FullMAPE={full_metrics_business.get('mape', float('nan')):.2f}% | "
            f"dt={elapsed:.1f}s | run_id={run.info.run_id[:12]}"
        )

        result = ModelResult(
            model_type=model_type,
            metrics=dict(nested_metrics),
            best_params=dict(best_params),
            mlflow_run_id=run.info.run_id,
            pipeline_path=str(local_pipeline),
            elapsed_seconds=round(elapsed, 2),
            business_metrics_oof=bv_oof_dump or None,
            full_metrics=full_metrics_business or None,
            business_validation=business_validation,
            full_metrics_h=full_metrics_h or None,
            oof_y_true=oof["y_true"],
            oof_y_pred=oof["y_pred"],
            model_uri=model_uri,
        )

    # liberar referencias grandes ANTES del cleanup global
    del X, y, preprocessor, final_pipeline, best_params, nested_metrics, oof
    del business_cols
    return result
