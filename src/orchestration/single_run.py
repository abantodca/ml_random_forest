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

Los helpers que envuelven MLflow viven en `_run_mlflow_logging.py`; los de
cálculo/persistencia local en `_run_outputs.py`. Este módulo deja solo el
orquestador `train_model` (+ un helper trivial de serialización OOF).
"""

from __future__ import annotations

import argparse
import math
import time

from src.config import ARTIFACTS_DIR, BACKEND_BUDGET_FRACTION
from src.orchestration._run_mlflow_logging import (
    log_full_metrics,
    log_nested_cv_summary,
    log_pipeline_with_signature,
    log_run_metadata_and_params,
    run_business_validation,
    set_initial_run_tags,
    write_residual_diagnostics,
)
from src.orchestration._run_outputs import (
    build_run_summary,
    full_dataset_metrics,
    persist_pipeline_and_oof_locally,
)
from src.pipeline.build_pipeline import create_preprocessing_pipeline
from src.step_01_load.data_loader import load_business_columns, load_data
from src.step_04_train.tuning import perform_nested_cv
from src.step_05_evaluate.champion import ModelResult
from src.step_06_track.business_validation import BusinessValidation
from src.step_06_track.mlflow_registry import (
    log_artifact,
    log_params,
    next_run_version,
    safe_start_run,
)
from src.utils.logger import PrefixAdapter
from src.utils.sklearn_helpers import dump_json_artifact
from src.variety_config import for_variety


def _build_bv_oof_dump(business_validation: BusinessValidation) -> dict[str, float]:
    """Filtra metrics_oof a floats finitos serializables en JSON.

    `math.isfinite` excluye NaN/Inf que romperian `json.dump` (default
    `allow_nan=False` en utils) y dashboards downstream.
    """
    if business_validation and not business_validation.is_empty():
        return {
            k: float(v)
            for k, v in business_validation.metrics_oof.items()
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
        set_initial_run_tags(variety, model_type, version, args)
        # Trazabilidad: git commit + dataset hash + n_rows. Hace cada run
        # reproducible y permite detectar drift automaticamente cuando
        # dataset_sha256 cambia.
        log_run_metadata_and_params(
            variety=variety,
            model_type=model_type,
            args=args,
            settings=eff,
            X=X,
            log=log,
        )

        log.info("[3/6] Nested CV con Optuna...")
        final_pipeline, best_params, nested_metrics, oof = perform_nested_cv(
            X=X,
            y=y,
            preprocessor=preprocessor,
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
                y_true=oof["y_true"],
                y_pred_oof=oof["y_pred"],
                fundo=X["FUNDO"],
                formato=X["FORMATO"],
            )
            if final_pipeline.conformal_:
                qbf = final_pipeline.conformal_["q_by_fundo"]
                log.info(
                    f"Bandas conformal adjuntas | q90_global="
                    f"{final_pipeline.conformal_['q_global']:.3f} | "
                    f"por_fundo={ {k: round(v, 2) for k, v in qbf.items()} } | "
                    f"known_ff={len(final_pipeline.conformal_['known_ff'])}"
                )
        except Exception as exc:
            logger.warning(f"Conformal metadata fallo (se omite): {exc}")
            final_pipeline.conformal_ = None

        local_pipeline, oof_arr_path = persist_pipeline_and_oof_locally(
            final_pipeline=final_pipeline,
            oof=oof,
            variety=variety,
            run_name=run_name,
            log=log,
        )

        log.info("[4/6] MLflow logging...")
        log_nested_cv_summary(nested_metrics)
        log_params(best_params)

        # ---- Validacion en unidad de negocio (KG/JR = KG/JR_H * H-EF) ----
        business_validation = run_business_validation(
            oof=oof,
            final_pipeline=final_pipeline,
            X=X,
            business_cols=business_cols,
            variety=variety,
            model_type=model_type,
            args=args,
            nested_metrics=nested_metrics,
            best_params=best_params,
            run_id=run.info.run_id,
            logger=logger,
            log=log,
        )

        # ---- Metricas en DATASET COMPLETO (refit + predict all) ----
        # "Aplicacion Total": tarjeta del dashboard ejecutivo. Es la perspectiva
        # del modelo de produccion aplicado a toda la historia disponible.
        full_metrics_business, full_metrics_h, _pred_h_full = full_dataset_metrics(
            final_pipeline,
            X,
            y,
            business_validation,
            logger=logger,
        )
        log_full_metrics(full_metrics_business, full_metrics_h)

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
        model_uri = log_pipeline_with_signature(final_pipeline, X)
        # MLflow 3.x guarda log_model en LoggedModel separado (visible en
        # tab "Models" del experimento). Subimos tambien el .joblib y el OOF
        # como artifacts tradicionales para que sean visibles bajo la
        # pestaña "Artifacts" del run, tanto en MLflow local como en
        # produccion (Fargate). Sin esto la pestaña aparece "No Artifacts
        # Recorded" aunque el modelo si este registrado.
        log_artifact(str(local_pipeline), artifact_path="pipeline")
        log_artifact(str(oof_arr_path), artifact_path="oof")

        write_residual_diagnostics(
            variety=variety,
            model_type=model_type,
            run_name=run_name,
            run_id=run.info.run_id,
            oof=oof,
            log=log,
        )

        elapsed = time.perf_counter() - t0
        bv_oof_dump = _build_bv_oof_dump(business_validation)
        # Summary local versionado por run_name. Cada run histórico (xgb_v19,
        # xgb_v20, ...) conserva su propio JSON sin sobrescribirse.
        summary = build_run_summary(
            variety=variety,
            model_type=model_type,
            run_id=run.info.run_id,
            nested_metrics=nested_metrics,
            bv_oof_dump=bv_oof_dump,
            full_metrics_business=full_metrics_business,
            full_metrics_h=full_metrics_h,
            best_params=best_params,
            local_pipeline=local_pipeline,
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
