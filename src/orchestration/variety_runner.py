"""Entrenamiento de UNA variedad: todos los modelos + champion + dashboard.

Logica:
  1. Setea el experimento MLflow dedicado de la variedad (sin timestamp).
  2. Itera modelos llamando `train_model` y acumula `ModelResult`s.
  3. Selecciona campeon con LEX-ORDER (en `champion.select_champion`):
        a. gate de |gap| Train-Test (restriccion anti-overfit, no objetivo),
        b. menor MAPE OOF de negocio (generalizacion honesta),
        c. menor tiempo (eficiencia ante empate practico).
  4. Renderiza UN dashboard ejecutivo: `reports/Winner_{variety}.html`.
  5. Borra los HTML residuales (per-modelo timestamped) de la variedad.
  6. Tagea el run ganador como `is_champion=true` y el resto como false.
  7. Registra UNICAMENTE el campeon en MLflow Model Registry (si --register).
  8. Persiste un `variety_summary_<NAME>.json` con todo el contexto.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

from src.config import ARTIFACTS_DIR, CHAMPION_MAX_GAP, CHAMPION_MAX_MAPE, REPORTS_DIR
from src.orchestration.cleanup import cleanup_residual_reports, cleanup_state
from src.orchestration.single_run import train_model
from src.step_01_load.data_loader import load_business_columns, load_data
from src.step_05_evaluate.champion import (
    ModelResult,
    champion_summary,
    select_champion,
)
from src.step_05_evaluate.html.winner_dashboard import render_winner_dashboard
from src.step_06_track.business_export import export_business_excel
from src.step_06_track.mlflow_registry import register_model, set_experiment
from src.utils.cloudwatch_metrics import emit_mape_metric
from src.utils.logger import setup_logging
from src.utils.sklearn_helpers import dump_json_artifact
from src.variety_config import for_variety


def train_variety(
    variety: str,
    model_types: list[str],
    args: argparse.Namespace,
    settings: dict,
    logger=None,
) -> dict:
    """Entrena TODOS los modelos para UNA variedad y elige campeon.

    Si `logger` es None, crea uno DEDICADO que escribe a
    `logs/variety_<NAME>.log` (modo paralelo: cada variedad su archivo,
    sin interleaving).
    """
    if logger is None:
        logger = setup_logging(
            name=f"variety.{variety}",
            log_file=f"variety_{variety}.log",
        )

    logger.info("#" * 78)
    logger.info(f"# VARIEDAD INICIO: {variety} | modelos={model_types} | pid={os.getpid()}")
    logger.info("#" * 78)

    experiment_name = f"{args.experiment_prefix}{variety}"
    set_experiment(experiment_name)
    logger.info(f"[{variety}] MLflow experiment: '{experiment_name}'")

    t_var = time.perf_counter()
    results, failures = _run_model_loop(
        variety=variety,
        model_types=model_types,
        args=args,
        settings=settings,
        logger=logger,
    )

    champion_decision: dict | None = None
    registered_name: str | None = None
    winner_report_path: str | None = None
    winner_excel_path: str | None = None
    if results:
        champion = select_champion(results)
        champion_decision = champion_summary(results, champion)
        # Salvaguarda anti-walkover (2026-06-13): si algun backend FALLO, el
        # "campeon" no gano una competencia completa — gano por default. Se
        # marca explicitamente (log + decision JSON) para que nadie confunda
        # un walkover con una victoria; el run tag queda via set_tags abajo.
        walkover = bool(failures)
        if walkover:
            champion_decision["walkover"] = True
            champion_decision["failed_backends"] = list(failures)
            logger.warning(
                f"[{variety}] CAMPEON POR WALKOVER: {champion.model_type} gano "
                f"con {len(results)}/{len(model_types)} backends en pie "
                f"(fallaron: {failures}). La comparacion NO es valida como "
                "competencia completa — revisar los FALLO de arriba."
            )
        logger.info(
            f"[{variety}] CAMPEON: {champion.model_type} | "
            f"|gap|={champion.abs_gap:.4f} | "
            f"MAPE_oof={champion.oof_mape:.2f}% | "
            f"MAPE_full={champion.full_mape:.2f}% | "
            f"dt={champion.elapsed_seconds:.1f}s | "
            f"composite_aux={champion.composite_score:.4f}"
        )
        logger.info(f"[{variety}] Decision: {champion_decision.get('justification', '')}")

        args_register = _apply_quality_gate(champion, args, variety, logger)
        emit_mape_metric(variety=variety, mape_value=champion.oof_mape)

        _delete_loser_runs(results, champion, variety, logger)

        # Cargar (X, y, business_cols) UNA sola vez. Antes se recargaba 2-3
        # veces (Excel + Dashboard + render). single_run las libera al
        # terminar; el costo aqui es leer 1 hoja del Excel (~10k filas).
        try:
            X_full, _, business_full = _load_variety_inputs(variety)
        except Exception:  # defensive: pandas/openpyxl puede lanzar varios tipos en IO
            logger.exception(f"[{variety}] no se pudo recargar data para outputs ejecutivos")
            X_full = None
            business_full = None

        # `run_label` con segundos para evitar colision si dos runs corren en
        # el mismo minuto (smoke tests <60s). Comparte el sufijo entre
        # Winner_<variety>_<run_label>.html y .xlsx para ligar visualmente
        # ambos artefactos del mismo training.
        run_label = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # ---- Excel del CAMPEON aplicado a la data real ----
        winner_excel_path = _export_winner_excel(
            champion=champion,
            variety=variety,
            logger=logger,
            X_raw=X_full,
            business_cols=business_full,
            run_label=run_label,
        )

        # ---- Dashboard ejecutivo por-run (Winner_{variety}_{run_label}.html) ----
        winner_report_path = _render_winner_and_cleanup(
            variety=variety,
            results=results,
            champion=champion,
            champion_decision=champion_decision,
            X_full=X_full,
            winner_excel_path=winner_excel_path,
            run_label=run_label,
            logger=logger,
        )

        _tag_champion(
            champion,
            results,
            variety,
            champion_decision,
            logger,
            winner_report_path=winner_report_path,
            winner_excel_path=winner_excel_path,
        )

        if args_register:
            registered_name = _register_champion(
                champion,
                variety,
                args,
                logger,
                winner_report_path=winner_report_path,
            )

        _write_global_dashboard_index(variety, logger)
        _log_final_summary(
            variety=variety,
            champion=champion,
            registered_name=registered_name,
            logger=logger,
        )

    elapsed = time.perf_counter() - t_var
    summary = _build_summary(
        variety=variety,
        experiment_name=experiment_name,
        model_types=model_types,
        failures=failures,
        registered_name=registered_name,
        winner_report_path=winner_report_path,
        winner_excel_path=winner_excel_path,
        elapsed=elapsed,
        champion_decision=champion_decision,
        results=results,
    )
    dump_json_artifact(
        ARTIFACTS_DIR / f"variety_summary_{variety}.json",
        summary,
    )
    return summary


def _run_model_loop(
    variety: str,
    model_types: list[str],
    args: argparse.Namespace,
    settings: dict,
    logger,
) -> tuple[list[ModelResult], list[str]]:
    """Itera modelos, llama `train_model` y acumula resultados / fallos.

    Cleanup post-modelo (memoria/MLflow run abierto) se hace en `finally`
    para que un FALLO no deje runs colgados.
    """
    results: list[ModelResult] = []
    failures: list[str] = []
    for model_type in model_types:
        try:
            results.append(train_model(variety, model_type, args, settings, logger))
        except (
            Exception
        ):  # defensive: train_model envuelve logica heterogenea (model fit, MLflow, IO)
            logger.exception(f"[{variety}/{model_type}] FALLO")
            failures.append(model_type)
        finally:
            cleanup_state(logger, f"post {variety}/{model_type}")
    return results, failures


def _apply_quality_gate(
    champion: ModelResult,
    args: argparse.Namespace,
    variety: str,
    logger,
) -> bool:
    """Decide si el campeon se registra segun MAPE_oof y |gap|.

    - MAPE_oof = calidad OPERATIVA real (lo que ve el negocio).
      Si supera threshold -> BLOQUEA registro (modelo inutil).
    - gap = sintoma DIAGNOSTICO de overfitting (diferencia train-test).
      Si supera threshold -> WARNING pero NO bloquea registro:
      un arbol boosted con gap alto puede igual generalizar bien
      (memoriza train por diseno; lo importante es MAE_test honesto).

    Devuelve True si pasa el gate operativo (MAPE_oof OK) y respeta el
    flag `--register-model`; False si MAPE_oof supera el threshold.

    Los runs `--tuning smoke` NUNCA registran: son sanity checks de ~1min
    con 5 trials / 2 folds, no modelos tuneados. Antes un smoke podia
    registrar (y promover) una version casi sin tunear en el Registry.
    """
    if args.tuning == "smoke":
        logger.info(
            f"[{variety}] Registro OMITIDO: tuning=smoke es un sanity check "
            f"(5 trials, 2 folds), no un modelo de produccion. Usa "
            f"--tuning dev|prod|prod_xl para registrar en Model Registry."
        )
        return False

    # Guard de registro (incidente 2026-06-13: un dev EXANTE_MODE=1 paso el
    # gate y registro v2 experimental — la API sirve la ULTIMA version).
    # Releemos el modulo (no import top-level) para honrar el env del run.
    from src import config as _cfg

    if not _cfg.REGISTER_ENABLED:
        logger.info(
            f"[{variety}] Registro OMITIDO: REGISTER_ENABLED=0 (guard "
            f"explicito para corridas experimentales)."
        )
        return False
    if _cfg.EXANTE_MODE:
        logger.warning(
            f"[{variety}] Registro BLOQUEADO: EXANTE_MODE=1 es un flag "
            f"EXPERIMENTAL — su campeon no debe llegar al Model Registry "
            f"(la API serviria la ultima version). Metricas y reportes del "
            f"run quedan intactos en MLflow Experiments."
        )
        return False

    mape_ok = champion.oof_mape <= CHAMPION_MAX_MAPE
    gap_ok = (champion.abs_gap * 100) <= CHAMPION_MAX_GAP

    if not mape_ok:
        logger.warning(
            f"[{variety}] CAMPEON RECHAZADO por calidad operativa | "
            f"MAPE_oof={champion.oof_mape:.2f}% supera threshold "
            f"{CHAMPION_MAX_MAPE}%. El modelo NO se registra en Model Registry "
            f"(predice mal en datos OOF -> inutilizable en produccion). "
            f"El run SI esta en MLflow Experiments (run_id={champion.mlflow_run_id[:8]}...) "
            f"con todos sus artifacts para diagnostico."
        )
        return False
    if not gap_ok:
        logger.warning(
            f"[{variety}] CAMPEON registra con WARNING de overfitting | "
            f"gap={champion.abs_gap * 100:.2f}pp supera threshold "
            f"{CHAMPION_MAX_GAP}pp pero MAPE_oof={champion.oof_mape:.2f}% "
            f"(<={CHAMPION_MAX_MAPE}%) confirma calidad operativa OK. "
            f"El gap es diagnostico de memoria del train, NO afecta predicciones "
            f"OOF/produccion. Modelo aprobado para Registry."
        )
        return args.register_model
    logger.info(
        f"[{variety}] CAMPEON pasa quality gate | "
        f"MAPE_oof={champion.oof_mape:.2f}% (max={CHAMPION_MAX_MAPE}%) | "
        f"gap={champion.abs_gap * 100:.2f}pp (max={CHAMPION_MAX_GAP}pp). "
        f"Registra en MLflow Model Registry."
    )
    return args.register_model


def _delete_loser_runs(
    results: list[ModelResult],
    champion: ModelResult,
    variety: str,
    logger,
) -> None:
    """Elimina runs de modelos NO campeon de MLflow Experiments.

    Soft delete via API: MLflow marca lifecycle_stage='deleted' en
    Postgres (recuperable con client.restore_run). El ModelResult sigue
    vivo en memoria asi que el dashboard HTML/Excel comparativos NO se
    afectan.
    Limpiar mlflow_run_id="" en el ModelResult evita que _tag_champion
    intente operar sobre runs que ya no existen.
    """
    losers = [r for r in results if r is not champion]
    if not losers:
        return
    client = mlflow.tracking.MlflowClient()
    deleted_count = 0
    for loser in losers:
        if not loser.mlflow_run_id:
            continue
        try:
            client.delete_run(loser.mlflow_run_id)
            loser.mlflow_run_id = ""
            deleted_count += 1
        except MlflowException:
            logger.exception(f"[{variety}] error eliminando run de {loser.model_type}")
    if deleted_count > 0:
        logger.info(
            f"[{variety}] Eliminados {deleted_count}/{len(losers)} run(s) "
            f"no campeon de MLflow (en .trash, recuperables): "
            f"{[lo.model_type for lo in losers]}"
        )


def _render_winner_and_cleanup(
    variety: str,
    results: list[ModelResult],
    champion: ModelResult,
    champion_decision: dict,
    X_full,
    winner_excel_path: str | None,
    run_label: str,
    logger,
) -> str | None:
    """Renderiza Winner_{variety}_{run_label}.html y limpia obsoletos.

    Devuelve el path como string del dashboard generado, o None si X_full
    no esta disponible o el render fallo.
    """
    if X_full is None:
        return None
    try:
        winner_path = render_winner_dashboard(
            variety=variety,
            results=results,
            champion=champion,
            decision=champion_decision,
            excel_path=winner_excel_path,
            X_raw=X_full,
            run_label=run_label,
        )
        logger.info(f"[{variety}] Winner dashboard: {winner_path}")
        # Limpia obsoletos (reporte_*, business_export_*) pero mantiene
        # Winners por-run y residuals: el index global los lista todos.
        keep = [winner_path]
        if winner_excel_path:
            keep.append(winner_excel_path)
        deleted = cleanup_residual_reports(variety=variety, keep=keep)
        if deleted:
            logger.info(f"[{variety}] Limpieza reports/: {len(deleted)} archivo(s) borrado(s)")
        return str(winner_path)
    except Exception:  # defensive: render HTML mezcla plotly, jinja, IO; varios tipos posibles
        logger.exception(f"[{variety}] no se pudo construir Winner dashboard")
        return None


def _write_global_dashboard_index(variety: str, logger) -> None:
    """Regenera el dashboard global (`reports/index.html`).

    Se llama DESPUES del register MLflow para que el sidebar liste el
    Winner recien generado y la pill 'Latest' apunte a el. En modo
    paralelo (varias variedades simultaneas) hay race condition al
    reescribir el mismo archivo: aceptable porque ambos procesos
    escanean reports/ y el ultimo en escribir ve todos los archivos.
    Try/except para no abortar la variedad si write_dashboard falla.
    """
    try:
        from src.diagnostics.dashboard_index import write_dashboard

        write_dashboard(REPORTS_DIR)
    except Exception:  # defensive: write_dashboard mezcla scan FS + jinja + IO
        logger.exception(f"[{variety}] no se pudo regenerar reports/index.html")


def _log_final_summary(
    variety: str,
    champion: ModelResult,
    registered_name: str | None,
    logger,
) -> None:
    """Mensaje final UNICO con champion + registry + comando UI.

    Sale solo aqui (no por cada modelo en single_run) porque hasta ahora
    el champion ya esta seleccionado, registrado y los losers eliminados.
    """
    registry_line = (
        f"version registrada: {registered_name}"
        if registered_name
        else "Model Registry no aplico (backend no SQL o gate falla)"
    )
    logger.info("=" * 78)
    logger.info(
        f"[{variety}] CHAMPION FINAL: {champion.model_type} "
        f"(run={champion.mlflow_run_id[:12]}) | "
        f"MAE_test={champion.metrics.get('nested_cv_mae_mean', float('nan')):.4f} | "
        f"|gap|={champion.abs_gap:.4f} | "
        f"MAPE_oof={champion.oof_mape:.2f}%"
    )
    logger.info(f"[{variety}] {registry_line}")
    logger.info(
        f"[{variety}] MLflow UI: http://localhost:5000 "
        "(local: `task up`; AWS: ALB del modulo mlflow)"
    )
    logger.info("=" * 78)


def _build_summary(
    variety: str,
    experiment_name: str,
    model_types: list[str],
    failures: list[str],
    registered_name: str | None,
    winner_report_path: str | None,
    winner_excel_path: str | None,
    elapsed: float,
    champion_decision: dict | None,
    results: list[ModelResult],
) -> dict:
    """Construye el diccionario `variety_summary_<NAME>.json`."""
    return {
        "variety": variety,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment_name,
        "model_types_attempted": model_types,
        "model_types_failed": failures,
        "registered_model": registered_name,
        "winner_report": winner_report_path,
        "winner_excel": winner_excel_path,
        "elapsed_seconds": round(elapsed, 2),
        "champion": champion_decision,
        "per_model": [
            {
                "model_type": r.model_type,
                "metrics": r.metrics,
                "full_metrics": r.full_metrics,
                "composite_score": r.composite_score,
                "abs_gap": r.abs_gap,
                "full_mape": r.full_mape if r.full_mape != float("inf") else None,
                "mlflow_run_id": r.mlflow_run_id,
                "pipeline_path": r.pipeline_path,
                "elapsed_seconds": r.elapsed_seconds,
            }
            for r in results
        ],
    }


def _load_variety_inputs(variety: str):
    """Recarga (X, y, business_cols) desde disco para la variedad indicada.

    Se usa al construir el dashboard ganador, el Excel del campeon, y el
    permutation_importance del feature importance, despues de que
    `single_run` libero las matrices grandes en su `del`. Es barato:
    una hoja del Excel de training (~10k filas).

    `rare_min_count` por variedad (P0.2): DEBE coincidir con el del
    training (single_run) o las categorias colapsadas a 'OTROS' no calzan
    con las que vio el pipeline.
    """
    cfg = for_variety(variety)
    X, y = load_data(sheet=variety, rare_min_count=cfg.rare_min_count)
    business_cols = load_business_columns(sheet=variety)
    return X, y, business_cols


def _export_winner_excel(
    champion: ModelResult,
    variety: str,
    logger,
    *,
    X_raw,
    business_cols,
    run_label: str | None = None,
) -> str | None:
    """Genera el Excel multi-hoja del CAMPEON aplicado a la data real.

    Recibe (X_raw, business_cols) ya cargados por el caller (train_variety)
    para evitar releer el Excel multiples veces. El archivo se escribe en
    `reports/` con nombre `Winner_{variety}_{run_label}.xlsx` (acumula uno
    por run); si `run_label` es None cae al patron viejo
    `Winner_{variety}.xlsx` (sobrescribe).

    Devuelve la ruta como string, o None si los inputs son None / faltan
    columnas KG/JR / H-EF o si la prediccion fallo.
    """
    import joblib

    if X_raw is None or business_cols is None:
        return None

    bv = champion.business_validation
    oof = {
        "y_true": champion.oof_y_true,
        "y_pred": champion.oof_y_pred,
    }
    if oof["y_true"] is None or oof["y_pred"] is None or bv is None:
        logger.warning(f"[{variety}] sin datos OOF/business para Excel del campeon")
        return None

    try:
        final_pipeline = joblib.load(champion.pipeline_path)
    except OSError:
        logger.exception(f"[{variety}] no se pudo cargar pipeline del campeon")
        return None

    excel_filename = f"Winner_{variety}_{run_label}.xlsx" if run_label else f"Winner_{variety}.xlsx"
    excel_path = export_business_excel(
        variety=variety,
        model_type=champion.model_type,
        X_raw=X_raw,
        business_cols=business_cols,
        oof=oof,
        final_pipeline=final_pipeline,
        business_validation=bv,
        nested_metrics=champion.metrics,
        output_dir=REPORTS_DIR,
        filename=excel_filename,
    )
    if excel_path is None:
        logger.warning(
            f"[{variety}] Excel del campeon NO generado (faltan columnas KG/JR/H-EF en el dataset)"
        )
        return None
    logger.info(f"[{variety}] Excel del campeon: {excel_path}")
    return str(excel_path)


def _tag_champion(
    champion: ModelResult,
    results: list[ModelResult],
    variety: str,
    champion_decision: dict,
    logger,
    winner_report_path: str | None = None,
    winner_excel_path: str | None = None,
) -> None:
    """Tags + artifacts comparativos en MLflow.

    - Tags: `is_champion=true` en el run ganador, `false` en los perdedores.
      Los tags resumen del campeon (composite_score, gap, full_mape) viven
      solo en el run ganador.
    - Artifacts comparativos (decision JSON, dashboard HTML, Excel ejecutivo,
      summary multi-modelo): se loguean en TODOS los runs de la variedad
      para que cada run histórico tenga su contexto comparativo accesible
      en MLflow UI. Sin esto, abrir lgb_v20 en MLflow no muestra contra qué
      compitió ni por qué perdió.
    """
    try:  # defensive: MLflow + IO (set_tag/log_artifact + dump_json_artifact)
        client = mlflow.tracking.MlflowClient()

        client.set_tag(champion.mlflow_run_id, "is_champion", "true")
        client.set_tag(
            champion.mlflow_run_id,
            "champion_composite_score",
            f"{champion.composite_score:.6f}",
        )
        client.set_tag(
            champion.mlflow_run_id,
            "champion_abs_gap",
            f"{champion.abs_gap:.6f}",
        )
        client.set_tag(
            champion.mlflow_run_id,
            "champion_full_mape",
            f"{champion.full_mape:.4f}",
        )
        client.set_tag(
            champion.mlflow_run_id,
            "champion_oof_mape",
            f"{champion.oof_mape:.4f}",
        )
        if champion_decision.get("walkover"):
            # Gano porque otros backends FALLARON, no una competencia
            # completa. Visible en MLflow UI para no confundir el resultado.
            client.set_tag(champion.mlflow_run_id, "champion_walkover", "true")

        _tag_drift_summary(client, champion, variety, logger)

        for r in results:
            # r.mlflow_run_id puede estar vacio si el run fue eliminado por
            # ser loser (cleanup post-quality-gate). En ese caso skip:
            # ya no existe en MLflow para taggearlo.
            if not r.mlflow_run_id:
                continue
            if r.mlflow_run_id != champion.mlflow_run_id:
                client.set_tag(r.mlflow_run_id, "is_champion", "false")

        decision_path = dump_json_artifact(
            ARTIFACTS_DIR / f"champion_{variety}.json",
            champion_decision,
        )
        summary_path = ARTIFACTS_DIR / f"variety_summary_{variety}.json"

        # Lista (path, mlflow_subpath) de artifacts comparativos. Se filtran
        # los None (artifacts opcionales que no se generaron) y los
        # inexistentes en disco. Loop unico evita ifs paralelos por artefacto.
        candidate_artifacts: list[tuple[str | None, str]] = [
            (str(decision_path), "champion"),
            (winner_report_path, "winner_dashboard"),
            (winner_excel_path, "winner_dashboard"),
            (str(summary_path) if summary_path.exists() else None, "champion"),
        ]

        # EDA mas reciente (HTML + sidecar JSON) adjunto al contexto del
        # run (fix 2026-06-13, pedido del usuario: el EDA no era visible
        # en MLflow). Solo si existe un EDA previo en reports/; el
        # training no genera EDA propio.
        try:
            from src.diagnostics.eda import find_latest_eda_sidecar

            sidecar = find_latest_eda_sidecar(variety)
            if sidecar is not None:
                eda_html = sidecar.with_suffix(".html")
                candidate_artifacts.append((str(sidecar), "eda"))
                if eda_html.exists():
                    candidate_artifacts.append((str(eda_html), "eda"))
        except Exception:  # defensive: el EDA es contexto opcional
            logger.warning(f"[{variety}] no se pudo adjuntar EDA al run")

        artifacts = [(p, sub) for p, sub in candidate_artifacts if p]

        for r in results:
            # Skip losers ya eliminados (mlflow_run_id="" tras cleanup)
            if not r.mlflow_run_id:
                continue
            for path, sub in artifacts:
                client.log_artifact(r.mlflow_run_id, path, artifact_path=sub)
    except Exception:  # defensive: MLflow + IO (set_tag/log_artifact + dump_json_artifact)
        logger.exception(f"[{variety}] no se pudo taggear campeon en MLflow")


def _tag_drift_summary(client, champion: ModelResult, variety: str, logger) -> None:
    """Capa de tags de drift/EDA sobre el run campeon.

    Si existe un EDA sidecar reciente, captura el peor PSI y count de
    severidad para que MLflow UI muestre cuando el data drift se vuelve
    critico run-tras-run sin abrir HTMLs manualmente.
    """
    try:  # defensive: import dinamico + IO (sidecar) + MLflow set_tag
        from src.diagnostics.eda import (
            extract_drift_summary,
            find_latest_eda_sidecar,
        )

        sidecar = find_latest_eda_sidecar(variety)
        if sidecar is not None:
            for k, v in extract_drift_summary(sidecar).items():
                if v:  # skip empty strings
                    client.set_tag(champion.mlflow_run_id, k, v)
            logger.info(f"[{variety}] Drift tags MLflow desde {sidecar.name}")
    except Exception:  # defensive: import dinamico + IO + MLflow
        logger.exception(f"[{variety}] no se pudo taggear drift (no critico)")


def _register_champion(
    champion: ModelResult,
    variety: str,
    args: argparse.Namespace,
    logger,
    winner_report_path: str | None = None,
) -> str | None:
    """Registra el modelo campeon en MLflow Model Registry."""
    logger.info(
        f"[{variety}] Registrando CAMPEON en Model Registry: "
        f"{champion.model_type} (run={champion.mlflow_run_id[:8]}...)"
    )
    if winner_report_path:
        report_uri = (
            f"runs:/{champion.mlflow_run_id}/winner_dashboard/{Path(winner_report_path).name}"
        )
    else:
        report_uri = None
    # register_model devuelve None cuando el backend es file:// (caso default
    # del proyecto local). Si MLFLOW_TRACKING_URI apunta a un backend SQL y
    # algo falla, MlflowException PROPAGA; la capturamos aqui para no
    # abortar la variedad por un fallo del Registry (el modelo ya esta
    # entrenado y logueado en el run).
    try:
        registered_name = register_model(
            run_id=champion.mlflow_run_id,
            artifact_name="model_pipeline",
            variety=variety,
            stage=args.registry_stage if args.registry_stage != "None" else None,
            metrics=champion.metrics,
            report_artifact_uri=report_uri,
            extra_tags={
                "tuning": args.tuning,
                "model_type": champion.model_type,
                "is_champion": "true",
                "composite_score": f"{champion.composite_score:.6f}",
            },
            # MLflow 3.x: pasamos el URI del Logged Model (`models:/m-<id>`)
            # capturado en single_run para que register_model NO caiga al URI
            # legacy `runs:/{run_id}/model_pipeline`, que el server resuelve
            # con un warning "no artifacts at artifact path". Si el log
            # absorbio un run inactivo `model_uri` es None y register_model
            # cae al fallback legacy por compatibilidad.
            model_uri=champion.model_uri,
        )
    except MlflowException:
        logger.exception(
            f"[{variety}] register_model FALLO (auth/red/schema). "
            "El run sigue logueado pero NO se registro version."
        )
        return None
    if registered_name:
        logger.info(f"[{variety}] Registrado: {registered_name}")
    else:
        logger.warning(
            f"[{variety}] Registry no disponible (backend file://); "
            "para versionado configura MLFLOW_TRACKING_URI con un SQL backend."
        )
    return registered_name
