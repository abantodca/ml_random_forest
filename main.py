"""Entrypoint del pipeline ML.

main.py es THIN: parse args -> bootstrap -> delega en `src/orchestration/`.

Uso normal (via Taskfile):
    task train VARIETIES=POP
    task train VARIETIES=POP,VENTURA
    task train VARIETIES=all

Uso directo (CLI avanzado):
    python main.py --varieties POP
    python main.py --varieties POP,JUPITER
    python main.py --varieties all --tuning prod_xl

El pipeline SIEMPRE entrena todos los backends del registry (XGB + LGB hoy)
y `champion.select_champion` elige el ganador por variedad (lex-order:
gap -> MAPE -> tiempo). No hay flag de usuario para forzar un modelo: ese
es el contrato del proyecto. Default --tuning dev. Cada variedad es
independiente: si una falla, las demas continuan.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import (
    ACCUMULATED_FILE,
    ARTIFACTS_DIR,
    MIN_ROWS_PER_VARIETY,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
    S3_ARTIFACTS_BUCKET,
    S3_ARTIFACTS_PREFIX,
    S3_REPORTS_PREFIX,
    TRAINING_FILE,
    init_dirs,
)
from src.orchestration.cli import (
    parse_args,
    resolve_settings,
    resolve_varieties,
)
from src.orchestration.runners import run_parallel, run_sequential
from src.step_04_train.registry import valid_backends
from src.step_06_track.mlflow_registry import init_mlflow
from src.utils.logger import setup_logging
from src.utils.sklearn_helpers import dump_json_artifact


def _hydrate_data_from_s3(logger) -> bool:
    """Descarga el Excel acumulado desde S3 y genera DB-HISTORICA.xlsx.

    Solo se activa si las env vars `S3_DATA_BUCKET` y `S3_DATA_KEY` estan
    presentes (e.g. inyectadas por el Lambda dispatcher de AWS Batch). El
    flujo es:

        s3://bucket/key  (BD_HISTORICO_ACUMULADO.xlsx)
            └─> data/BD_HISTORICO_ACUMULADO.xlsx
                  └─> scripts.prepare_data.split_workbook(...)
                        └─> data/training/DB-HISTORICA.xlsx

    En local (sin esas env vars) no hace nada y el pipeline asume que el
    usuario ya ejecuto `task data:split`. Devuelve True si hidrato (o no
    hizo falta), False si fallo.
    """
    bucket = os.environ.get("S3_DATA_BUCKET")
    key = os.environ.get("S3_DATA_KEY")
    if not (bucket and key):
        return True  # local: prepare_data ya corrio offline

    logger.info(f"Hydratando data desde s3://{bucket}/{key}...")
    try:
        import boto3

        from scripts.prepare_data import split_workbook

        ACCUMULATED_FILE.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(bucket, key, str(ACCUMULATED_FILE))
        logger.info(
            f"Acumulado descargado: {ACCUMULATED_FILE} ({ACCUMULATED_FILE.stat().st_size} bytes)"
        )

        summary = split_workbook(
            input_path=ACCUMULATED_FILE,
            output_path=TRAINING_FILE,
            min_rows=MIN_ROWS_PER_VARIETY,
        )
        logger.info(
            f"Split OK | hojas={summary['varieties_written']} "
            f"({summary['rows_written']} filas) -> {TRAINING_FILE} | "
            f"descartadas={summary['varieties_skipped']} (<{summary['min_rows']} filas)"
        )
        return True
    except Exception as exc:
        logger.exception(f"Hydrate desde S3 fallo: {exc}")
        return False


def _resolve_parallelism(args, settings: dict) -> int:
    """Ajusta el presupuesto de hilos cuando hay paralelismo de variedades
    para evitar oversubscription de CPU.

    Con `--parallel-varieties N`, cada variedad corre en su PROCESO y cada
    `fit` del modelo abre `MODEL_N_JOBS` hilos nativos (LightGBM/XGB). Si
    `MODEL_N_JOBS=-1` (default), cada uno de los N procesos toma TODOS los
    cores -> N*cores hilos peleando por cores cores. Repartimos
    `MODEL_N_JOBS = cores // N` para que N*MODEL_N_JOBS ~= cores. Se setea
    en el entorno del proceso padre ANTES de lanzar el pool, asi los
    subprocesos lo heredan. Un override explicito de MODEL_N_JOBS se respeta.
    """
    parallel = max(1, int(args.parallel_varieties))
    inner_n_jobs = args.inner_cv_n_jobs
    if parallel > 1:
        cores = os.cpu_count() or 4
        per_variety = max(1, cores // parallel)
        if inner_n_jobs == -1:
            inner_n_jobs = per_variety
        if "MODEL_N_JOBS" not in os.environ:
            os.environ["MODEL_N_JOBS"] = str(per_variety)
    settings["inner_cv_n_jobs"] = inner_n_jobs
    return parallel


def _write_aggregate_summary(
    aggregate: dict[str, dict],
    failed_varieties: list[str],
    varieties: list[str],
    models: list[str],
    elapsed_seconds: float,
) -> Path:
    return dump_json_artifact(
        ARTIFACTS_DIR / "run_summary_AGGREGATE.json",
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds_total": round(elapsed_seconds, 2),
            "n_varieties": len(varieties),
            "n_failed_varieties": len(failed_varieties),
            "failed_varieties": failed_varieties,
            "models_trained": models,
            "champions": {
                v: data.get("champion", {}).get("champion_model")
                for v, data in aggregate.items()
                if isinstance(data, dict) and data.get("champion") is not None
            },
            "per_variety": aggregate,
        },
    )


def _run_eda_mode(varieties: list[str], logger) -> int:
    """Modo --eda: corre el EDA exploratorio por variedad (standalone, sin
    entrenar) y sincroniza los HTML/JSON a S3. Repetible on-demand. Asume que
    el hydrate de data desde S3 ya corrio (igual que el training)."""
    from src.diagnostics.eda import run_eda

    init_mlflow()  # run_eda loguea a MLflow por variedad
    logger.info("=" * 78)
    logger.info(f"Modo EDA | variedades={varieties}")
    logger.info("=" * 78)

    failed: list[str] = []
    for v in varieties:
        try:
            out_path = run_eda(v)
            logger.info(f"[EDA/{v}] reporte: {out_path}")
        except Exception as exc:
            logger.error(f"[EDA/{v}] fallo: {exc}")
            failed.append(v)

    # Mismo sync que el training: empuja reports/ (HTML EDA) + artifacts/ a S3.
    # En local (sin S3_ARTIFACTS_BUCKET) es no-op.
    if S3_ARTIFACTS_BUCKET:
        from scripts.s3_sync import sync_to_s3

        sync_to_s3(
            artifacts_dir=ARTIFACTS_DIR,
            reports_dir=REPORTS_DIR,
            bucket=S3_ARTIFACTS_BUCKET,
            artifacts_prefix=S3_ARTIFACTS_PREFIX,
            reports_prefix=S3_REPORTS_PREFIX,
        )

    logger.info(f"FIN EDA | variedades={len(varieties)} | falladas={len(failed)}")
    return 0 if not failed else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = resolve_settings(args)
    init_dirs()  # crea logs/, artifacts/, reports/ (idempotente)
    logger = setup_logging()

    # En AWS Batch: descarga BD_HISTORICO_ACUMULADO.xlsx desde S3 y genera
    # DB-HISTORICA.xlsx via scripts.prepare_data.split_workbook. En local
    # (sin S3_DATA_BUCKET/S3_DATA_KEY) es no-op: asume que `task data:split`
    # ya corrio offline.
    if not _hydrate_data_from_s3(logger):
        logger.error("Hydrate de data desde S3 fallo; abortando.")
        return 2

    if not TRAINING_FILE.exists():
        logger.error(
            f"No existe el archivo de training: {TRAINING_FILE}. "
            f"Corre `task data:split` (o seteá S3_DATA_BUCKET+S3_DATA_KEY)."
        )
        return 2

    # resolve_varieties valida contra el catalogo y aborta con sys.exit(2)
    # ante variedades desconocidas (no lanza ValueError).
    varieties = resolve_varieties(args.varieties)
    # El proyecto NO permite forzar un modelo: siempre entrena todos los
    # backends del registry y `champion.select_champion` decide el ganador
    # por variedad (lex-order: gap -> MAPE -> tiempo).
    models = list(valid_backends())
    if not varieties:
        logger.error("varieties vacio; revisa --varieties")
        return 2

    # Modo EDA (--eda): standalone, NO entrena. No necesita el registry de
    # modelos; corre el analisis exploratorio por variedad y sincroniza a S3.
    if args.eda:
        return _run_eda_mode(varieties, logger)

    if not models:
        logger.error("BACKEND_REGISTRY vacio; revisa src/step_04_train/registry.py")
        return 2

    parallel = _resolve_parallelism(args, settings)

    logger.info("=" * 78)
    logger.info(f"Inicio | tuning={args.tuning} | models={models} | varieties={varieties}")
    logger.info(
        f"Paralelismo: variedades={parallel} | "
        f"inner_cv_n_jobs={settings['inner_cv_n_jobs']} | cores={os.cpu_count()}"
    )
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info("=" * 78)

    init_mlflow()  # solo seteamos URI; experimento se setea por variedad

    t_total = time.perf_counter()

    if parallel > 1 and len(varieties) > 1:
        aggregate, failed_varieties = run_parallel(
            varieties,
            models,
            args,
            settings,
            logger,
            n_workers=min(parallel, len(varieties)),
        )
    else:
        aggregate, failed_varieties = run_sequential(
            varieties,
            models,
            args,
            settings,
            logger,
        )

    total_dt = time.perf_counter() - t_total
    aggregate_path = _write_aggregate_summary(
        aggregate,
        failed_varieties,
        varieties,
        models,
        total_dt
    )

    logger.info("=" * 78)
    logger.info(
        f"FIN | variedades={len(varieties)} | falladas={len(failed_varieties)} "
        f"| tiempo_total={total_dt:.1f}s"
    )
    logger.info("Campeones por variedad:")
    for v, data in aggregate.items():
        if isinstance(data, dict) and data.get("champion"):
            ch = data["champion"]
            logger.info(
                f"  {v:25s} -> {ch['champion_model']:5s} "
                f"composite={ch['champion_composite_score']:.4f}"
            )
    logger.info(f"Resumen agregado: {aggregate_path}")
    logger.info("=" * 78)

    # S3 sync: solo si S3_ARTIFACTS_BUCKET esta configurado (EC2/CI).
    # En local el bucket esta vacio -> se omite silenciosamente.
    if S3_ARTIFACTS_BUCKET:
        from scripts.s3_sync import sync_to_s3

        sync_to_s3(
            artifacts_dir=ARTIFACTS_DIR,
            reports_dir=REPORTS_DIR,
            bucket=S3_ARTIFACTS_BUCKET,
            artifacts_prefix=S3_ARTIFACTS_PREFIX,
            reports_prefix=S3_REPORTS_PREFIX,
        )

    return 0 if not failed_varieties else 1


if __name__ == "__main__":
    sys.exit(main())
