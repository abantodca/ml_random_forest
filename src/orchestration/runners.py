"""Estrategias de ejecucion: secuencial o paralela (multi-proceso).

`_train_variety_worker` es el entry-point del subproceso (debe ser top-level
para que `ProcessPoolExecutor` lo pueda picklear).

Trade-off: paralelo libera memoria al terminar el subproceso (cleanup
trivial), pero usa ~N veces la memoria pico. En t3.large (8 GB) con 4
variedades en paralelo se ajusta `inner_cv_n_jobs` para evitar
oversubscription de CPU.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import init_dirs
from src.orchestration.cleanup import cleanup_state
from src.orchestration.variety_runner import train_variety
from src.step_06_track.mlflow_registry import init_mlflow
from src.utils.logger import setup_logging


def _train_variety_worker(
    variety: str,
    model_types: list[str],
    args_dict: dict,
    settings: dict,
) -> dict:
    """Entry-point del subproceso. Top-level (picklable)."""
    args = argparse.Namespace(**args_dict)
    init_dirs()  # subproceso fresco -> garantizar dirs antes de loguear
    var_logger = setup_logging(
        name=f"variety.{variety}",
        log_file=f"variety_{variety}.log",
    )
    init_mlflow()  # cada worker setea su tracking URI propio
    return train_variety(variety, model_types, args, settings, var_logger)


def run_sequential(
    varieties: list[str],
    models: list[str],
    args: argparse.Namespace,
    settings: dict,
    logger,
) -> tuple[dict[str, dict], list[str]]:
    aggregate: dict[str, dict] = {}
    failed: list[str] = []
    for variety in varieties:
        logger.info(f"==> START variedad {variety}")
        try:
            aggregate[variety] = train_variety(variety, models, args, settings, logger=logger)
            logger.info(f"==> DONE  variedad {variety}")
        except Exception as exc:
            logger.exception(f"Variedad {variety} FALLO globalmente")
            aggregate[variety] = {"error": str(exc)}
            failed.append(variety)
        finally:
            cleanup_state(logger, f"post variedad {variety}")
    return aggregate, failed


def run_parallel(
    varieties: list[str],
    models: list[str],
    args: argparse.Namespace,
    settings: dict,
    logger,
    n_workers: int,
) -> tuple[dict[str, dict], list[str]]:
    args_dict = vars(args)
    aggregate: dict[str, dict] = {}
    failed: list[str] = []

    logger.info(
        f"PARALELO | {n_workers} workers | "
        f"inner_cv_n_jobs={settings.get('inner_cv_n_jobs')}"
    )
    logger.info(
        "Cada variedad corre en un PROCESO independiente. Logs van a "
        "logs/variety_<NAME>.log (no se interleavean)."
    )

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_train_variety_worker, v, models, args_dict, settings): v
            for v in varieties
        }
        logger.info(f"==> Lanzadas {len(futures)} variedades en paralelo")
        for fut in as_completed(futures):
            variety = futures[fut]
            try:
                aggregate[variety] = fut.result()
                logger.info(f"==> DONE  variedad {variety}")
            except Exception as exc:
                logger.exception(f"Variedad {variety} FALLO en worker")
                aggregate[variety] = {"error": str(exc)}
                failed.append(variety)
    return aggregate, failed
