"""Quality gate de registro del campeon.

Decide si el modelo campeon de una variedad se registra en el MLflow Model
Registry, segun MAPE_oof (calidad operativa, bloqueante), |gap| (diagnostico de
overfitting, NO bloqueante) y los guards de flags experimentales
(REGISTER_ENABLED, EXANTE_MODE, tuning=smoke).

Extraido de `variety_runner.py` (2026-06-26) como unidad cohesiva y testeable:
la suite P0 `tests/test_register_guard.py` ejercita `apply_quality_gate`
aisladamente.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from src.config import (
    CHAMPION_MAX_GAP_REL,
    CHAMPION_MAX_MAPE,
    CHAMPION_WARN_TEMPORAL_MAPE,
    CHAMPION_WARN_TEMPORAL_R2,
)

if TYPE_CHECKING:
    from src.step_05_evaluate.champion import ModelResult


def _warn_temporal_generalization(champion: ModelResult, variety: str, logger) -> None:
    """Emite un WARNING si el chequeo honesto temporal salio pobre.

    NO bloquea el registro: el MAPE_oof stratified (gate operativo) ya
    confirmo calidad de interpolacion. Esto solo da visibilidad a que el
    forecast de un anio NO visto generaliza peor (riesgo de drift). Si las
    metricas temporales no estan en champion.metrics (DUAL_CV_REPORT=0 o
    outer ya temporal), no hace nada.
    """
    t_mape = champion.metrics.get("temporal_mape_oof")
    t_r2 = champion.metrics.get("temporal_r2_oof")
    if t_mape is None and t_r2 is None:
        return
    # Con <2 folds temporales (3 anios de historia) las metricas son UNA sola
    # ventana: demasiado ruidosas para acusar drift. El chequeo ya lo loguea
    # como indicativo; aqui NO warneamos (2026-07-01). Runs viejos sin
    # temporal_n_folds conservan el comportamiento historico (warnean).
    n_folds = champion.metrics.get("temporal_n_folds")
    if n_folds is not None and n_folds < 2:
        return
    mape_bad = t_mape is not None and t_mape > CHAMPION_WARN_TEMPORAL_MAPE
    r2_bad = t_r2 is not None and t_r2 < CHAMPION_WARN_TEMPORAL_R2
    if not (mape_bad or r2_bad):
        return
    logger.warning(
        f"[{variety}] AVISO de generalizacion temporal (NO bloquea registro) | "
        f"temporal_MAPE_oof={t_mape:.2f}% (aviso>{CHAMPION_WARN_TEMPORAL_MAPE}%) | "
        f"temporal_R2_oof={t_r2:.4f} (aviso<{CHAMPION_WARN_TEMPORAL_R2}). "
        f"El MAPE_oof del gate mide INTERPOLACION (anios mezclados); este mide "
        f"forecast de un anio NO visto. Brecha grande = riesgo de drift en "
        f"produccion: revisar reentrenos frecuentes o CV_OUTER_STRATEGY=temporal_year "
        f"para esta variedad."
    )


def apply_quality_gate(
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

    # Gap RELATIVO (unificado 2026-07-01): select_champion decide con gap_rel
    # (adimensional, comparable entre variedades); este warning usaba el gap
    # absoluto viejo (kilos*100 llamado "pp", ver caveat en config) — dos
    # definiciones conviviendo confundian el log. Ahora ambos hablan gap_rel.
    mape_ok = champion.oof_mape <= CHAMPION_MAX_MAPE
    gap_ok = champion.gap_rel <= CHAMPION_MAX_GAP_REL

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

    # Aviso temporal NO bloqueante (2026-06-25): el gate de arriba usa el MAPE
    # stratified (interpolacion, optimista). Si el chequeo honesto temporal
    # (forecast de anio no visto) salio pobre, lo avisamos para visibilidad de
    # drift — sin tocar la decision de registro (no rompe campeones existentes).
    _warn_temporal_generalization(champion, variety, logger)

    if not gap_ok:
        logger.warning(
            f"[{variety}] CAMPEON registra con WARNING de overfitting | "
            f"gap_rel={champion.gap_rel:.2f} supera threshold "
            f"{CHAMPION_MAX_GAP_REL} pero MAPE_oof={champion.oof_mape:.2f}% "
            f"(<={CHAMPION_MAX_MAPE}%) confirma calidad operativa OK. "
            f"El gap es diagnostico de memoria del train, NO afecta predicciones "
            f"OOF/produccion. Modelo aprobado para Registry."
        )
        return args.register_model
    logger.info(
        f"[{variety}] CAMPEON pasa quality gate | "
        f"MAPE_oof={champion.oof_mape:.2f}% (max={CHAMPION_MAX_MAPE}%) | "
        f"gap_rel={champion.gap_rel:.2f} (max={CHAMPION_MAX_GAP_REL}). "
        f"Registra en MLflow Model Registry."
    )
    return args.register_model
