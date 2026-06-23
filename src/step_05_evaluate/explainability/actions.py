"""Acciones recomendadas auto-generadas del analisis de errores.

Reglas (umbrales en config.py):
  - Si un FORMATO/FUNDO tiene MAPE >= REPORT_SUBGROUP_WARN_RATIO * global -> warning.
  - Si abs_gap > ABS_GAP_WARN -> warning de overfitting en lenguaje natural.
  - Si full_mape > FULL_MAPE_CRITICAL_PCT -> critical.
  - Si nada de arriba aplica -> info "todo OK".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    ABS_GAP_WARN,
    FULL_MAPE_CRITICAL_PCT,
    REPORT_SUBGROUP_MIN_N,
    REPORT_SUBGROUP_WARN_RATIO,
)
from src.step_05_evaluate.metrics import mape_safe


@dataclass(frozen=True)
class Action:
    """Recomendacion accionable auto-generada."""

    severity: str  # 'critical' | 'warning' | 'info'
    icon: str
    title: str
    body: str


def recommended_actions(
    *,
    abs_errors: np.ndarray,
    real: np.ndarray,
    X_aligned: pd.DataFrame | None,
    global_mape: float,
    abs_gap: float,
    full_mape: float,
) -> list[Action]:
    """Genera 0-5 acciones a partir de subgroups y metricas globales."""
    actions: list[Action] = []

    # Globales
    if full_mape > FULL_MAPE_CRITICAL_PCT:
        actions.append(
            Action(
                severity="critical",
                icon="🚫",
                title="Error global alto",
                body=(
                    f"El error promedio del modelo ({full_mape:.1f}%) excede "
                    "el umbral aceptable. No usar predicciones para decisiones "
                    "operativas críticas hasta diagnosticar la causa."
                ),
            )
        )
    if abs_gap > ABS_GAP_WARN:
        actions.append(
            Action(
                severity="warning",
                icon="⚠",
                title="El modelo memorizó parte del entrenamiento",
                body=(
                    f"Hay una diferencia notable entre el error en datos vistos "
                    f"y datos nuevos (brecha = {abs_gap:.3f}). Esto suele "
                    "indicar que el modelo aprendió patrones específicos del "
                    "histórico que pueden no repetirse. Considerar reducir "
                    "complejidad o agregar más datos antes de desplegar."
                ),
            )
        )

    # Subgrupos
    if X_aligned is not None and len(X_aligned) == abs_errors.size and global_mape > 0:
        warn_thr = global_mape * REPORT_SUBGROUP_WARN_RATIO
        for col in ("FORMATO", "FUNDO"):
            if col not in X_aligned.columns:
                continue
            groups = X_aligned[col].astype(str).reset_index(drop=True)
            for cat in groups.unique():
                if pd.isna(cat) or cat == "":
                    continue
                mask = (groups == cat).to_numpy()
                if mask.sum() < REPORT_SUBGROUP_MIN_N:  # ignora subgrupos diminutos
                    continue
                cat_real = real[mask]
                cat_err = abs_errors[mask]
                nonzero = cat_real != 0
                if nonzero.sum() == 0:
                    continue
                # `cat_err = |pred - real|`, asi que `mape_safe(real, real - |err|)`
                # reproduce el MAPE original sin necesidad de propagar `pred`.
                cat_mape = mape_safe(cat_real, cat_real - cat_err)
                if cat_mape >= warn_thr:
                    ratio = cat_mape / global_mape if global_mape > 0 else float("inf")
                    actions.append(
                        Action(
                            severity="warning",
                            icon="⚠",
                            title=f"{col} '{cat}': error {ratio:.1f}× mayor que el promedio",
                            body=(
                                f"En este segmento ({int(mask.sum())} cosechas) el error "
                                f"medio es {cat_mape:.1f}% vs {global_mape:.1f}% del global. "
                                "Recomendamos NO automatizar predicciones aquí — usar "
                                "criterio operativo o entrenar modelo dedicado."
                            ),
                        )
                    )

    if not actions:
        actions.append(
            Action(
                severity="info",
                icon="✅",
                title="No se detectaron problemas significativos",
                body=(
                    "Las métricas globales y por subgrupo están dentro de rangos "
                    "esperados. Continuar con el plan de despliegue y monitoreo."
                ),
            )
        )

    return actions
