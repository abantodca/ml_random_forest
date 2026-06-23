"""Metadata de bandas conformal por grupo, calibrada con residuos OOF.

Reemplaza la heuristica ±1.96·std del ensemble que la API servia como "IC
95%" (revision experta 2026-06-11: los K=5 pipelines comparten ~80% del
train y sus predicciones estan correlacionadas — el std entre ellos mide
varianza del estimador, no el ruido residual; cobertura real << nominal).

Diseno:
    - Split conformal (cuantil de |residuo OOF| con correccion finita,
      mismo fundamento que statistical_tests.conformal_intervals) pero
      calibrado POR FUNDO: la heterogeneidad medida es real (q90 LN=0.66
      vs C6=1.57 en POP 2026-06-11) y una banda global miente en ambos
      extremos.
    - Grupos chicos (n < MIN_GROUP) caen al cuantil global (mismo patron
      cascade que OutlierCapper / imputer jerarquico).
    - `known_ff`: combinaciones FUNDO__FORMATO con historia suficiente en
      train. Una fila cuyo grupo NO esta aqui es cold-start: error medido
      ~2x (MAPE 31-36% vs 14%) -> la API debe marcar confidence="baja" y
      ensanchar la banda con `cold_factor`.

La metadata se ADJUNTA al pipeline final como atributo `conformal_`
(dict plano, pickle-safe) en single_run; la API la lee si existe y cae a
la heuristica de std solo con modelos legacy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Minimo de residuos OOF para calibrar el cuantil de un fundo; bajo esto
# cae al global (cuantil sobre n chico = ruidoso).
MIN_GROUP = 100
# Minimo de observaciones de un FUNDO__FORMATO en train para NO considerarlo
# cold-start en inferencia (alineado con MIN_PERIODS=3 de los lags, con
# margen: bajo ~10 obs los lags siguen mayormente en sentinel).
MIN_FF_KNOWN = 10
# Factor de ensanche para filas cold-start: error medido ~2x el normal.
COLD_FACTOR = 2.0


def build_conformal_metadata(
    y_true: np.ndarray,
    y_pred_oof: np.ndarray,
    fundo: pd.Series,
    formato: pd.Series,
    alpha: float = 0.10,
) -> dict[str, object] | None:
    """Construye la metadata de bandas a partir del OOF del nested CV.

    Devuelve dict pickle-safe:
        {
          "alpha": 0.10,
          "q_global": float,            # cuantil |residuo| con corr. finita
          "q_by_fundo": {fundo: q},     # solo fundos con n >= MIN_GROUP
          "known_ff": [..],             # claves FUNDO__FORMATO con historia
          "cold_factor": 2.0,
          "n_calibration": int,
        }
    o None si no hay residuos suficientes (<20, igual que conformal_intervals).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred_oof, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 20:
        return None

    abs_res = np.abs(yp[mask] - yt[mask])

    def _q(arr: np.ndarray) -> float:
        n = len(arr)
        level = min(1.0, (1 - alpha) * (n + 1) / n)
        return float(np.quantile(arr, level))

    fundo_m = fundo.reset_index(drop=True)[mask].astype(str)
    q_by_fundo: dict[str, float] = {}
    for f, idx in fundo_m.groupby(fundo_m).groups.items():
        res_f = abs_res[fundo_m.index.get_indexer(idx)]
        if len(res_f) >= MIN_GROUP:
            q_by_fundo[str(f)] = _q(res_f)

    ff_keys = (
        fundo.reset_index(drop=True).astype(str) + "__" + formato.reset_index(drop=True).astype(str)
    )
    counts = ff_keys.value_counts()
    known_ff = sorted(counts[counts >= MIN_FF_KNOWN].index.tolist())

    return {
        "alpha": alpha,
        "q_global": _q(abs_res),
        "q_by_fundo": q_by_fundo,
        "known_ff": known_ff,
        "cold_factor": COLD_FACTOR,
        "n_calibration": int(mask.sum()),
    }
