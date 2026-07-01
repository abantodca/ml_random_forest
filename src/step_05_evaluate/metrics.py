"""Calculo de metricas de regresion."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MAPE_MIN_DENOM


def _mape_valid_mask(y_true: np.ndarray, min_denom: float) -> np.ndarray:
    """Filas cuyo denominador es utilizable para el MAPE.

    Excluye |y_true| < min_denom (no solo == 0): un KG/JR ~ 0 hace que el
    termino |y-yhat|/|y| explote y una sola fila artefacto domina la media.
    Ver `MAPE_MIN_DENOM` en config.py para el porque fisico.
    """
    yt = np.asarray(y_true, dtype=float)
    return np.abs(yt) >= min_denom


def mape_safe(y_true, y_pred, min_denom: float = MAPE_MIN_DENOM) -> float:
    """MAPE en porcentaje, descartando denominadores < `min_denom`.

    Antes descartaba solo y_true == 0 EXACTO, lo que dejaba pasar filas
    casi-cero (KG/JR ~ 0.004) que inflaban el MAPE a cientos de %. Ahora usa
    un piso fisico (`MAPE_MIN_DENOM`). Si no queda ninguna observacion valida
    devuelve NaN en lugar de propagar la division. Usado por
    `calculate_regression_metrics`, el bootstrap de IC y los MAPE por subgrupo
    del dashboard.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    valid = _mape_valid_mask(yt, min_denom)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs((yt[valid] - yp[valid]) / yt[valid])) * 100.0)


def calculate_regression_metrics(
    y_true, y_pred, min_denom: float = MAPE_MIN_DENOM
) -> dict[str, float]:
    """Devuelve {mae, rmse, r2, mape, mape_n_excluded}.

    MAE/RMSE/R2 se calculan sobre TODAS las filas (son robustos a escala).
    MAPE descarta observaciones con |y_true| < `min_denom` para evitar que un
    denominador ~0 domine la media; `mape_n_excluded` reporta cuantas se
    descartaron (transparencia: aparece en la auditoria de negocio). Si no
    queda ninguna observacion valida, MAPE = NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    valid = _mape_valid_mask(y_true, min_denom)
    n_excluded = int((~valid).sum())
    if valid.any():
        mape = float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0)
    else:
        mape = float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "mape_n_excluded": n_excluded}
