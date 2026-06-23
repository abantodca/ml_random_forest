"""Calculo de metricas de regresion."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE en porcentaje, descartando observaciones con y_true == 0.

    Si todas las observaciones tienen y_true == 0 (divisor cero global),
    devuelve NaN en lugar de propagar una division por cero. Usado por
    `calculate_regression_metrics`, por el bootstrap de IC y por los
    calculos de MAPE por subgrupo en el dashboard.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    nz = yt != 0
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100.0)


def calculate_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Devuelve {mae, rmse, r2, mape}.

    MAPE se calcula descartando observaciones con y_true == 0 para evitar
    divisiones por cero. Si no quedan observaciones validas, MAPE = NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    nonzero = y_true != 0
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0)
    else:
        mape = float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
