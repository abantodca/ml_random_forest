from __future__ import annotations

import numpy as np
import pandas as pd

from src.step_03_features.lag_features import (
    _rolling_lag_exante,
    _seasonal_lag_for_group,
)


def test_exante_lag_excludes_every_row_from_current_day() -> None:
    frame = pd.DataFrame(
        {
            "FUNDO": ["A"] * 8,
            "FORMATO": ["G"] * 8,
            "FECHA": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-06",
                ]
            ),
            "value": [1.0, 2.0, 3.0, 4.0, 1000.0, 2000.0, 6.0, 7.0],
        }
    )

    lag = _rolling_lag_exante(
        frame,
        "value",
        ["FUNDO", "FORMATO"],
        window=3,
    )

    # Ambas filas del 5 de enero usan solo días 2, 3 y 4: el valor extremo
    # de una fila hermana del mismo día nunca entra en la otra.
    assert lag.iloc[4] == 3.0
    assert lag.iloc[5] == 3.0


def test_seasonal_lag_cannot_use_current_or_future_values() -> None:
    dates = pd.date_range("2023-01-01", periods=800, freq="D").to_numpy(dtype="datetime64[D]")
    values = np.arange(800, dtype=float)
    original = _seasonal_lag_for_group(dates, values)

    changed_future = values.copy()
    changed_future[-50:] = 1_000_000.0
    recomputed = _seasonal_lag_for_group(dates, changed_future)

    # Alterar el futuro no puede cambiar features ya calculadas para el pasado.
    assert np.allclose(original[:-50], recomputed[:-50], equal_nan=True)
