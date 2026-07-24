from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.step_01_load.data_loader import load_business_columns, load_data


def test_loader_filters_invalid_target_and_date_and_preserves_alignment(tmp_path) -> None:
    rows = 8
    df = pd.DataFrame(
        {
            **{column: np.arange(1, rows + 1, dtype=float) for column in NUMERIC_FEATURES},
            **{column: ["A"] * rows for column in CATEGORICAL_FEATURES},
            "FECHA": pd.to_datetime(
                [
                    "2024-01-08",
                    "2024-01-07",
                    "2024-01-06",
                    "2024-01-05",
                    "2024-01-04",
                    "2024-01-03",
                    "2024-01-02",
                    "2024-01-01",
                ]
            ),
            TARGET: [1.0, "malo", 3.0, -1.0, 5.0, 6.0, 7.0, 8.0],
            "KG/JR": np.arange(101, 109, dtype=float),
            "H-EF": [8.0] * rows,
        }
    )
    df.loc[4, "FECHA"] = pd.NaT
    path = tmp_path / "training.xlsx"
    df.to_excel(path, sheet_name="POP", index=False)

    X, y = load_data(path=path, sheet="POP", collapse_rare_categories=False)
    business = load_business_columns(path=path, sheet="POP")

    assert len(X) == len(y) == len(business) == 5
    assert X["FECHA"].is_monotonic_increasing
    assert np.all(np.isfinite(y))
    assert np.allclose(
        business["KG/JR"] / business["H-EF"],
        [13.5, 13.375, 13.25, 12.875, 12.625],
    )
