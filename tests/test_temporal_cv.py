from __future__ import annotations

import numpy as np
import pandas as pd

from src.step_04_train.temporal_cv import PurgedDateSplit, TemporalYearSplit


def test_purged_date_split_never_shares_or_reverses_dates() -> None:
    dates = pd.date_range("2024-01-01", periods=180, freq="D").repeat(2)
    X = pd.DataFrame({"FECHA": dates})
    splitter = PurgedDateSplit(
        n_splits=3,
        gap_periods=2,
        min_train_periods=30,
        min_test_periods=10,
    )

    splits = list(splitter.split(X))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        train_dates = pd.to_datetime(X.iloc[train_idx]["FECHA"])
        test_dates = pd.to_datetime(X.iloc[test_idx]["FECHA"])
        assert train_dates.max() < test_dates.min()
        assert not set(train_dates).intersection(set(test_dates))
        assert (test_dates.min() - train_dates.max()).days >= 3


def test_year_split_is_expanding_window() -> None:
    years = np.repeat([2022, 2023, 2024, 2025], 5)
    X = pd.DataFrame({"ANIO": years})
    splitter = TemporalYearSplit(n_splits=2, min_train_years=2)

    splits = list(splitter.split(X))
    assert len(splits) == 2
    assert set(X.iloc[splits[0][0]]["ANIO"]) == {2022, 2023}
    assert set(X.iloc[splits[0][1]]["ANIO"]) == {2024}
    assert set(X.iloc[splits[1][0]]["ANIO"]) == {2022, 2023, 2024}
    assert set(X.iloc[splits[1][1]]["ANIO"]) == {2025}

