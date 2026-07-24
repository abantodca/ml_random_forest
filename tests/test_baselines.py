from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.step_05_evaluate.baselines import (
    HierarchicalMedianRegressor,
    mae_skill_score,
)


def test_hierarchical_baseline_uses_fallbacks_without_test_target() -> None:
    X_train = pd.DataFrame(
        {
            "FUNDO": ["A", "A", "A", "B", "B"],
            "FORMATO": ["G", "G", "C", "G", "G"],
        }
    )
    y_train = pd.Series([2.0, 4.0, 10.0, 8.0, 12.0])
    model = HierarchicalMedianRegressor(min_group_size=2).fit(X_train, y_train)

    X_test = pd.DataFrame(
        {
            "FUNDO": ["A", "B", "NUEVO"],
            "FORMATO": ["G", "C", "NUEVO"],
        }
    )
    pred = model.predict(X_test)

    assert np.allclose(pred, [3.0, 10.0, 8.0])


def test_mae_skill_score_interpretation() -> None:
    assert mae_skill_score(8.0, 10.0) == pytest.approx(0.2)
    assert mae_skill_score(10.0, 10.0) == 0.0
    assert mae_skill_score(12.0, 10.0) == pytest.approx(-0.2)
