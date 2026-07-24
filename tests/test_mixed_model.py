from __future__ import annotations

import joblib
import numpy as np

from src.step_04_train.model_mixed import ResidualHybridRegressor


def test_residual_hybrid_is_reproducible_and_serializable(tmp_path) -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(250, 4))
    y = 2.0 * X[:, 0] + np.sin(3.0 * X[:, 1]) + rng.normal(0, 0.05, 250)
    kwargs = {
        "max_iter": 80,
        "min_samples_leaf": 10,
        "random_state": 42,
    }
    sample_weight = np.linspace(0.5, 1.5, len(y))
    first = ResidualHybridRegressor(**kwargs).fit(X, y, sample_weight=sample_weight)
    second = ResidualHybridRegressor(**kwargs).fit(X, y, sample_weight=sample_weight)
    expected = first.predict(X)

    assert np.allclose(expected, second.predict(X))

    path = tmp_path / "mixed.joblib"
    joblib.dump(first, path)
    restored = joblib.load(path)
    assert np.allclose(expected, restored.predict(X))
