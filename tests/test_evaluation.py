from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics.multivariate import compute_mutual_information
from src.step_05_evaluate.champion import ModelResult, select_champion
from src.step_05_evaluate.conformal_bands import build_conformal_metadata
from src.step_05_evaluate.metrics import calculate_regression_metrics


def _result(model: str, mape: float, elapsed: float) -> ModelResult:
    return ModelResult(
        model_type=model,
        metrics={
            "nested_cv_gap_mean": 0.1,
            "nested_cv_mae_mean": 1.0,
        },
        best_params={},
        mlflow_run_id=model,
        pipeline_path=f"{model}.joblib",
        elapsed_seconds=elapsed,
        business_metrics_oof={"mape": mape},
    )


def test_champion_tolerance_has_no_bucket_boundary_bug() -> None:
    slower_but_tiny_bit_better = _result("xgb", 14.49, 20.0)
    faster = _result("lgb", 14.51, 10.0)
    assert select_champion([slower_but_tiny_bit_better, faster]) is faster


def test_metrics_include_scale_relative_and_bias_views() -> None:
    metrics = calculate_regression_metrics([10, 20], [8, 22], min_denom=1)
    assert metrics["mae"] == 2.0
    assert metrics["median_ae"] == 2.0
    assert metrics["bias"] == 0.0
    assert np.isclose(metrics["wape"], 100 * 4 / 30)
    assert "smape" in metrics


def test_mi_keeps_rows_with_missing_and_includes_categories() -> None:
    n = 200
    category = np.where(np.arange(n) % 2 == 0, "alto", "bajo")
    y = pd.Series(np.where(category == "alto", 10.0, 1.0))
    numeric = np.linspace(0, 1, n)
    numeric[::3] = np.nan
    X = pd.DataFrame({"categoria": category, "numerica": numeric})

    result = compute_mutual_information(X, y)
    by_name = {item.feature: item.mi for item in result}
    assert set(by_name) == {"categoria", "numerica"}
    assert by_name["categoria"] > by_name["numerica"]


def test_conformal_uses_recent_residuals_when_they_are_wider() -> None:
    n = 400
    y_true = np.zeros(n)
    y_pred = np.r_[np.full(380, 0.5), np.full(20, 3.0)]
    dates = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    fundo = pd.Series(["A"] * n)
    formato = pd.Series(["G"] * n)

    metadata = build_conformal_metadata(
        y_true,
        y_pred,
        fundo,
        formato,
        dates=dates,
        alpha=0.10,
    )

    assert metadata is not None
    assert metadata["q_global_recent"] > metadata["q_global_all"]
    assert metadata["q_global"] == metadata["q_global_recent"]
    assert metadata["backtest_coverage"] < metadata["target_coverage"]
