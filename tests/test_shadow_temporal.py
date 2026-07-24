from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.orchestration.cli import parse_args
from src.orchestration.quality_gate import apply_quality_gate
from src.step_04_train.cv_strategy import build_cv_splitters
from src.step_04_train.temporal_cv import (
    PurgedDateSplit,
    TemporalYearSplit,
    recent_training_window,
)
from src.step_05_evaluate.champion import ModelResult
from src.variety_config import for_variety, shadow_temporal_varieties


def test_shadow_policy_is_opt_in_and_only_beauty() -> None:
    normal = for_variety("BEAUTY")
    shadow = for_variety("BEAUTY", shadow_temporal=True)

    assert normal.cv_outer_strategy is None
    assert normal.training_window_days is None
    assert shadow_temporal_varieties() == {"BEAUTY"}
    assert shadow.cv_outer_strategy == "temporal_year"
    assert shadow.training_window_days == 365
    assert shadow.shadow_temporal_max_mape == 35.0


def test_recent_training_window_uses_only_past_365_days() -> None:
    X = pd.DataFrame(
        {
            "FECHA": pd.to_datetime(
                ["2023-12-31", "2024-01-01", "2024-06-01", "2024-12-31"]
            )
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    X_recent, y_recent, keep = recent_training_window(
        X,
        y,
        window_days=365,
        reference_date=pd.Timestamp("2024-12-31"),
    )

    assert X_recent["FECHA"].min() == pd.Timestamp("2024-01-01")
    assert y_recent.tolist() == [2.0, 3.0, 4.0]
    assert keep.tolist() == [False, True, True, True]


def test_cv_override_is_temporal_without_changing_global_default() -> None:
    dates = pd.date_range("2022-01-01", "2025-12-01", freq="MS")
    X = pd.DataFrame(
        {
            "FECHA": dates,
            "ANIO": dates.year,
            "FUNDO": "F1",
            "FORMATO": "A",
        }
    )

    outer, inner, _, strategy = build_cv_splitters(
        X,
        outer_folds=2,
        inner_folds=2,
        random_state=42,
        outer_strategy="temporal_year",
    )

    assert isinstance(outer, TemporalYearSplit)
    assert isinstance(inner, PurgedDateSplit)
    assert strategy == "temporal_year"


def test_cli_exposes_shadow_as_explicit_flag() -> None:
    args = parse_args(
        ["--varieties", "BEAUTY", "--tuning", "dev", "--shadow-temporal-window"]
    )
    assert args.shadow_temporal_window is True


def test_shadow_candidate_can_never_register() -> None:
    champion = ModelResult(
        model_type="lgb",
        metrics={"nested_cv_gap_mean": 0.1, "nested_cv_mae_mean": 1.0},
        best_params={},
        mlflow_run_id="12345678",
        pipeline_path="model.joblib",
        elapsed_seconds=10.0,
        business_metrics_oof={"mape": 20.0},
    )
    args = argparse.Namespace(
        tuning="dev",
        register_model=True,
        shadow_temporal_window=True,
    )

    assert not apply_quality_gate(
        champion,
        args,
        "BEAUTY",
        logging.getLogger(__name__),
    )
