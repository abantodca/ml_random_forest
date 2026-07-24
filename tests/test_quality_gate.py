from __future__ import annotations

import argparse
import logging

from src import config
from src.orchestration.quality_gate import apply_quality_gate
from src.step_05_evaluate.champion import ModelResult


def _champion(skill: float | None) -> ModelResult:
    metrics = {
        "nested_cv_gap_mean": 0.1,
        "nested_cv_mae_mean": 1.0,
    }
    if skill is not None:
        metrics["baseline_skill_mae"] = skill
    return ModelResult(
        model_type="lgb",
        metrics=metrics,
        best_params={},
        mlflow_run_id="12345678",
        pipeline_path="model.joblib",
        elapsed_seconds=10.0,
        business_metrics_oof={"mape": 10.0},
    )


def test_quality_gate_rejects_model_without_minimum_baseline_skill(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "REGISTER_ENABLED", True)
    monkeypatch.setattr(config, "EXANTE_MODE", False)
    args = argparse.Namespace(tuning="prod", register_model=True)

    assert not apply_quality_gate(
        _champion(skill=0.0),
        args,
        "POP",
        logging.getLogger(__name__),
    )


def test_quality_gate_keeps_legacy_models_without_baseline_metric(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "REGISTER_ENABLED", True)
    monkeypatch.setattr(config, "EXANTE_MODE", False)
    args = argparse.Namespace(tuning="prod", register_model=True)

    assert apply_quality_gate(
        _champion(skill=None),
        args,
        "POP",
        logging.getLogger(__name__),
    )
