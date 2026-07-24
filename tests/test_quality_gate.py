from __future__ import annotations

import argparse
import logging

from src import config, variety_config
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


def test_strict_temporal_gate_fails_closed_without_temporal_evidence(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "REGISTER_ENABLED", True)
    monkeypatch.setattr(config, "EXANTE_MODE", False)
    monkeypatch.setattr(config, "REQUIRE_TEMPORAL_GATE", True)
    args = argparse.Namespace(tuning="prod", register_model=True)

    assert not apply_quality_gate(
        _champion(skill=0.20),
        args,
        "POP",
        logging.getLogger(__name__),
    )


def test_strict_temporal_gate_requires_skill_and_acceptable_mape(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "REGISTER_ENABLED", True)
    monkeypatch.setattr(config, "EXANTE_MODE", False)
    monkeypatch.setattr(config, "REQUIRE_TEMPORAL_GATE", True)
    monkeypatch.setattr(config, "MIN_TEMPORAL_FOLDS", 2)
    monkeypatch.setattr(config, "CHAMPION_MAX_TEMPORAL_MAPE", 35.0)
    monkeypatch.setattr(config, "CHAMPION_MIN_TEMPORAL_BASELINE_SKILL", 0.0)
    args = argparse.Namespace(tuning="prod", register_model=True)

    accepted = _champion(skill=0.20)
    accepted.metrics.update(
        {
            "temporal_n_folds": 3.0,
            "temporal_mape_oof": 28.0,
            "temporal_baseline_skill_mae": 0.10,
        }
    )
    assert apply_quality_gate(
        accepted,
        args,
        "POP",
        logging.getLogger(__name__),
    )

    rejected = _champion(skill=0.20)
    rejected.metrics.update(
        {
            "temporal_n_folds": 3.0,
            "temporal_mape_oof": 28.0,
            "temporal_baseline_skill_mae": -0.01,
        }
    )
    assert not apply_quality_gate(
        rejected,
        args,
        "POP",
        logging.getLogger(__name__),
    )


def test_strict_temporal_gate_honors_per_variety_thresholds(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "REGISTER_ENABLED", True)
    monkeypatch.setattr(config, "EXANTE_MODE", False)
    monkeypatch.setattr(config, "REQUIRE_TEMPORAL_GATE", True)
    monkeypatch.setitem(
        variety_config.VARIETY_OVERRIDES,
        "TEST_VARIETY",
        {
            "max_temporal_mape": 20.0,
            "min_temporal_baseline_skill": 0.10,
            "min_temporal_folds": 3,
        },
    )
    candidate = _champion(skill=0.20)
    candidate.metrics.update(
        {
            "temporal_n_folds": 3.0,
            "temporal_mape_oof": 21.0,
            "temporal_baseline_skill_mae": 0.20,
        }
    )
    args = argparse.Namespace(tuning="prod", register_model=True)

    assert not apply_quality_gate(
        candidate,
        args,
        "TEST_VARIETY",
        logging.getLogger(__name__),
    )
