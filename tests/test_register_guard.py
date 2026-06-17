"""P0.5 — guard de registro (incidente 2026-06-13: dev EXANTE registro v2).

`_apply_quality_gate` debe devolver False (no registrar) cuando:
  - tuning == smoke (regla previa),
  - REGISTER_ENABLED=0 (guard explicito),
  - EXANTE_MODE=1 (flag experimental: su campeon no va al Registry).
"""
from __future__ import annotations

from argparse import Namespace

import src.config as config
from src.orchestration.variety_runner import _apply_quality_gate
from src.step_05_evaluate.champion import ModelResult


class _LoggerNulo:
    def info(self, *a, **k): ...
    def warning(self, *a, **k): ...


def _champion_sano() -> ModelResult:
    """Campeon que pasa todos los gates de calidad (MAPE bajo, gap chico)."""
    return ModelResult(
        model_type="lgb",
        metrics={"nested_cv_gap_mean": 0.05, "nested_cv_mae_mean": 1.0},
        best_params={},
        mlflow_run_id="run_x",
        pipeline_path="",
        elapsed_seconds=10.0,
        business_metrics_oof={"mape": 14.0},
    )


def _args(tuning="dev"):
    return Namespace(tuning=tuning, register_model=True)


def test_campeon_sano_en_dev_si_registra():
    assert _apply_quality_gate(_champion_sano(), _args(), "POP", _LoggerNulo())


def test_smoke_nunca_registra():
    assert not _apply_quality_gate(
        _champion_sano(), _args(tuning="smoke"), "POP", _LoggerNulo()
    )


def test_register_enabled_0_bloquea(monkeypatch):
    monkeypatch.setattr(config, "REGISTER_ENABLED", False)
    assert not _apply_quality_gate(
        _champion_sano(), _args(), "POP", _LoggerNulo()
    )


def test_exante_mode_bloquea_aunque_pase_el_gate(monkeypatch):
    monkeypatch.setattr(config, "EXANTE_MODE", True)
    assert not _apply_quality_gate(
        _champion_sano(), _args(), "POP", _LoggerNulo()
    )
