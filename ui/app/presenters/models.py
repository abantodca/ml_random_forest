"""Presenter de la página **Modelos** — view-models del dashboard MLOps.

Toma los `VarietyViewModel` crudos (métricas OOF, best-params del registry) y
los agrega en estructuras listas para render: cobertura, salud/sobreajuste y
los hiperparámetros del campeón. Sin `streamlit`: lógica pura y testeable; la
vista solo pinta los campos (incluidas las `*_variant` ya resueltas).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas import VarietyViewModel

# ── Umbrales de sobreajuste ───────────────────────────────────────────────────
# MAE gap: (test_mae - train_mae) / test_mae  (positivo = train < test → sobreajuste)
# R² gap:  train_r2 - test_r2                 (positivo = tren mejor que test)
_OVERFIT_MAE_THR_WARN = 0.15   # gap relativo MAE > 15 % → warning
_OVERFIT_MAE_THR_BAD = 0.30    # gap relativo MAE > 30 % → danger
_OVERFIT_R2_THR_WARN = 0.08    # gap absoluto R² > 0.08 → warning
_OVERFIT_R2_THR_BAD = 0.15     # gap absoluto R² > 0.15 → danger


def short_param(key: str) -> str:
    """`regressor__regressor__n_estimators` → `n_estimators` (legible)."""
    return (
        key.replace("regressor__regressor__", "")
        .replace("regressor__", "")
        .replace("preprocessor__", "")
        .replace("__", " · ")
    )


def _overfit_badge(mae_gap_rel: float | None, r2_gap: float | None) -> tuple[str, str]:
    """Retorna (label, variant) para el badge de salud del modelo."""
    if mae_gap_rel is None and r2_gap is None:
        return "Sin datos train", "info"
    # Combina ambos gaps: el peor determina el nivel
    is_bad = (
        (mae_gap_rel is not None and mae_gap_rel > _OVERFIT_MAE_THR_BAD)
        or (r2_gap is not None and r2_gap > _OVERFIT_R2_THR_BAD)
    )
    is_warn = (
        (mae_gap_rel is not None and mae_gap_rel > _OVERFIT_MAE_THR_WARN)
        or (r2_gap is not None and r2_gap > _OVERFIT_R2_THR_WARN)
    )
    if is_bad:
        return "Sobreajuste", "danger"
    if is_warn:
        return "Sobreajuste leve", "warning"
    return "Generaliza bien", "success"


def _train_test_gaps(
    metrics: dict,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    """Extrae (train_mae, test_mae, mae_gap_rel, train_r2, test_r2, r2_gap).

    Devuelve None en cada posición cuando el metric no está disponible.
    """
    def _f(v: object) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    tr_mae = _f(metrics.get("train_mae"))
    te_mae = _f(metrics.get("test_mae"))
    tr_r2 = _f(metrics.get("train_r2"))
    te_r2 = _f(metrics.get("test_r2"))

    mae_gap_rel: float | None = None
    if tr_mae is not None and te_mae is not None and te_mae != 0:
        # gap = cuánto mejor (o peor) entrena vs generaliza: train < test → sobreajuste
        mae_gap_rel = (te_mae - tr_mae) / te_mae  # positivo cuando test > train

    r2_gap: float | None = None
    if tr_r2 is not None and te_r2 is not None:
        r2_gap = tr_r2 - te_r2  # positivo cuando train > test

    return tr_mae, te_mae, mae_gap_rel, tr_r2, te_r2, r2_gap


# ── View-models ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CoverageVM:
    total: int
    n_with_model: int
    pending: int
    coverage_pct: float
    coverage_variant: str          # success / warning / danger
    with_model_variant: str        # success / accent


@dataclass(frozen=True)
class GapsVM:
    has_train: bool
    train_mae: float | None
    test_mae: float | None
    mae_gap_rel: float | None
    train_r2: float | None
    test_r2: float | None
    r2_gap: float | None
    mae_card_variant: str
    r2_card_variant: str
    gap_card_variant: str


@dataclass(frozen=True)
class ParamGroupsVM:
    has_any: bool
    reg_rows: list[dict] = field(default_factory=list)   # regresor + otros (mezclados)
    prep_rows: list[dict] = field(default_factory=list)  # preprocesador


@dataclass(frozen=True)
class ModelDetailVM:
    vm: VarietyViewModel
    badge_label: str
    badge_variant: str
    r2_variant: str
    mape_variant: str
    gaps: GapsVM
    params: ParamGroupsVM


def build_coverage_vm(varieties: list[VarietyViewModel]) -> CoverageVM:
    total = len(varieties)
    n = sum(1 for v in varieties if v.model_loaded)
    pending = total - n
    coverage = (n / total * 100) if total else 0.0
    cov_variant = "success" if coverage >= 80 else "warning" if coverage >= 50 else "danger"
    return CoverageVM(
        total=total,
        n_with_model=n,
        pending=pending,
        coverage_pct=coverage,
        coverage_variant=cov_variant,
        with_model_variant="success" if n == total and total else "accent",
    )


def _build_gaps_vm(metrics: dict) -> GapsVM:
    tr_mae, te_mae, mae_gap_rel, tr_r2, te_r2, r2_gap = _train_test_gaps(metrics)
    return GapsVM(
        has_train=tr_mae is not None or tr_r2 is not None,
        train_mae=tr_mae,
        test_mae=te_mae,
        mae_gap_rel=mae_gap_rel,
        train_r2=tr_r2,
        test_r2=te_r2,
        r2_gap=r2_gap,
        mae_card_variant=(
            "success" if mae_gap_rel is not None and mae_gap_rel < _OVERFIT_MAE_THR_WARN
            else "warning"
        ),
        r2_card_variant=(
            "success" if r2_gap is not None and r2_gap < _OVERFIT_R2_THR_WARN
            else "warning"
        ),
        gap_card_variant=(
            "success" if mae_gap_rel is not None and abs(mae_gap_rel) < _OVERFIT_MAE_THR_WARN
            else "warning"
        ),
    )


def _build_params_vm(best_params: dict) -> ParamGroupsVM:
    params = {k: v for k, v in best_params.items() if k != "model_type"}
    if not params:
        return ParamGroupsVM(has_any=False)
    prep = {k: v for k, v in params.items() if k.startswith("preprocessor__")}
    reg = {k: v for k, v in params.items()
           if k.startswith(("regressor__", "classifier__"))}
    other = {k: v for k, v in params.items() if k not in prep and k not in reg}

    def _rows(d: dict) -> list[dict]:
        return [{"Hiperparámetro": short_param(k), "Valor": v} for k, v in d.items()]

    return ParamGroupsVM(
        has_any=True,
        reg_rows=_rows({**reg, **other}),
        prep_rows=_rows(prep),
    )


def build_detail_vm(vm: VarietyViewModel) -> ModelDetailVM:
    """Arma el detalle del modelo seleccionado: salud, gaps train/test y params."""
    gaps = _build_gaps_vm(vm.metrics)
    badge_label, badge_variant = _overfit_badge(gaps.mae_gap_rel, gaps.r2_gap)
    return ModelDetailVM(
        vm=vm,
        badge_label=badge_label,
        badge_variant=badge_variant,
        r2_variant="success" if vm.r2 >= 0.7 else "warning",
        mape_variant="success" if vm.mape <= 15 else "warning",
        gaps=gaps,
        params=_build_params_vm(vm.best_params),
    )
