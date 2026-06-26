"""Presenter del **Dashboard** ejecutivo.

Agrega tres bloques para una landing gerencial: estado general (backend +
cobertura de modelos), precisión en producción (proy-vs-real ya cargada) y
calidad de modelos (métricas OOF + ranking). Sin `streamlit`: invoca los
`dependencies`/`services` y devuelve view-models con variantes ya resueltas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core import logger
from app.dependencies import (
    get_cached_accuracy,
    get_cached_health,
    get_cached_varieties,
    get_forecast_service,
    get_loaded_variety_names,
)
from app.schemas import AccuracyPoint, VarietyViewModel
from app.services import forecast_verdict


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@dataclass(frozen=True)
class OverviewVM:
    is_online: bool
    mlflow_ok: bool
    total: int
    n_loaded: int
    models_variant: str
    loaded: list[VarietyViewModel] = field(default_factory=list)


@dataclass(frozen=True)
class LiveDataVM:
    total_fc: int
    n_points: int
    mae: float
    mape: float
    bias: float
    has_points: bool
    mape_variant: str
    bias_variant: str
    verdict_status: str | None = None  # ok / warning / alert
    verdict_msg: str | None = None
    top_variety: str | None = None
    n_top: int = 0
    top_fechas: list[str] = field(default_factory=list)
    top_pred: list[float] = field(default_factory=list)
    top_real: list[float] = field(default_factory=list)
    top_err: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class QualityVM:
    has_models: bool
    best_r2_name: str = ""
    best_r2_val: float = 0.0
    worst_mae_name: str = ""
    worst_mae_val: float = 0.0
    avg_r2: float = 0.0
    avg_mae: float = 0.0
    n_loaded: int = 0
    ranking_names: list[str] = field(default_factory=list)
    ranking_mae: list[float] = field(default_factory=list)
    ranking_r2: list[float] = field(default_factory=list)


def build_overview_vm() -> OverviewVM:
    health = get_cached_health()
    varieties = get_cached_varieties()
    loaded = [v for v in varieties if v.model_loaded]
    is_online = bool(health and health.is_healthy)
    total = len(varieties)
    n_loaded = len(loaded)
    return OverviewVM(
        is_online=is_online,
        mlflow_ok=bool(health and health.mlflow_connected),
        total=total,
        n_loaded=n_loaded,
        models_variant="success" if n_loaded == total and total else "warning",
        loaded=loaded,
    )


def build_live_data_vm() -> LiveDataVM:
    """Precisión en vivo: pares proy-vs-real ya cargados (sin dry-run, barato)."""
    try:
        total_fc = get_forecast_service().list(limit=1).total
    except Exception:
        total_fc = 0

    points: list[AccuracyPoint] = []
    for name in get_loaded_variety_names():
        # Una variedad que falle (modelo recién registrado, backend ocupado)
        # no debe tumbar el resumen de las demás. Se loguea a debug para no
        # ensuciar la consola pero dejar rastro diagnosticable.
        try:
            points.extend(get_cached_accuracy(name, with_decomposition=False))
        except Exception as exc:  # noqa: BLE001 — resumen best-effort, una variedad no tumba el resto
            logger.debug("Precisión en vivo: variedad %s omitida (%s)", name, exc)

    errs = [p.error_total for p in points]
    mae = _mean([abs(e) for e in errs])
    mapes = [p.abs_pct_error for p in points if p.abs_pct_error is not None]
    mape = _mean(mapes)
    bias = _mean(errs)
    has_points = bool(points)

    base = dict(
        total_fc=total_fc,
        n_points=len(points),
        mae=mae,
        mape=mape,
        bias=bias,
        has_points=has_points,
        mape_variant="warning" if mape > 15 else "success",
        bias_variant=("warning" if has_points and abs(bias) > max(mae * 0.5, 1e-9) else "success"),
    )
    if not has_points:
        return LiveDataVM(**base)

    status, msg = forecast_verdict(mape, bias)
    by_var: dict[str, list[AccuracyPoint]] = {}
    for p in points:
        by_var.setdefault(p.variety, []).append(p)
    top = max(by_var, key=lambda v: len(by_var[v]))
    pts = sorted(by_var[top], key=lambda p: p.fecha)
    return LiveDataVM(
        **base,
        verdict_status=status,
        verdict_msg=msg,
        top_variety=top,
        n_top=len(pts),
        top_fechas=[p.fecha for p in pts],
        top_pred=[p.pred_original for p in pts],
        top_real=[p.real for p in pts],
        top_err=[p.error_total for p in pts],
    )


def build_quality_vm(loaded: list[VarietyViewModel]) -> QualityVM:
    if not loaded:
        return QualityVM(has_models=False)
    best_r2 = max(loaded, key=lambda v: v.r2)
    worst_mae = max(loaded, key=lambda v: v.mae)
    ranked = sorted(loaded, key=lambda v: v.r2, reverse=True)
    return QualityVM(
        has_models=True,
        best_r2_name=best_r2.name,
        best_r2_val=best_r2.r2,
        worst_mae_name=worst_mae.name,
        worst_mae_val=worst_mae.mae,
        avg_r2=_mean([v.r2 for v in loaded]),
        avg_mae=_mean([v.mae for v in loaded]),
        n_loaded=len(loaded),
        ranking_names=[v.name for v in ranked],
        ranking_mae=[v.mae for v in ranked],
        ranking_r2=[v.r2 for v in ranked],
    )
