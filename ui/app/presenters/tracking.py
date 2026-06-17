"""Presenter de **Seguimiento / Precisión**.

Agrega los pares proyectado↔real (`AccuracyPoint`) en los view-models de la
página: KPIs + veredicto, diagnóstico datos-vs-modelo, series de gráficos,
cierre semanal y tabla de detalle. También expone helpers puros de filtrado y
de la grilla de datos reales. Sin `streamlit`: la vista solo pinta/grafica.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from app.dependencies import get_cached_catalogs, get_tracking_service
from app.schemas import AccuracyPoint
from app.services import TrackingService, forecast_verdict

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Orden de columnas de la grilla de datos REALES (= inputs del pronóstico + KG/JR_H).
REAL_COLS = [
    "FUNDO", "FORMATO", "FECHA", "KG/HA", "KG/JR_H",
    "DPC", "%INDUS", "P/BAYA", "HA", "DIA_COSECHA",
]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def iso_week(fecha: str) -> str:
    """`'2026-04-10'` → `'2026-W15'`; `'—'` si no parsea."""
    try:
        iso = date.fromisoformat(fecha[:10]).isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    except (ValueError, TypeError):
        return "—"


# ── Filtros (la UI multiselect vive en la vista; acá solo opciones + criba) ──
def filter_options(points: list[AccuracyPoint]) -> tuple[list[str], list[str]]:
    """Valores únicos de FUNDO y SEMANA ISO presentes en los puntos."""
    fundos = sorted({p.fundo for p in points})
    weeks = sorted({iso_week(p.fecha) for p in points})
    return fundos, weeks


def apply_filters(
    points: list[AccuracyPoint], sel_fundos: list[str], sel_weeks: list[str]
) -> list[AccuracyPoint]:
    return [
        p for p in points
        if p.fundo in sel_fundos and iso_week(p.fecha) in sel_weeks
    ]


# ── Grilla / plantilla de datos reales (lógica pura, sin st) ─────────────────
def build_real_template_xlsx() -> bytes:
    """Plantilla del Excel de datos REALES (= inputs del pronóstico + KG/JR_H)."""
    cat = get_cached_catalogs()
    fundo = cat.fundos[0] if cat.fundos else "C5"
    fmt = cat.formato_default or (cat.formatos[0] if cat.formatos else "GRANEL")
    df = pd.DataFrame(
        {
            "FECHA": [date.today().isoformat()],
            "FUNDO": [fundo],
            "FORMATO": [fmt],
            "KG/HA": [4800.0],
            "KG/JR_H": [3.9],        # ← productividad REAL realizada (obligatoria)
            "DPC": [118.0],          # opcionales: habilitan descomposición exacta
            "HA": [9.0],
            "DIA_COSECHA": [32],
            "%INDUS": [5.0],
            "P/BAYA": [2.4],
        }
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="reales")
    buf.seek(0)
    return buf.getvalue()


def real_grid_from_history(variety: str) -> pd.DataFrame:
    obs = get_tracking_service().list_history(variety)
    if not obs:
        return pd.DataFrame(columns=REAL_COLS)
    df = pd.DataFrame([
        {
            "FUNDO": o.fundo, "FORMATO": o.formato, "FECHA": o.fecha[:10],
            "KG/HA": o.kg_ha, "KG/JR_H": o.kg_jr_h, "DPC": o.dpc,
            "%INDUS": o.indus_pct, "P/BAYA": o.p_baya, "HA": o.ha,
            "DIA_COSECHA": o.dia_cosecha,
        }
        for o in obs
    ])[REAL_COLS]
    # DateColumn del data_editor exige dtype fecha/datetime, no string.
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    return df


def coerce_real(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza FECHA a ISO y descarta filas sin FUNDO/KG_JR_H."""
    df = df.copy()

    def _iso(v: object) -> str | None:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return pd.to_datetime(v).date().isoformat()
        except Exception:
            return str(v)

    df["FECHA"] = df["FECHA"].map(_iso)
    df = df[df["FUNDO"].notna() & df["KG/JR_H"].notna()]
    return df.reset_index(drop=True)


# ── View-models de análisis ──────────────────────────────────────────────────
@dataclass(frozen=True)
class KpiVM:
    n_points: int
    mae: float
    mape: float
    has_mape: bool
    bias: float
    verdict_status: str    # ok / warning / alert
    verdict_msg: str
    sesgo_dir: str         # sobreestima / subestima
    mape_variant: str
    bias_variant: str


@dataclass(frozen=True)
class DecompVM:
    available: bool
    mean_data: float = 0.0
    mean_model: float = 0.0
    predominant: str = "data"   # data / model
    data_variant: str = "primary"
    model_variant: str = "primary"


@dataclass(frozen=True)
class ChartsVM:
    fechas: list[str] = field(default_factory=list)
    pred: list[float] = field(default_factory=list)
    real: list[float] = field(default_factory=list)
    err: list[float] = field(default_factory=list)
    has_decomp: bool = False
    decomp_data: list[float] = field(default_factory=list)
    decomp_model: list[float] = field(default_factory=list)
    decomp_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WeeklyVM:
    has_weeks: bool
    weeks: list[str] = field(default_factory=list)
    proj_sums: list[float] = field(default_factory=list)
    real_sums: list[float] = field(default_factory=list)
    table_rows: list[dict] = field(default_factory=list)
    # KPIs de cierre (None cuando no hay semanas con real > 0)
    cumplimiento: float | None = None
    cumpl_variant: str = "success"
    mejor_week: str = ""
    mejor_pct: float = 0.0
    mejor_n: int = 0
    peor_week: str = ""
    peor_pct: float = 0.0
    peor_n: int = 0


def build_kpi_vm(points: list[AccuracyPoint]) -> KpiVM:
    errs = [p.error_total for p in points]
    mapes = [p.abs_pct_error for p in points if p.abs_pct_error is not None]
    mae = _mean([abs(e) for e in errs])
    mape = _mean(mapes)
    bias = _mean(errs)
    status, msg = forecast_verdict(mape, bias)
    return KpiVM(
        n_points=len(points),
        mae=mae,
        mape=mape,
        has_mape=bool(mapes),
        bias=bias,
        verdict_status=status,
        verdict_msg=msg,
        sesgo_dir="sobreestima" if bias > 0 else "subestima",
        mape_variant="warning" if mape > 15 else "success",
        bias_variant="warning" if abs(bias) > max(mae * 0.5, 1e-9) else "success",
    )


def build_decomp_vm(points: list[AccuracyPoint]) -> DecompVM:
    dec = [p for p in points if p.pred_on_real is not None]
    if not dec:
        return DecompVM(available=False)
    mean_data = _mean([abs(p.error_data) for p in dec])    # type: ignore[arg-type]
    mean_model = _mean([abs(p.error_model) for p in dec])  # type: ignore[arg-type]
    data_predom = mean_data > mean_model
    return DecompVM(
        available=True,
        mean_data=mean_data,
        mean_model=mean_model,
        predominant="data" if data_predom else "model",
        data_variant="warning" if data_predom else "primary",
        model_variant="warning" if not data_predom else "primary",
    )


def build_charts_vm(points: list[AccuracyPoint]) -> ChartsVM:
    dec = [p for p in points if p.pred_on_real is not None]
    return ChartsVM(
        fechas=[p.fecha for p in points],
        pred=[p.pred_original for p in points],
        real=[p.real for p in points],
        err=[p.error_total for p in points],
        has_decomp=bool(dec),
        decomp_data=[p.error_data for p in dec],    # type: ignore[misc]
        decomp_model=[p.error_model for p in dec],  # type: ignore[misc]
        decomp_labels=[f"{p.fecha} · {p.fundo}" for p in dec],
    )


def _weekly_semaforo(pct_diff: float | None) -> str:
    if pct_diff is None:
        return "—"
    abs_pct = abs(pct_diff)
    return "✅" if abs_pct <= 10 else "🟡" if abs_pct <= 20 else "🔴"


def build_weekly_vm(points: list[AccuracyPoint]) -> WeeklyVM:
    weeks = TrackingService.weekly_aggregate(points)
    if not weeks:
        return WeeklyVM(has_weeks=False)

    table_rows = [
        {
            "": _weekly_semaforo(w.pct_diff),
            "Semana": w.week,
            "Proyectado": round(w.proj_sum, 1),
            "Real": round(w.real_sum, 1),
            "Cumplimiento Δ%": f"{w.pct_diff:+.1f}%" if w.pct_diff is not None else "—",
            "Puntos": w.n,
        }
        for w in weeks
    ]
    vm = dict(
        has_weeks=True,
        weeks=[w.week for w in weeks],
        proj_sums=[w.proj_sum for w in weeks],
        real_sums=[w.real_sum for w in weeks],
        table_rows=table_rows,
    )

    diffs_w = [(w, w.pct_diff) for w in weeks if w.pct_diff is not None]
    if not diffs_w:
        return WeeklyVM(**vm)

    total_proj = sum(w.proj_sum for w in weeks)
    total_real = sum(w.real_sum for w in weeks)
    cumplimiento = (total_real / total_proj * 100.0) if total_proj else 0.0
    mejor = min(diffs_w, key=lambda t: abs(t[1]))
    peor = max(diffs_w, key=lambda t: abs(t[1]))
    return WeeklyVM(
        **vm,
        cumplimiento=cumplimiento,
        cumpl_variant="success" if abs(100 - cumplimiento) < 10 else "warning",
        mejor_week=mejor[0].week, mejor_pct=mejor[1], mejor_n=mejor[0].n,
        peor_week=peor[0].week, peor_pct=peor[1], peor_n=peor[0].n,
    )


def build_table_rows(points: list[AccuracyPoint]) -> list[dict]:
    rows: list[dict] = []
    for p in points:
        row = {
            "Fecha": p.fecha,
            "Fundo": p.fundo,
            "Formato": p.formato,
            "Proyectado": round(p.pred_original, 2),
            "Real": round(p.real, 2),
            "Error": round(p.error_total, 2),
            "Error %": round(p.abs_pct_error, 1) if p.abs_pct_error is not None else None,
        }
        if p.pred_on_real is not None:
            row["Err datos"] = round(p.error_data, 2)    # type: ignore[arg-type]
            row["Err modelo"] = round(p.error_model, 2)  # type: ignore[arg-type]
        rows.append(row)
    return rows
