"""Presenter de **Pronosticar** — grilla editable + ejecución de lote.

Concentra la lógica no-visual del workspace: semilla/plantilla/coerción de la
grilla, ejecución del lote (predice+guarda por variedad reusando `/batch`) y
el armado del view-model de resultados (KPIs, tabla ordenada, histograma).
Sin `streamlit`: el progreso se reporta vía callback y los errores se
devuelven, de modo que la vista solo orqueste widgets y render.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from app.core import ApiConnectionError, ApiResponseError
from app.dependencies import get_forecast_service
from app.schemas import ForecastRecord
from app.services import ValidationIssue, row_to_record

DRIFT_BADGE = {"ok": "🟢 OK", "warning": "🟡 Atención", "alert": "🔴 Alerta"}

# Orden de columnas = requeridas + opcionales del validador de lotes.
COLS = [
    "VARIEDAD",
    "FECHA",
    "FUNDO",
    "FORMATO",
    "KG/HA",
    "DPC",
    "HA",
    "DIA_COSECHA",
    "%INDUS",
    "P/BAYA",
    "HORAS_EFECTIVAS",
    "EXTERNAL_ID",
]


def seed_row(variety: str, fundo: str, formato: str) -> dict:
    return {
        "VARIEDAD": variety,
        "FECHA": date.today(),
        "FUNDO": fundo,
        "FORMATO": formato,
        "KG/HA": 5000.0,
        "DPC": 120.0,
        "HA": 10.0,
        "DIA_COSECHA": 30,
        "%INDUS": None,
        "P/BAYA": None,
        "HORAS_EFECTIVAS": None,
        "EXTERNAL_ID": "",
    }


def empty_grid(variety: str, fundo: str, formato: str) -> pd.DataFrame:
    df = pd.DataFrame([seed_row(variety, fundo, formato)], columns=COLS)
    # DateColumn exige dtype fecha/datetime (no string ni object mixto).
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    return df


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza FECHA a ISO y descarta filas en blanco que agrega el editor."""
    df = df.copy()

    def _iso(v: object) -> str | None:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return pd.to_datetime(v).date().isoformat()
        except Exception:
            return str(v)

    df["FECHA"] = df["FECHA"].map(_iso)
    df = df[df["VARIEDAD"].notna() & df["KG/HA"].notna()]
    return df.reset_index(drop=True)


def normalize_upload(raw: pd.DataFrame) -> pd.DataFrame:
    """Alinea un Excel/CSV cargado al esquema de la grilla (`COLS`)."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    for c in COLS:
        if c not in raw.columns:
            raw[c] = None
    # FECHA en texto (CSV) rompería el DateColumn → a datetime.
    raw["FECHA"] = pd.to_datetime(raw["FECHA"], errors="coerce")
    return raw[COLS]


def template_xlsx(variety: str, fundo: str, formato: str) -> bytes:
    df = pd.DataFrame([seed_row(variety, fundo, formato)], columns=COLS)
    df["FECHA"] = df["FECHA"].astype(str)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="pronosticos")
    buf.seek(0)
    return buf.getvalue()


def pre_summary_text(df: pd.DataFrame) -> str | None:
    """Resumen pre-predicción: n filas + variedades involucradas. None si vacío."""
    clean = coerce(df)
    if clean.empty:
        return None
    n_rows = len(clean)
    varieties_in = sorted(clean["VARIEDAD"].dropna().unique())
    n_var = len(varieties_in)
    var_list = ", ".join(f"`{v}`" for v in varieties_in[:5])
    suffix = f" ... (+{n_var - 5} más)" if n_var > 5 else ""
    return (
        f"**{n_rows}** fila(s) listas para predecir · **{n_var}** variedad(es): {var_list}{suffix}"
    )


def issue_rows(issues: list[ValidationIssue]) -> list[dict]:
    return [
        {"Fila": i.fila, "Columna": i.columna, "Valor": i.valor, "Motivo": i.motivo} for i in issues
    ]


def affected_columns(issues: list[ValidationIssue]) -> list[str]:
    return sorted({i.columna for i in issues})


@dataclass
class BatchRunResult:
    preds: list[dict] = field(default_factory=list)
    records: list[ForecastRecord] = field(default_factory=list)
    batch_drifts: list[tuple] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def execute_batch(
    df: pd.DataFrame,
    *,
    progress_cb: Callable[[float, str], None] | None = None,
) -> BatchRunResult:
    """Predice+guarda por variedad (reusa `/batch`).

    `progress_cb(frac, texto)` reporta avance; los errores por variedad se
    acumulan en `result.errors` (la vista los pinta). Sin `streamlit`.
    """
    service = get_forecast_service()
    out = BatchRunResult()
    groups = list(df.groupby("VARIEDAD"))
    for i, (variety, g) in enumerate(groups):
        if progress_cb is not None:
            progress_cb((i + 1) / len(groups), f"{variety} ({len(g)} filas)...")
        recs = [row_to_record(r) for _, r in g.iterrows()]
        try:
            result = service.create_batch(str(variety), recs)
        except (ApiResponseError, ApiConnectionError) as exc:
            out.errors.append(f"Error en **{variety}**: {exc}")
            continue
        if result.batch_drift is not None:
            out.batch_drifts.append((str(variety), result.batch_drift))
        for it in result.items:
            badge = DRIFT_BADGE.get(it.drift.status, "—") if it.drift else "—"
            out.preds.append(
                {
                    "ID": it.id,
                    "Variedad": it.variety,
                    "Fecha": it.fecha,
                    "Fundo": it.fundo,
                    "Formato": it.formato,
                    "KG/HA": it.kg_ha,
                    "KGHORA pred": round(it.kghora_pred, 3),
                    "KGJN pred": round(it.kgjn_pred, 3) if it.kgjn_pred else None,
                    "Confiabilidad": badge,
                }
            )
            out.records.append(it)
    return out


@dataclass(frozen=True)
class ResultsVM:
    n_preds: int
    avg_kghora: float
    n_ok: int
    n_warning: int
    n_alert: int
    flagged: int
    flagged_variant: str
    alert_msg: str | None
    warning_msg: str | None
    results_df: pd.DataFrame
    hist_df: pd.DataFrame


# Prioridad de columnas en la tabla de resultados (Confiabilidad primero).
_RESULT_PRIORITY = [
    "Confiabilidad",
    "Variedad",
    "Fecha",
    "KGHORA pred",
    "KGJN pred",
    "Fundo",
    "Formato",
    "KG/HA",
    "ID",
]


def build_results_vm(preds: list[dict], records: list[ForecastRecord]) -> ResultsVM:
    kghoras = [p["KGHORA pred"] for p in preds]
    counts = {"ok": 0, "warning": 0, "alert": 0}
    for r in records:
        if r.drift and r.drift.status in counts:
            counts[r.drift.status] += 1
    flagged = counts["warning"] + counts["alert"]

    alert_msg = warning_msg = None
    if counts["alert"] > 0:
        alert_msg = (
            f"**{counts['alert']} registro(s) con alerta de drift 🔴** — "
            "los inputs están muy lejos del histórico de entrenamiento. "
            "Revisá la sección «Drift» abajo antes de usar estos pronósticos."
        )
    elif counts["warning"] > 0:
        warning_msg = (
            f"**{counts['warning']} registro(s) con atención 🟡** — "
            "algunos inputs se alejan del rango usual. Revisá el detalle de drift."
        )

    results_df = pd.DataFrame(preds)
    ordered = [c for c in _RESULT_PRIORITY if c in results_df.columns] + [
        c for c in results_df.columns if c not in _RESULT_PRIORITY
    ]
    results_df = results_df[ordered] if ordered else results_df

    hist_df = pd.DataFrame([{"variedad": r.variety, "kghora_pred": r.kghora_pred} for r in records])

    return ResultsVM(
        n_preds=len(preds),
        avg_kghora=(sum(kghoras) / len(kghoras)) if kghoras else 0.0,
        n_ok=counts["ok"],
        n_warning=counts["warning"],
        n_alert=counts["alert"],
        flagged=flagged,
        flagged_variant="warning" if flagged else "success",
        alert_msg=alert_msg,
        warning_msg=warning_msg,
        results_df=results_df,
        hist_df=hist_df,
    )
