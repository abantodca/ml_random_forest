"""Panel de confiabilidad de la predicción (drift).

Componente reutilizable que muestra:
  - Medidor del `score` global (semaforizado)
  - Tarjeta de insight con el verdict del backend
  - Tabla por feature: valor enviado, rango usual, z-score / frecuencia,
    estado.

Si el backend no devolvió drift (modelo legacy sin baseline disponible)
el panel se renderiza como un caption neutro en lugar de fallar.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts.gauge import build_drift_gauge
from app.components.layout.insight_card import insight_card
from app.schemas import BatchDriftReport, DriftReport

# Rango z-score que se considera "normal" (espeja Z_OK_THRESHOLD del backend).
_Z_OK = 1.0
_STATUS_LABEL = {
    "ok": "🟢 OK",
    "warning": "🟡 Atención",
    "alert": "🔴 Fuera de rango",
}
_STATUS_VARIANT = {
    "ok": "success",
    "warning": "warning",
    "alert": "danger",
}
_STATUS_HEADER = {
    "ok": "✅ Predicción confiable",
    "warning": "⚠️ Confianza moderada",
    "alert": "🚨 Predicción extrapolando",
}


def render_drift_panel(
    drift: DriftReport | None,
    *,
    title: str = "🔍 Confiabilidad de la predicción",
    expanded: bool = False,
) -> None:
    """Renderiza el panel de drift dentro de un expander."""
    if drift is None:
        st.caption(
            "ℹ️ Reporte de confiabilidad no disponible para esta variedad "
            "(modelo entrenado antes de habilitar drift detection)."
        )
        return

    badge = _STATUS_LABEL.get(drift.status, drift.status)
    header = f"{title}  ·  score={drift.score:.2f}  ·  {badge}"
    with st.expander(header, expanded=expanded):
        _render_summary(drift)
        _render_feature_table(drift)


def render_drift_summary_inline(drift: DriftReport | None) -> None:
    """Versión compacta para listas: solo gauge + verdict (sin tabla)."""
    if drift is None:
        return
    _render_summary(drift)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _render_summary(drift: DriftReport) -> None:
    col_gauge, col_text = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(
            build_drift_gauge(drift.score, drift.status),
            use_container_width=True,
        )
    with col_text:
        insight_card(
            label=_STATUS_HEADER.get(drift.status, "Confiabilidad"),
            value=drift.verdict or "—",
            meta=_format_window_meta(drift),
            variant=_STATUS_VARIANT.get(drift.status, "primary"),
        )


def _format_window_meta(drift: DriftReport) -> str:
    tw = drift.training_window
    if tw.n_samples and tw.start and tw.end:
        return f"Baseline: {tw.n_samples:,} cosechas · {tw.start} a {tw.end}"
    if tw.n_samples:
        return f"Baseline: {tw.n_samples:,} cosechas históricas"
    return ""


def _render_feature_table(drift: DriftReport) -> None:
    if not drift.per_feature:
        st.caption("Sin features evaluables para drift en este registro.")
        return

    rows = []
    for f in drift.per_feature:
        is_missing = f.value_str == "no enviado"
        estado = (
            "⚪ No enviado" if is_missing else _STATUS_LABEL.get(f.status, f.status)
        )
        rows.append({
            "Variable": f.feature,
            "Valor enviado": f.display_value,
            "Rango usual": _format_baseline_range(f),
            "z-score": f"{f.z_score:+.2f}σ" if f.z_score is not None else "—",
            "Frec. histórica": (
                f"{f.baseline_freq * 100:.1f}%"
                if f.baseline_freq is not None
                else "—"
            ),
            "Estado": estado,
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Variable": st.column_config.TextColumn(width="medium"),
            "Valor enviado": st.column_config.TextColumn(width="small"),
            "Rango usual": st.column_config.TextColumn(width="medium"),
            "z-score": st.column_config.TextColumn(width="small"),
            "Frec. histórica": st.column_config.TextColumn(width="small"),
            "Estado": st.column_config.TextColumn(width="small"),
        },
    )

    alerts = [f for f in drift.per_feature if f.status == "alert"]
    if alerts:
        bullets = "\n".join(
            f"- **{a.feature}** = {a.value_str or a.value}: {_alert_reason(a)}"
            for a in alerts
        )
        st.warning("Variables fuera de rango:\n\n" + bullets)


def _format_baseline_range(f) -> str:
    if f.baseline_p05 is not None and f.baseline_p95 is not None:
        return f"{f.baseline_p05:,.1f} a {f.baseline_p95:,.1f}"
    if f.baseline_freq is not None:
        return f"{f.baseline_freq * 100:.1f}% del baseline"
    return "—"


def _alert_reason(f) -> str:
    if f.is_unseen_category:
        return "categoría no observada en entrenamiento"
    if f.z_score is not None and abs(f.z_score) >= _Z_OK:
        return f"z={f.z_score:+.2f}σ — fuera del rango habitual"
    if f.baseline_freq is not None and f.baseline_freq < 0.01:
        return f"sólo {f.baseline_freq * 100:.1f}% del histórico"
    return "fuera del rango habitual"


# ---------------------------------------------------------------------------
# Batch drift panel (PSI + K-S + Chi²)
# ---------------------------------------------------------------------------


def render_batch_drift_panel(
    batch_drift: BatchDriftReport | None,
    *,
    title: str = "📈 Drift estadístico del lote",
    expanded: bool = True,
) -> None:
    """Panel de drift agregado para uploads/batch (PSI, K-S, Chi²).

    Si `batch_drift` es None (lote chico o variedad sin baseline),
    muestra un caption neutro.
    """
    if batch_drift is None:
        st.caption(
            "ℹ️ Drift estadístico del lote no calculado: el lote es muy "
            "pequeño (<30 filas) o la variedad no tiene baseline."
        )
        return

    badge = _STATUS_LABEL.get(batch_drift.status, batch_drift.status)
    header = (
        f"{title}  ·  PSI prom.={batch_drift.score:.3f}  ·  {badge}"
    )
    with st.expander(header, expanded=expanded):
        _render_batch_summary(batch_drift)
        _render_batch_feature_table(batch_drift)
        _render_batch_row_breakdown(batch_drift)
        _render_method_legend()


def _render_batch_summary(batch: BatchDriftReport) -> None:
    col_gauge, col_text = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(
            build_drift_gauge(batch.score, batch.status),
            use_container_width=True,
        )
    with col_text:
        meta_parts = [f"Lote: {batch.n_rows:,} filas"]
        tw = batch.training_window
        if tw.n_samples and tw.start and tw.end:
            meta_parts.append(
                f"Baseline: {tw.n_samples:,} cosechas · {tw.start} a {tw.end}"
            )
        insight_card(
            label=_STATUS_HEADER.get(batch.status, "Drift del lote"),
            value=batch.verdict or "—",
            meta="  ·  ".join(meta_parts),
            variant=_STATUS_VARIANT.get(batch.status, "primary"),
        )


def _render_batch_feature_table(batch: BatchDriftReport) -> None:
    if not batch.per_feature:
        st.caption("No hay features evaluables en el lote.")
        return

    rows = []
    for f in batch.per_feature:
        rows.append({
            "Variable": f.feature,
            "Tipo": "Numérica" if f.kind == "numeric" else "Categórica",
            "PSI": f"{f.psi:.3f}",
            "K-S p-value": (
                f"{f.ks_pvalue:.4f}" if f.ks_pvalue is not None else "—"
            ),
            "Chi² p-value": (
                f"{f.chi2_pvalue:.4f}" if f.chi2_pvalue is not None else "—"
            ),
            "Tests": f.method.upper(),
            "Estado": _STATUS_LABEL.get(f.status, f.status),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Variable": st.column_config.TextColumn(width="medium"),
            "Tipo": st.column_config.TextColumn(width="small"),
            "PSI": st.column_config.TextColumn(
                width="small",
                help="<0.10 estable | 0.10–0.25 leve | ≥0.25 severo",
            ),
            "K-S p-value": st.column_config.TextColumn(
                width="small",
                help="Kolmogorov-Smirnov: p<0.05 indica drift significativo",
            ),
            "Chi² p-value": st.column_config.TextColumn(
                width="small",
                help="Chi-cuadrado goodness-of-fit: p<0.05 indica drift",
            ),
            "Tests": st.column_config.TextColumn(width="small"),
            "Estado": st.column_config.TextColumn(width="small"),
        },
    )

    alerts = [f for f in batch.per_feature if f.status == "alert"]
    if alerts:
        bullets = []
        for a in alerts:
            reasons = []
            if a.psi >= 0.25:
                reasons.append(f"PSI={a.psi:.2f}")
            if a.ks_pvalue is not None and a.ks_pvalue < 0.05:
                reasons.append(f"K-S p={a.ks_pvalue:.4f}")
            if a.chi2_pvalue is not None and a.chi2_pvalue < 0.05:
                reasons.append(f"Chi² p={a.chi2_pvalue:.4f}")
            if a.unseen_categories > 0:
                reasons.append(
                    f"{a.unseen_categories} categoría(s) no vista(s)"
                )
            bullets.append(f"- **{a.feature}**: {', '.join(reasons)}")
        st.warning(
            "Variables con drift severo:\n\n" + "\n".join(bullets)
        )


def _render_batch_row_breakdown(batch: BatchDriftReport) -> None:
    counts = batch.row_status_counts
    total = counts.ok + counts.warning + counts.alert
    if total == 0:
        return
    st.caption(
        "**Distribución de filas por estado de drift individual:**"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "🟢 Confiables",
        counts.ok,
        f"{counts.ok / total * 100:.0f}%" if total else None,
    )
    c2.metric(
        "🟡 Atención",
        counts.warning,
        f"{counts.warning / total * 100:.0f}%" if total else None,
    )
    c3.metric(
        "🔴 Fuera de rango",
        counts.alert,
        f"{counts.alert / total * 100:.0f}%" if total else None,
    )


def _render_method_legend() -> None:
    with st.expander("📚 Cómo leer estos números", expanded=False):
        st.markdown(
            """
**PSI (Population Stability Index)** — regla de oro de la industria:
- `< 0.10` → estable, sin cambios
- `0.10 – 0.25` → cambio leve, monitorear
- `≥ 0.25` → drift severo, considerar reentrenar

**K-S (Kolmogorov-Smirnov)** — compara distribuciones numéricas continuas.
`p-value < 0.05` indica que las distribuciones son significativamente
distintas. Solo se calcula sobre variables con muestras crudas del
baseline (KG/HA).

**Chi² (chi-cuadrado goodness-of-fit)** — compara frecuencias de
categorías. `p-value < 0.05` indica que la distribución de FORMATO/FUNDO
del lote difiere significativamente del histórico.

**Combinación**: el estado final por variable usa PSI como métrica
principal y eleva el estado a "atención" si K-S o Chi² detectan drift
significativo que PSI no capturó.
            """
        )
