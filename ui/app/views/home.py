"""Dashboard ejecutivo: estado del sistema + calidad de modelos + accesos.

Visión gerencial: abre con lo que un directivo necesita de un vistazo
(¿está operativo? ¿qué tan buenos son los modelos? ¿a dónde voy ahora?),
sin saturar de gráficos. El detalle comparativo de modelos vive en la
página **Modelos**; la precisión real del pronóstico, en **Seguimiento**.

Vista delgada: la agregación (estado, precisión en vivo, calidad) vive en
`app.presenters.home`; acá solo se compone el UI.
"""

from __future__ import annotations

import streamlit as st
from app.components.charts import (
    build_pred_vs_real_line,
    build_residual_bars,
    build_table_with_bars,
)
from app.components.layout import empty_state, insight_card, kpi_card, page_header, section_title
from app.presenters.home import build_live_data_vm, build_overview_vm, build_quality_vm


def _render_live_data() -> None:
    """Dashboard VIVO: veredicto + tendencias proy-vs-real con la data ya cargada.

    No exige ir a Seguimiento: si ya hay pronósticos + reales, el veredicto
    gerencial y la tendencia aparecen aquí por defecto.
    """
    section_title("📈 Precisión en producción (en vivo)")
    vm = build_live_data_vm()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Pronósticos", str(vm.total_fc), icon="📅", variant="primary")
    with c2:
        kpi_card("Pares real↔proy", str(vm.n_points), icon="🔗", variant="accent")
    with c3:
        kpi_card("MAPE real", f"{vm.mape:.1f}%" if vm.has_points else "—", icon="📐",
                 variant=vm.mape_variant)
    with c4:
        kpi_card("Sesgo", f"{vm.bias:+.2f}" if vm.has_points else "—", icon="🎯",
                 variant=vm.bias_variant)

    if not vm.has_points:
        empty_state(
            "Sin datos reales para medir precisión",
            help="Cargá la cosecha real en <strong>Seguimiento</strong> "
                 "y el veredicto se activa solo.",
            icon="📭",
        )
        return

    # ── Veredicto gerencial — primera respuesta: ¿confío en el pronóstico? ──
    {"ok": st.success, "warning": st.warning, "alert": st.error}[vm.verdict_status](vm.verdict_msg)

    st.markdown(f"**Proyectado vs Real — {vm.top_variety}** · {vm.n_top} pronósticos comparados")
    _g1, _g2 = st.columns(2)
    with _g1:
        st.plotly_chart(
            build_pred_vs_real_line(vm.top_fechas, vm.top_pred, vm.top_real),
            use_container_width=True,
        )
    with _g2:
        st.plotly_chart(
            build_residual_bars(vm.top_fechas, vm.top_err),
            use_container_width=True,
        )
    st.page_link("views/tracking.py",
                 label="Ver análisis completo (paridad, cierre semanal, diagnóstico) →",
                 icon=":material/monitoring:")


page_header("Dashboard", "Resumen ejecutivo del sistema de pronósticos", "📊")

_overview = build_overview_vm()

if not _overview.is_online:
    st.warning(
        "⚠️ Backend desconectado — los indicadores pueden estar incompletos. "
        "Verifica la conexión."
    )

# ── Estado general ──────────────────────────────────────────────────────
section_title("📊 Estado general")
_k1, _k2, _k3, _k4 = st.columns(4)
with _k1:
    kpi_card(
        "Backend",
        "Online" if _overview.is_online else "Offline",
        icon="🟢" if _overview.is_online else "🔴",
        variant="success" if _overview.is_online else "danger",
    )
with _k2:
    kpi_card("Variedades", str(_overview.total), icon="🍇", variant="primary")
with _k3:
    kpi_card("Modelos", f"{_overview.n_loaded}/{_overview.total}", icon="🤖",
             variant=_overview.models_variant)
with _k4:
    kpi_card(
        "MLflow",
        "Conectado" if _overview.mlflow_ok else "Desconectado",
        icon="🧪",
        variant="success" if _overview.mlflow_ok else "danger",
    )

# ── Precisión en producción (data viva, por defecto) ────────────────────
_render_live_data()

# ── Calidad de modelos (métricas reales out-of-fold) ────────────────────
_quality = build_quality_vm(_overview.loaded)
if _quality.has_models:
    section_title("🏅 Calidad de modelos")
    _c1, _c2, _c3, _c4 = st.columns(4)
    with _c1:
        insight_card("Mejor R²", f"{_quality.best_r2_val:.3f}",
                     f"Variedad <strong>{_quality.best_r2_name}</strong>", "success")
    with _c2:
        insight_card("R² promedio", f"{_quality.avg_r2:.3f}",
                     f"Sobre {_quality.n_loaded} modelo(s) cargado(s)", "primary")
    with _c3:
        insight_card("MAE promedio", f"{_quality.avg_mae:.2f}", "Menor es mejor — kg/h", "accent")
    with _c4:
        insight_card("Mayor MAE", f"{_quality.worst_mae_val:.2f}",
                     f"Variedad <strong>{_quality.worst_mae_name}</strong> — revisar", "warning")

    # Un solo gráfico: ranking por R² (el detalle comparativo va en "Modelos").
    section_title("📋 Ranking de variedades")
    st.plotly_chart(
        build_table_with_bars(
            _quality.ranking_names, _quality.ranking_mae, _quality.ranking_r2,
            {
                "names": _quality.ranking_names,
                "status": ["✅" for _ in _quality.ranking_names],
                "mae": [f"{v:.2f}" for v in _quality.ranking_mae],
                "r2": [f"{v:.3f}" for v in _quality.ranking_r2],
            },
        ),
        use_container_width=True,
    )
else:
    empty_state(
        "Aún no hay modelos cargados",
        help="Se cargan bajo demanda al predecir, "
             "o con <strong>Recargar Modelos</strong> en el menú lateral.",
        icon="🤖",
    )

# ── Accesos directos (orientado al flujo de pronósticos) ────────────────
section_title("🚀 Accesos directos")
_n1, _n2, _n3 = st.columns(3)
with _n1:
    st.page_link("views/forecast.py", label="Pronosticar (individual / lote)",
                 icon=":material/edit_note:", use_container_width=True)
with _n2:
    st.page_link("views/tracking.py", label="Seguimiento de precisión",
                 icon=":material/monitoring:", use_container_width=True)
with _n3:
    st.page_link("views/models.py", label="Detalle de modelos",
                 icon=":material/model_training:", use_container_width=True)
