"""Modelos — dashboard MLOps: rendimiento + hiperparámetros del campeón.

Para el experto en MLOps: métricas out-of-fold, versión del registry y los
best-params del modelo ganador por variedad (cards + tabla), con filtro.
Overfit gap: compara train vs test MAE/R² cuando el backend los expone.

Vista delgada: la agregación (cobertura, salud/sobreajuste, agrupado de
hiperparámetros) vive en `app.presenters.models`; acá solo se compone el UI.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from app.components.charts import build_metrics_heatmap, build_simple_gauge
from app.components.layout import empty_state, insight_card, kpi_card, page_header, section_title
from app.dependencies import get_cached_varieties
from app.presenters.models import build_coverage_vm, build_detail_vm

page_header("Modelos", "Rendimiento e hiperparámetros de los modelos MLflow", "📈")

_varieties = get_cached_varieties()
if not _varieties:
    st.warning("No se pudo obtener información de variedades.")
    st.stop()

_cov = build_coverage_vm(_varieties)
_with_model = [v for v in _varieties if v.model_loaded]

# ── Resumen ──────────────────────────────────────────────────────────────
section_title("📊 Resumen")
_c1, _c2, _c3, _c4 = st.columns(4)
with _c1:
    kpi_card("Total Variedades", str(_cov.total), icon="🍇", variant="primary")
with _c2:
    kpi_card("Con modelo", str(_cov.n_with_model), icon="🤖", variant=_cov.with_model_variant)
with _c3:
    kpi_card("Pendientes", str(_cov.pending), icon="⏳",
             variant="warning" if _cov.pending else "success")
with _c4:
    kpi_card("Cobertura", f"{_cov.coverage_pct:.0f}%", icon="📈", variant=_cov.coverage_variant)

_gc, _gnotes = st.columns([3, 2], vertical_alignment="center")
with _gc:
    st.plotly_chart(build_simple_gauge(_cov.coverage_pct), use_container_width=True)
with _gnotes:
    st.markdown(
        f"""
        <div class="ctx-panel">
            <div class="ctx-label">Contexto</div>
            <p class="ctx-line"><strong>{_cov.n_with_model}</strong> de
               <strong>{_cov.total}</strong> variedades tienen modelo entrenado en el registry.</p>
            <p class="ctx-line ctx-line--muted">Las pendientes se entrenan con el
               trainer; los modelos se cargan bajo demanda al predecir.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not _with_model:
    empty_state(
        "Aún no hay modelos entrenados",
        "Entrená con el trainer para ver métricas, salud e hiperparámetros.",
        icon="🤖",
    )
    st.stop()

# ── Comparativo (out-of-fold) ─────────────────────────────────────────────
if len(_with_model) >= 2:
    section_title("🔬 Comparativo de métricas (out-of-fold)")
    st.plotly_chart(
        build_metrics_heatmap(
            [v.name for v in _with_model],
            [v.mae for v in _with_model],
            [v.r2 for v in _with_model],
            mape_vals=[v.mape for v in _with_model],
        ),
        use_container_width=True,
    )

# ── Detalle por variedad (filtro + cards + tabla de best-params) ──────────
section_title("🔎 Detalle por variedad")
_sel = st.selectbox("Variedad", sorted(v.name for v in _with_model))
_vm = next(v for v in _with_model if v.name == _sel)
_detail = build_detail_vm(_vm)

_m1, _m2, _m3, _m4, _m5 = st.columns(5)
with _m1:
    insight_card("Algoritmo", _vm.model_type.upper(),
                 f"Versión <strong>v{_vm.version or '—'}</strong>", "primary")
with _m2:
    insight_card("Test R²", f"{_vm.r2:.3f}", "Varianza explicada (OOF)", _detail.r2_variant)
with _m3:
    insight_card("Test MAE", f"{_vm.mae:.3f}", "Error absoluto medio (OOF)", "accent")
with _m4:
    insight_card("Test MAPE", f"{_vm.mape:.1f}%", "Error porcentual (OOF)", _detail.mape_variant)
with _m5:
    insight_card("Salud", _detail.badge_label, "Brecha train vs test", _detail.badge_variant)

# ── Overfit gap (train vs test) ───────────────────────────────────────────
_gaps = _detail.gaps
if _gaps.has_train:
    section_title("⚖️ Brecha train vs test")
    _g1, _g2, _g3 = st.columns(3)
    with _g1:
        if _gaps.train_mae is not None and _gaps.test_mae is not None:
            insight_card("MAE train", f"{_gaps.train_mae:.3f}",
                         f"test: {_gaps.test_mae:.3f}", _gaps.mae_card_variant)
    with _g2:
        if _gaps.train_r2 is not None and _gaps.test_r2 is not None:
            insight_card("R² train", f"{_gaps.train_r2:.3f}",
                         f"test: {_gaps.test_r2:.3f}", _gaps.r2_card_variant)
    with _g3:
        if _gaps.mae_gap_rel is not None:
            insight_card("Gap MAE relativo", f"{_gaps.mae_gap_rel * 100:+.1f}%",
                         "positivo → train < test (sobreajuste)", _gaps.gap_card_variant)
else:
    st.caption(
        "ℹ️ Métricas `train_mae`/`train_r2` no disponibles en este registro. "
        "El análisis de brecha requiere que el trainer exponga métricas de entrenamiento."
    )

# ── Hiperparámetros del campeón (agrupados por etapa) ─────────────────────
section_title("⚙️ Hiperparámetros del campeón")
_params = _detail.params
if _params.has_any:
    if _params.reg_rows:
        st.markdown("**Regresor / Modelo**")
        st.dataframe(pd.DataFrame(_params.reg_rows), use_container_width=True, hide_index=True)
    if _params.prep_rows:
        with st.expander("Preprocesador", expanded=False):
            st.dataframe(pd.DataFrame(_params.prep_rows), use_container_width=True, hide_index=True)
else:
    st.caption("Sin hiperparámetros registrados para este modelo.")
