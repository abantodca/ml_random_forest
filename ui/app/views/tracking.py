"""Seguimiento / Precisión: proyectado vs real, error y descomposición.

Responde las preguntas comunes del día a día con pronósticos:
¿le atina? ¿está sesgado? ¿es culpa de los datos o del modelo? ¿cómo cierra
la semana? Los datos reales se cargan con un Excel casi idéntico al de
pronósticos (+ columna KG/JR_H realizada).

Vista delgada: la agregación (KPIs, diagnóstico, cierre semanal, series de
gráficos) y los helpers de datos reales viven en `app.presenters.tracking`;
acá solo se compone el UI (filtros, editor, cards, gráficos).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from app.components.charts import (
    build_decomp_scatter,
    build_parity_plot,
    build_pred_vs_real_line,
    build_residual_bars,
    build_weekly_bars,
)
from app.components.layout import empty_state, insight_card, kpi_card, page_header, section_title
from app.core import ApiConnectionError, ApiResponseError
from app.dependencies import (
    get_all_variety_names,
    get_cached_accuracy,
    get_cached_catalogs,
    get_cached_health,
    get_tracking_service,
)
from app.presenters import tracking as P
from app.schemas import AccuracyPoint

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _api_call(spinner_msg: str, fn, *, error_prefix: str = "Error"):
    """Ejecuta una mutación contra el backend con spinner y manejo uniforme
    de errores. Devuelve el resultado, o None si la llamada falló (el error
    ya quedó pintado en pantalla)."""
    with st.spinner(spinner_msg):
        try:
            return fn()
        except (ApiResponseError, ApiConnectionError) as exc:
            st.error(f"{error_prefix}: {exc}")
            return None


def _commit_ok(msg: str) -> None:
    """Cierre común de toda mutación exitosa: feedback + invalidar el cache
    de precisión (los pares proy↔real cambiaron) + rerun."""
    st.success(msg)
    get_cached_accuracy.clear()
    st.rerun()


def _render_filters(points: list[AccuracyPoint]) -> list[AccuracyPoint]:
    """Filtro por FUNDO + SEMANA (la variedad ya viene del selector superior)."""
    fundos, weeks = P.filter_options(points)
    section_title("🔍 FILTROS")
    f1, f2 = st.columns(2)
    sel_f = f1.multiselect("Fundo", fundos, default=fundos, key="seg_f_fundo")
    sel_w = f2.multiselect("Semana ISO", weeks, default=weeks, key="seg_f_week")
    return P.apply_filters(points, sel_f, sel_w)


def _real_column_config() -> dict:
    cat = get_cached_catalogs()
    cc = st.column_config
    return {
        "FUNDO": cc.SelectboxColumn("Fundo", options=list(cat.fundos), required=True),
        "FORMATO": cc.SelectboxColumn("Formato", options=list(cat.formatos), required=True, width="medium"),
        "FECHA": cc.DateColumn("Fecha", format="YYYY-MM-DD", required=True),
        "KG/HA": cc.NumberColumn("KG/HA", min_value=0.0, step=100.0, format="%.0f", required=True),
        "KG/JR_H": cc.NumberColumn("KG/JR_H real", min_value=0.0, step=0.1, format="%.2f", required=True),
        "DPC": cc.NumberColumn("DPC", min_value=0.0, max_value=400.0),
        "%INDUS": cc.NumberColumn("%INDUS", min_value=0.0, max_value=100.0),
        "P/BAYA": cc.NumberColumn("P/BAYA", min_value=0.0),
        "HA": cc.NumberColumn("HA", min_value=0.0),
        "DIA_COSECHA": cc.NumberColumn("Día cosecha", min_value=0, max_value=365, step=1, format="%d"),
    }


# ── Datos reales: SOLO masivo (Excel) + editar/eliminar/reemplazar ──────
@st.fragment
def _render_real_upload(variety: str) -> None:
    st.info(
        "📥 **Acá cargás la cosecha REAL que YA ocurrió**, con la columna "
        "**`KG/JR_H`** (lo realmente logrado) — sirve para medir qué tan bien "
        "predijo el modelo. Es **solo masivo** (Excel). ¿Querés **predecir** algo "
        "nuevo? Eso va en **Pronosticar**, no acá.",
        icon="📥",
    )
    st.caption(
        "Mismas columnas del Excel de pronóstico **+ `KG/JR_H`** (obligatoria). "
        "Opcionales (DPC, HA, DIA_COSECHA, %INDUS, P/BAYA) habilitan la "
        "descomposición exacta datos-vs-modelo."
    )
    _t_sub, _t_edit = st.tabs(["📤 Subir Excel", "✏️ Editar / eliminar"])
    with _t_sub:
        _real_subir(variety)
    with _t_edit:
        _real_editar(variety)


def _real_subir(variety: str) -> None:
    st.download_button(
        "⬇️ Plantilla Excel", P.build_real_template_xlsx(),
        file_name="plantilla_datos_reales.xlsx", mime=_XLSX_MIME, key="real_tpl",
    )
    uploaded = st.file_uploader(
        "Excel de DATOS REALES (cosecha ocurrida + KG/JR_H)",
        type=["xlsx", "xls"], key="real_upload",
    )
    replace = st.checkbox(
        "Reemplazar TODO el histórico real de la variedad", value=True, key="real_replace",
        help="Activado: borra lo anterior y deja solo este archivo. Desactivado: agrega.",
    )
    if uploaded is None:
        return
    if not st.button("📤 Subir", type="primary", key="btn_real_up"):
        return
    res = _api_call(
        "Subiendo...",
        lambda: get_tracking_service().upload_real_excel(
            variety, uploaded.getvalue(), uploaded.name, replace=replace,
        ),
        error_prefix="Error al subir",
    )
    if res is None:
        return
    _commit_ok(
        f"✅ {res.get('inserted', 0)} reales cargados "
        f"({res.get('skipped_invalid_rows', 0)} descartados)."
    )


def _real_editar(variety: str) -> None:
    df = P.real_grid_from_history(variety)
    if df.empty:
        empty_state(
            "Sin datos reales cargados",
            help="Subí un Excel en la pestaña <strong>📤 Subir Excel</strong>.",
            icon="📋",
        )
        return
    st.caption(
        f"**{len(df)}** observaciones reales. **Borrar fila(s):** marcá la casilla ☑ "
        "a la izquierda y apretá `Supr` (Delete). **Agregar:** escribí en la fila vacía "
        "del final. Los cambios recién se aplican al apretar **💾 Guardar cambios**, "
        "que **reemplaza** todo el histórico real de la variedad (borrar fila + Guardar = "
        "eliminar esa observación; «🗑 Eliminar todos» borra el set completo)."
    )
    edited = st.data_editor(
        df, column_config=_real_column_config(), num_rows="dynamic",
        use_container_width=True, hide_index=True, key="real_editor",
    )
    b1, b2, _ = st.columns([1.6, 1.2, 3])
    if b1.button("💾 Guardar cambios (reemplaza)", type="primary", key="real_save"):
        clean = P.coerce_real(edited)
        if clean.empty:
            st.warning("La grilla quedó vacía — usá 'Eliminar todos' para borrar.")
            return
        res = _api_call(
            "Guardando...",
            lambda: get_tracking_service().replace_real_from_rows(variety, clean),
        )
        if res is None:
            return
        _commit_ok(
            f"✅ Reemplazado: {res.get('inserted', 0)} reales "
            f"({res.get('skipped_invalid_rows', 0)} descartados)."
        )
    if b2.button("🗑 Eliminar todos", key="real_del"):
        n = _api_call(
            "Eliminando...",
            lambda: get_tracking_service().delete_history(variety),
        )
        if n is None:
            return
        _commit_ok(f"✅ {n} observaciones reales eliminadas.")


# ── Secciones de análisis (renderers delgados sobre los view-models) ─────
def _render_kpis(points: list[AccuracyPoint]) -> None:
    vm = P.build_kpi_vm(points)

    # ── Veredicto gerencial — primera respuesta: ¿confío? ¿reentrenar? ──────
    {"ok": st.success, "warning": st.warning, "alert": st.error}[vm.verdict_status](vm.verdict_msg)

    section_title("📊 PRECISIÓN DEL PRONÓSTICO")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Puntos comparados", str(vm.n_points), icon="🔗", variant="primary")
    with k2:
        kpi_card("MAE real", f"{vm.mae:.2f}", icon="📏", variant="accent")
    with k3:
        kpi_card("MAPE real", f"{vm.mape:.1f}%" if vm.has_mape else "—", icon="📐",
                 variant=vm.mape_variant)
    with k4:
        kpi_card("Sesgo", f"{vm.bias:+.2f}", icon="🎯", variant=vm.bias_variant)
    st.caption(
        f"El pronóstico **{vm.sesgo_dir}** en promedio ({vm.bias:+.2f}). "
        "MAE/MAPE miden el error típico; sesgo indica si hay una tendencia sistemática."
    )


def _render_decomp_insight(points: list[AccuracyPoint]) -> None:
    vm = P.build_decomp_vm(points)
    if not vm.available:
        empty_state(
            "Descomposición datos-vs-modelo no disponible",
            help="Subí datos reales con las features completas "
                 "(DPC, HA, DIA_COSECHA…) para saber si el error viene "
                 "del <strong>input proyectado</strong> o del "
                 "<strong>modelo</strong>.",
            icon="🧭",
        )
        return

    section_title("🧭 ¿ERROR DE DATOS O DE MODELO?")
    c1, c2 = st.columns(2)
    with c1:
        insight_card(
            "Error por datos", f"{vm.mean_data:.2f}",
            "Promedio |error| atribuible a la <strong>proyección de KG/HA</strong>",
            vm.data_variant,
        )
    with c2:
        insight_card(
            "Error del modelo", f"{vm.mean_model:.2f}",
            "Promedio |error| del modelo dado el <strong>input real</strong>",
            vm.model_variant,
        )
    if vm.predominant == "data":
        st.caption(
            "→ Predomina el **error de datos proyectados**: mejorar la proyección "
            "de los inputs (sobre todo KG/HA) reduciría más el error."
        )
    else:
        st.caption(
            "→ Predomina el **error del modelo**: aún con inputs reales el modelo "
            "se desvía; considerar reentrenar o revisar features."
        )


def _render_charts(points: list[AccuracyPoint]) -> None:
    vm = P.build_charts_vm(points)
    section_title("📈 PROYECTADO VS REAL")
    st.plotly_chart(build_pred_vs_real_line(vm.fechas, vm.pred, vm.real),
                    use_container_width=True)
    st.plotly_chart(build_residual_bars(vm.fechas, vm.err), use_container_width=True)

    section_title("🎯 PRECISIÓN (PARIDAD)")
    st.plotly_chart(build_parity_plot(vm.real, vm.pred), use_container_width=True)

    if vm.has_decomp:
        section_title("🧭 DIAGNÓSTICO: DATOS VS MODELO")
        st.plotly_chart(
            build_decomp_scatter(vm.decomp_data, vm.decomp_model, vm.decomp_labels),
            use_container_width=True,
        )


def _render_weekly(points: list[AccuracyPoint]) -> None:
    vm = P.build_weekly_vm(points)
    if not vm.has_weeks:
        return
    section_title("🗓️ CIERRE SEMANAL")

    # ── KPIs de cierre: cumplimiento global + mejor/peor semana ─────────────
    if vm.cumplimiento is not None:
        w1, w2, w3 = st.columns(3)
        with w1:
            kpi_card("Cumplimiento global", f"{vm.cumplimiento:.1f}%", icon="📊",
                     variant=vm.cumpl_variant)
        with w2:
            insight_card("Mejor semana", vm.mejor_week,
                         f"Δ {vm.mejor_pct:+.1f}% · {vm.mejor_n} punto(s)", "success")
        with w3:
            insight_card("Peor semana", vm.peor_week,
                         f"Δ {vm.peor_pct:+.1f}% · {vm.peor_n} punto(s)", "warning")

    st.plotly_chart(
        build_weekly_bars(vm.weeks, vm.proj_sums, vm.real_sums),
        use_container_width=True,
    )
    st.dataframe(pd.DataFrame(vm.table_rows), use_container_width=True, hide_index=True)


def _render_table(points: list[AccuracyPoint]) -> None:
    section_title("📋 DETALLE")
    st.dataframe(pd.DataFrame(P.build_table_rows(points)),
                 use_container_width=True, hide_index=True)


# ── Render principal ────────────────────────────────────────────────────
page_header(
    "Seguimiento / Precisión",
    "Proyectado vs real, error del pronóstico y diagnóstico datos-vs-modelo",
    "🎯",
)

# Solo variedades con modelo entrenado: el seguimiento compara el pronóstico
# del modelo contra la cosecha real, así que sin modelo no hay nada que medir.
_all_names = get_all_variety_names()
if not _all_names:
    # Lista vacía ⇒ backend caído vs. registry sin modelos (p. ej. entrenando).
    # `get_cached_health()` solo es None si el backend no responde.
    if get_cached_health() is None:
        st.error("No se puede conectar al backend. Verifica que el servicio esté corriendo.")
    else:
        empty_state(
            "Aún no hay modelos entrenados",
            "El backend está conectado pero todavía no hay variedades con modelo "
            "en el registry. Entrená al menos una variedad (o esperá a que termine "
            "el entrenamiento y recargá modelos) para hacer seguimiento de su precisión.",
            icon="🎯",
        )
    st.stop()

section_title("🌿 VARIEDAD")
_variety = st.selectbox("Variedad", _all_names, label_visibility="collapsed")

with st.spinner("Comparando proyectado vs real..."):
    _points = get_cached_accuracy(_variety, with_decomposition=True)

# Visión gerencial: la página abre con el INSIGHT (KPIs → diagnóstico →
# gráficos), no con la tarea operativa de subir archivos. La carga de datos
# reales vive en un expander al pie (expandido solo si aún no hay datos).
if _points:
    _filtered = _render_filters(_points)
    if not _filtered:
        st.warning("Sin datos para los filtros seleccionados — ajustá fundo/semana.")
    else:
        _render_kpis(_filtered)
        _render_decomp_insight(_filtered)
        _render_charts(_filtered)
        _render_weekly(_filtered)
        _render_table(_filtered)
else:
    empty_state(
        "Aún no hay pares proyectado ↔ real para esta variedad",
        help="Cargá datos reales (abajo) cuyo <strong>(fundo, formato, fecha)</strong> "
             "coincida con pronósticos ya almacenados para empezar a medir la precisión.",
        icon="📭",
    )

with st.expander("📥 Cargar / actualizar datos reales", expanded=not _points):
    _render_real_upload(_variety)
