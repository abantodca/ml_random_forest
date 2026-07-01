"""Pronosticar — workspace unificado (grilla editable estilo Excel) + historial.

Reemplaza las páginas "Predicción" y "Pronósticos" (Crear/Subir/Listar):
**1 fila = pronóstico individual · N filas = lote**. Tecleás o pegás desde
Excel, validás inline (dropdowns de FUNDO/FORMATO, rangos por celda) y
predecís+guardás el lote en un paso, con confiabilidad/drift inline. El
historial (tabla + editar/borrar) queda como segunda pestaña.

Vista delgada: la grilla/coerción, la ejecución del lote y el armado de
resultados viven en `app.presenters.forecast`; acá solo se compone el UI.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from app.components.charts import build_kghora_histogram
from app.components.forecast_tabs import render_list_tab
from app.components.layout import (
    empty_state,
    kpi_card,
    page_header,
    render_batch_drift_panel,
    render_drift_panel,
    section_title,
)
from app.dependencies import (
    get_all_variety_names,
    get_cached_catalogs,
    get_cached_health,
    get_loaded_variety_names,
)
from app.presenters import forecast as F
from app.services import BatchValidationError, validate_batch_upload

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_SS_GRID = "prono_grid_df"
_EDITOR_KEY = "prono_grid_editor"


def _column_config(varieties: list[str], fundos: list[str], formatos: list[str]) -> dict:
    cc = st.column_config
    return {
        "VARIEDAD": cc.SelectboxColumn(
            "Variedad", options=varieties, required=True, width="medium"
        ),
        "FECHA": cc.DateColumn("Fecha", format="YYYY-MM-DD", required=True),
        "FUNDO": cc.SelectboxColumn("Fundo", options=fundos, required=True),
        "FORMATO": cc.SelectboxColumn("Formato", options=formatos, required=True, width="medium"),
        "KG/HA": cc.NumberColumn(
            "KG/HA", min_value=0.0, max_value=100_000.0, step=100.0, format="%.0f", required=True
        ),
        "DPC": cc.NumberColumn(
            "DPC", min_value=0.0, max_value=400.0, step=1.0, format="%.0f", required=True
        ),
        "HA": cc.NumberColumn(
            "HA", min_value=0.0, max_value=10_000.0, step=0.5, format="%.1f", required=True
        ),
        "DIA_COSECHA": cc.NumberColumn(
            "Día cosecha", min_value=0, max_value=365, step=1, format="%d", required=True
        ),
        "%INDUS": cc.NumberColumn(
            "%INDUS", min_value=0.0, max_value=100.0, step=0.5, help="Opcional"
        ),
        "P/BAYA": cc.NumberColumn(
            "P/BAYA (g)", min_value=0.0, max_value=100.0, step=0.1, help="Opcional"
        ),
        "HORAS_EFECTIVAS": cc.NumberColumn(
            "Horas efect.", min_value=0.0, max_value=24.0, step=0.5, help="Opcional → KGJN"
        ),
        "EXTERNAL_ID": cc.TextColumn("ID externo", help="Opcional"),
    }


def _render_results(result: F.BatchRunResult) -> None:
    vm = F.build_results_vm(result.preds, result.records)
    st.success(f"✅ {vm.n_preds} pronósticos generados y guardados")

    # KPIs ejecutivos
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Pronósticos", str(vm.n_preds), icon="🔮", variant="primary")
    with k2:
        kpi_card("KGHORA prom.", f"{vm.avg_kghora:.2f}", icon="📊", variant="accent")
    with k3:
        kpi_card("Confiables 🟢", str(vm.n_ok), icon="✅", variant="success")
    with k4:
        kpi_card("A revisar 🟡🔴", str(vm.flagged), icon="⚠️", variant=vm.flagged_variant)

    # Alerta visible si hay registros fuera de distribución
    if vm.alert_msg:
        st.error(vm.alert_msg)
    elif vm.warning_msg:
        st.warning(vm.warning_msg)

    section_title("📊 RESULTADOS")
    st.dataframe(
        vm.results_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "KGHORA pred": st.column_config.NumberColumn("KGHORA pred", format="%.3f"),
            "KGJN pred": st.column_config.NumberColumn("KGJN pred", format="%.3f"),
        },
    )

    if not vm.hist_df.empty:
        st.plotly_chart(build_kghora_histogram(vm.hist_df), use_container_width=True)

    if result.batch_drifts:
        section_title("🔬 DRIFT ESTADÍSTICO DEL LOTE")
        if len(result.batch_drifts) == 1:
            name, bd = result.batch_drifts[0]
            st.markdown(f"**Variedad:** `{name}`")
            render_batch_drift_panel(bd, expanded=True)
        else:
            opts = {
                f"{n} ({d.n_rows} filas · {F.DRIFT_BADGE.get(d.status, '—')})": d
                for n, d in result.batch_drifts
            }
            lbl = st.selectbox("Variedad", list(opts.keys()), key="grid_bd_sel")
            render_batch_drift_panel(opts[lbl], expanded=True)

    if result.records:
        section_title("🔬 DETALLE DE DRIFT POR REGISTRO")
        opts = {
            f"#{r.id} — {r.variety} — {r.fecha} "
            f"({F.DRIFT_BADGE.get(r.drift.status, '—') if r.drift else '—'})": r
            for r in result.records
        }
        lbl = st.selectbox("Registro", list(opts.keys()), key="grid_row_sel")
        render_drift_panel(opts[lbl].drift, expanded=True)


def _render_defaults_section(
    all_names: list[str], fundos: list[str], formatos: list[str], fmt_default: str
) -> tuple[str, str, str]:
    """Fila de defaults (variedad/fundo/formato) + botón de plantilla.

    Devuelve los tres valores elegidos para sembrar la grilla nueva.
    """
    default_variety = (get_loaded_variety_names() or all_names)[0]

    section_title("🌿 DEFAULTS (filas nuevas y plantilla)")
    d1, d2, d3, d4 = st.columns([2, 1, 2, 2], vertical_alignment="bottom")
    variety = d1.selectbox(
        "Variedad",
        all_names,
        index=all_names.index(default_variety) if default_variety in all_names else 0,
        key="grid_def_var",
    )
    fundo = d2.selectbox("Fundo", fundos, key="grid_def_fundo") if fundos else ""
    formato = (
        d3.selectbox(
            "Formato",
            formatos,
            index=formatos.index(fmt_default) if fmt_default in formatos else 0,
            key="grid_def_fmt",
        )
        if formatos
        else ""
    )
    with d4:
        st.download_button(
            "⬇️ Plantilla Excel",
            F.template_xlsx(variety, fundo, formato),
            file_name="plantilla_pronosticos.xlsx",
            mime=_XLSX_MIME,
            use_container_width=True,
            key="grid_tpl",
        )
    return variety, fundo, formato


def _render_upload_section() -> None:
    """Expander para cargar un Excel/CSV de inputs a la grilla."""
    with st.expander(
        "📤 Cargar Excel de PRONÓSTICOS (inputs a predecir · SIN KG/JR_H)",
        expanded=False,
    ):
        up = st.file_uploader(
            "Excel de pronósticos (a predecir)",
            type=["xlsx", "xls", "csv"],
            key="grid_upload",
            label_visibility="collapsed",
        )
        if up is not None and st.button("Cargar a la grilla", key="grid_load"):
            try:
                raw = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
                st.session_state[_SS_GRID] = F.normalize_upload(raw)
                st.session_state.pop(_EDITOR_KEY, None)  # forzar reseed del editor
                st.rerun(scope="fragment")
            except Exception as exc:
                st.error(f"No se pudo leer el archivo: {exc}")


def _handle_predict(edited: pd.DataFrame, all_names: list[str]) -> None:
    """Coerción → validación → ejecución del lote → resultados."""
    clean = F.coerce(edited)
    if clean.empty:
        st.warning("La grilla está vacía — agregá al menos una fila con VARIEDAD y KG/HA.")
        return
    try:
        valid = validate_batch_upload(clean, valid_varieties=all_names)
    except BatchValidationError as exc:
        section_title("⛔ ERRORES DE VALIDACIÓN")
        st.error(
            f"**{len(exc.issues)} problema(s)** encontrados. "
            "Corregí las celdas marcadas abajo y volvé a predecir."
        )
        cols_con_error = F.affected_columns(exc.issues)
        st.caption(f"Columnas afectadas: {', '.join(f'**{c}**' for c in cols_con_error)}")
        st.dataframe(
            pd.DataFrame(F.issue_rows(exc.issues)), use_container_width=True, hide_index=True
        )
        return
    except Exception as exc:
        st.error(f"Error inesperado de validación: {exc}")
        return

    progress = st.progress(0.0, text="Procesando...")
    with st.spinner("Prediciendo y guardando..."):
        result = F.execute_batch(
            valid,
            progress_cb=lambda frac, txt: progress.progress(frac, text=txt),
        )
    progress.empty()
    for err in result.errors:
        st.error(err)
    if result.preds:
        _render_results(result)


@st.fragment
def _render_new(all_names: list[str]) -> None:
    catalogs = get_cached_catalogs()
    fundos = list(catalogs.fundos)
    formatos = list(catalogs.formatos)
    fmt_default = catalogs.formato_default or (formatos[0] if formatos else "")

    variety, fundo, formato = _render_defaults_section(all_names, fundos, formatos, fmt_default)

    if _SS_GRID not in st.session_state:
        st.session_state[_SS_GRID] = F.empty_grid(variety, fundo, formato)

    _render_upload_section()

    st.caption(
        "**Cómo usar la grilla:** pegá desde Excel (Ctrl+V) en cualquier celda · "
        "**agregar fila** → escribí en la fila vacía del final · "
        "**borrar fila(s)** → marcá la casilla ☑ a la izquierda y apretá `Supr` (Delete). "
        "Las columnas opcionales (%INDUS, P/BAYA, HORAS_EFECTIVAS) se pueden dejar vacías. "
        "Nada se guarda hasta apretar **🔮 Predecir y guardar lote**; para editar o borrar "
        "pronósticos ya guardados, usá la pestaña **📋 Historial**."
    )
    section_title("✏️ GRILLA — tecleá o pegá desde Excel · 1 fila = individual, N = lote")
    edited = st.data_editor(
        st.session_state[_SS_GRID],
        column_config=_column_config(all_names, fundos, formatos),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=_EDITOR_KEY,
    )

    # Resumen pre-predicción: n filas y variedades involucradas
    _summary = F.pre_summary_text(edited)
    if _summary:
        st.info(_summary)

    if not st.button("🔮 Predecir y guardar lote", type="primary", key="grid_predict"):
        return

    _handle_predict(edited, all_names)


# ── Render principal ────────────────────────────────────────────────────
page_header(
    "Pronosticar",
    "Grilla editable: individual o masivo, con confiabilidad y drift",
    "🔮",
)

_all = get_all_variety_names()
if not _all:
    # Lista vacía ⇒ dos causas distintas: backend caído vs. registry sin modelos
    # (p. ej. entrenamiento en curso). `get_cached_health()` solo es None si el
    # backend no responde, así que lo usamos para no mostrar un error de conexión
    # falso cuando en realidad el servicio está OK pero aún no hay modelos.
    if get_cached_health() is None:
        st.error("No se puede conectar al backend. Verifica que el servicio esté corriendo.")
    else:
        empty_state(
            "Aún no hay modelos para pronosticar",
            "El backend está conectado pero todavía no hay variedades con modelo "
            "en el registry. Si el entrenamiento está en curso, esperá a que termine "
            "y recargá modelos desde <strong>Sistema</strong> o <strong>Modelos</strong>.",
            icon="🔮",
        )
    st.stop()

st.info(
    "🔮 **Acá PREDECÍS** cosechas que **aún no ocurrieron**. El Excel trae los "
    "inputs (KG/HA, DPC, HA…) y el modelo calcula el KGHORA — **NO** incluye el "
    "resultado real. ¿Ya tenés la cosecha real (columna `KG/JR_H`)? Esa va en "
    "**Seguimiento → Cargar datos reales**, no acá.",
    icon="🔮",
)

_tab_new, _tab_hist = st.tabs(["✏️ Nuevo pronóstico", "📋 Historial"])
with _tab_new:
    _render_new(_all)
with _tab_hist:
    render_list_tab(_all)
