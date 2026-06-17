"""Tab Listar pronósticos: filtros, tabla nativa y CRUD por selección."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app.components.forms import SearchCriteria, render_search_filters
from app.components.layout import empty_state, section_title
from app.core import ApiConnectionError, ApiResponseError
from app.dependencies import get_cached_catalogs, get_forecast_service
from app.schemas import ForecastRecord
from app.services import build_forecast_payload

_SS_ITEMS = "fc_items"
_SS_TOTAL = "fc_total"
_SS_LAST_CRITERIA = "fc_last_criteria"
_SS_ACTION = "fc_action"  # "detail" | "edit" | "delete" | None

_FORMATO_EMOJI = {"FRESCO": "🥬", "PROCESO": "⚙️"}


def _items_to_df(items: tuple[ForecastRecord, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": it.id,
                "Variedad": it.variety,
                "Fecha": it.fecha,
                "Formato": f"{_FORMATO_EMOJI.get(it.formato, '')} {it.formato}".strip(),
                "Fundo": it.fundo or "—",
                "KG/HA": it.kg_ha,
                "DPC": it.dpc,
                "%INDUS": it.indus_pct,
                "KGHORA Pred": it.kghora_pred,
            }
            for it in items
        ]
    )


def _column_config() -> dict:
    return {
        "ID": st.column_config.NumberColumn(width="small", format="%d"),
        "Variedad": st.column_config.TextColumn(width="medium"),
        "Fecha": st.column_config.TextColumn(width="small"),
        "Formato": st.column_config.TextColumn(width="small"),
        "Fundo": st.column_config.TextColumn(width="small"),
        "KG/HA": st.column_config.NumberColumn(format="%.0f"),
        "DPC": st.column_config.NumberColumn(format="%.0f"),
        "%INDUS": st.column_config.NumberColumn(format="%.1f%%"),
        "KGHORA Pred": st.column_config.NumberColumn(format="%.2f", help="kg/h"),
    }


def _load_forecasts(criteria: SearchCriteria) -> None:
    result = get_forecast_service().list(
        variety=criteria.variety_filter,
        fecha=str(criteria.fecha) if criteria.fecha else None,
        limit=criteria.limit,
    )
    st.session_state[_SS_ITEMS] = result.items
    st.session_state[_SS_TOTAL] = result.total
    st.session_state[_SS_LAST_CRITERIA] = criteria
    st.session_state[_SS_ACTION] = None


def _reload_last() -> None:
    criteria = st.session_state.get(_SS_LAST_CRITERIA)
    if criteria:
        _load_forecasts(criteria)


def _set_action(action: str | None) -> None:
    st.session_state[_SS_ACTION] = action


def _render_detail_panel(item: ForecastRecord) -> None:
    with st.expander(f"📋 Detalle del pronóstico #{item.id}", expanded=True):
        cols = st.columns(4)
        rows = [
            ("ID", str(item.id)),
            ("Variedad", item.variety),
            ("Fecha", item.fecha or "—"),
            ("External ID", item.external_id or "—"),
            ("Fundo", item.fundo or "—"),
            ("Formato", item.formato),
            ("KG/HA", f"{item.kg_ha:,.2f}"),
            ("DPC", str(item.dpc)),
            ("HA", str(item.ha)),
            ("DIA_COSECHA", str(item.dia_cosecha)),
            ("%INDUS", f"{item.indus_pct:.2f}%" if item.indus_pct is not None else "—"),
            ("P/BAYA", str(item.p_baya) if item.p_baya is not None else "—"),
            ("Horas Efectivas", str(item.horas_efectivas) if item.horas_efectivas else "—"),
            ("KGHORA Pred", f"{item.kghora_pred:.4f}"),
            ("KGJN Pred", f"{item.kgjn_pred:.4f}" if item.kgjn_pred else "—"),
        ]
        for i, (label, value) in enumerate(rows):
            cols[i % 4].metric(label, value)
        st.caption(f"📅 Creado: {item.created_at}  ·  🔄 Actualizado: {item.updated_at}")


def _render_edit_form(item: ForecastRecord) -> None:
    catalogs = get_cached_catalogs()
    formatos = list(catalogs.formatos)
    fundos = list(catalogs.fundos)
    fmt_idx = formatos.index(item.formato) if item.formato in formatos else 0
    fundo_idx = fundos.index(item.fundo) if item.fundo in fundos else 0

    with st.expander(f"✏️ Editar pronóstico #{item.id}", expanded=True):
        with st.form(f"edit_form_{item.id}", border=False):
            ec1, ec2 = st.columns(2)
            with ec1:
                new_fecha = st.date_input(
                    "Fecha",
                    value=date.fromisoformat(item.fecha) if item.fecha else date.today(),
                )
                new_ext = st.text_input("External ID", value=item.external_id or "")
                new_fmt = st.selectbox("Formato", formatos, index=fmt_idx)
                new_fundo = st.selectbox("Fundo", fundos, index=fundo_idx)
                new_kgha = st.number_input("KG/HA", 0.1, 100_000.0, value=float(item.kg_ha))
                new_dpc = st.number_input("DPC", 0.0, 400.0, value=float(item.dpc))
            with ec2:
                new_ha = st.number_input("HA", 0.1, 10_000.0, value=float(item.ha))
                new_dia = st.number_input("DIA_COSECHA", 0, 365, value=int(item.dia_cosecha))
                # `value=None` cuando el item no tenía valor preserva la
                # semántica "campo vacío = no enviar al PATCH". Sólo se manda
                # si el usuario explícitamente escribe un número aquí.
                new_ind = st.number_input(
                    "%INDUS", 0.0, 100.0,
                    value=float(item.indus_pct) if item.indus_pct is not None else None,
                    placeholder="Opcional",
                )
                new_baya = st.number_input(
                    "P/BAYA (g)", 0.0, 100.0,
                    value=float(item.p_baya) if item.p_baya is not None else None,
                    placeholder="Opcional",
                )
                new_horas = st.number_input(
                    "Horas Efectivas", 0.0, 24.0,
                    value=float(item.horas_efectivas) if item.horas_efectivas is not None else None,
                    placeholder="Opcional",
                )
            saved = st.form_submit_button("💾 Guardar cambios", type="primary")

        if not saved:
            return

        payload = build_forecast_payload(
            fecha=new_fecha, kg_ha=new_kgha, dpc=new_dpc, ha=new_ha,
            dia_cosecha=int(new_dia), formato=new_fmt, fundo=new_fundo,
            indus_pct=new_ind, p_baya=new_baya, horas=new_horas,
            external_id=new_ext,
        )
        try:
            updated = get_forecast_service().update(item.id, payload)
        except (ApiResponseError, ApiConnectionError) as exc:
            st.error(f"Error: {exc}")
            return

        st.success(
            f"✅ Pronóstico #{updated.id} actualizado — "
            f"KGHORA: **{updated.kghora_pred:.2f}**"
        )
        _set_action(None)
        _reload_last()
        st.rerun(scope="fragment")


def _render_delete_confirmation(item: ForecastRecord) -> None:
    st.warning(
        f"¿Eliminar pronóstico **#{item.id}** ({item.variety} — {item.fecha})?"
    )
    cc1, cc2, _ = st.columns([1, 1, 5])
    if cc1.button("✅ Sí, eliminar", key=f"yes_del_{item.id}", type="primary"):
        try:
            get_forecast_service().delete(item.id)
        except (ApiResponseError, ApiConnectionError) as exc:
            st.error(f"Error: {exc}")
            return
        st.success(f"Pronóstico #{item.id} eliminado")
        _set_action(None)
        _reload_last()
        st.rerun(scope="fragment")
    if cc2.button("❌ Cancelar", key=f"no_del_{item.id}"):
        _set_action(None)
        st.rerun(scope="fragment")


def _render_action_panel(item: ForecastRecord) -> None:
    action = st.session_state.get(_SS_ACTION)
    if action == "detail":
        _render_detail_panel(item)
    elif action == "edit":
        _render_edit_form(item)
    elif action == "delete":
        _render_delete_confirmation(item)


@st.fragment
def render_list_tab(all_variety_names: list[str]) -> None:
    section_title("🔍 FILTROS DE BÚSQUEDA")
    criteria = render_search_filters(all_variety_names)

    if criteria.submitted:
        with st.spinner("Cargando pronósticos..."):
            try:
                _load_forecasts(criteria)
            except ApiConnectionError as exc:
                st.error(f"Error de conexión: {exc}")

    items: tuple[ForecastRecord, ...] = st.session_state.get(_SS_ITEMS, ())
    total: int = st.session_state.get(_SS_TOTAL, 0)

    if not items:
        if _SS_ITEMS in st.session_state:
            empty_state(
                "Sin pronósticos con esos filtros",
                "Ajustá la variedad o la fecha y volvé a buscar.",
                icon="🔎",
            )
        return

    st.info(f"Mostrando **{len(items)}** de **{total}** pronósticos")

    df = _items_to_df(items)
    event = st.dataframe(
        df,
        column_config=_column_config(),
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="fc_table",
    )

    selected_rows = event.selection.rows if event and getattr(event, "selection", None) else []
    if not selected_rows:
        st.caption("Selecciona una fila para ver acciones.")
        return

    item = items[selected_rows[0]]

    bc1, bc2, bc3, _ = st.columns([1, 1, 1, 4])
    if bc1.button("👁 Detalle", key="act_detail", use_container_width=True):
        _set_action("detail")
        st.rerun(scope="fragment")
    if bc2.button("🖊 Editar", key="act_edit", use_container_width=True):
        _set_action("edit")
        st.rerun(scope="fragment")
    if bc3.button("🗑 Eliminar", key="act_delete", use_container_width=True):
        _set_action("delete")
        st.rerun(scope="fragment")

    _render_action_panel(item)
