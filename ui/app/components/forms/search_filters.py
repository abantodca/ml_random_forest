"""Form de filtros de búsqueda de pronósticos."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import streamlit as st


@dataclass(frozen=True)
class SearchCriteria:
    variety: str
    fecha: date | None
    limit: int
    submitted: bool

    @property
    def variety_filter(self) -> str | None:
        return None if self.variety == "Todas" else self.variety


def render_search_filters(all_variety_names: list[str]) -> SearchCriteria:
    with st.form("search_form", border=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            variety = st.selectbox(
                "Filtrar por variedad",
                ["Todas"] + all_variety_names,
                key="filter_variety",
            )
        with col2:
            fecha = st.date_input(
                "Filtrar por fecha (opcional)",
                value=None,
                key="filter_fecha",
            )
        with col3:
            limit = st.number_input(
                "Límite",
                min_value=10,
                max_value=5000,
                value=100,
                step=10,
                key="filter_limit",
            )
        submitted = st.form_submit_button("🔍 Buscar", type="primary")
    return SearchCriteria(variety=variety, fecha=fecha, limit=int(limit), submitted=submitted)
