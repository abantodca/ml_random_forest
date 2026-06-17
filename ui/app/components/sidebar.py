"""Sidebar persistente: branding, health, navegación y acciones (nativo)."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from app.core import LONGITUD_VISIBLE_API_URL
from app.dependencies import get_cached_health, get_config, reload_models_and_clear_cache


def _section_label(text: str) -> None:
    """Etiqueta de sección del sidebar (estilo `.section-label`, paleta arándano)."""
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def _render_brand() -> None:
    st.title("🫐 RND Forest")
    st.caption("ML Forecasting")


def _render_health() -> None:
    health = get_cached_health()
    if health:
        meta = f"{health.models_loaded}/{health.total_varieties} modelos"
        if health.mlflow_connected:
            st.success(f"API conectada · {meta}")
        else:
            st.warning(f"API conectada · MLflow desconectado · {meta}")
        return

    api_url = get_config().api_url[:LONGITUD_VISIBLE_API_URL]
    st.error(f"API desconectada · {api_url}…")


def _render_nav(pages: Iterable[st.Page]) -> None:
    _section_label("Navegación")
    for page in pages:
        st.page_link(page, label=page.title, icon=page.icon)


@st.fragment
def _render_reload_action() -> None:
    if st.button("🔄  Recargar modelos", use_container_width=True):
        result = reload_models_and_clear_cache()
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"✓ {result['models_loaded']} modelos recargados")


def render_sidebar(pages: Iterable[st.Page]) -> None:
    """Renderiza el sidebar (orden fijo: brand → health → nav → acciones → footer)."""
    pages = list(pages)
    with st.sidebar:
        _render_brand()
        _render_health()
        st.divider()
        _render_nav(pages)
        st.divider()
        _section_label("Acciones")
        _render_reload_action()
        st.divider()
        st.caption("Sistema RND · v1.2.0")
