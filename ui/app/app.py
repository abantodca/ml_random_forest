"""Punto de entrada y hub de navegación.

Define la configuración global de página, inyecta el CSS, renderiza el
sidebar persistente (una sola vez, evitando el re-flash al navegar) y
registra las páginas con `st.navigation()`.

Uso:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path[:1] != [_PKG_ROOT]:
    sys.path.insert(0, _PKG_ROOT)

import streamlit as st  # noqa: E402  — debe ir tras el fix de sys.path de arriba

from app.components import inject_css, render_sidebar  # noqa: E402

st.set_page_config(
    page_title="RND Forest - Pronósticos",
    page_icon="🫐",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


_PAGES: list[st.Page] = [
    st.Page(
        "views/home.py",
        title="Dashboard",
        icon=":material/dashboard:",
        url_path="",
        default=True,
    ),
    st.Page(
        "views/forecast.py",
        title="Pronosticar",
        icon=":material/edit_note:",
        url_path="pronosticar",
    ),
    st.Page(
        "views/tracking.py",
        title="Seguimiento",
        icon=":material/monitoring:",
        url_path="seguimiento",
    ),
    st.Page(
        "views/models.py",
        title="Modelos",
        icon=":material/model_training:",
        url_path="modelos",
    ),
    st.Page(
        "views/model_report.py",
        title="Reporte Modelo",
        icon=":material/leaderboard:",
        url_path="dashboard",
    ),
    st.Page(
        "views/system.py",
        title="Sistema",
        icon=":material/settings:",
        url_path="sistema",
    ),
]


pg = st.navigation(_PAGES, position="hidden")
render_sidebar(_PAGES)
pg.run()
