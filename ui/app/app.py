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

# `streamlit run app/app.py` inserta el directorio del entrypoint (`…/app`) al
# frente de `sys.path` (streamlit/web/bootstrap.py). Como ESTE archivo se llama
# `app.py`, ahí ensombrece al paquete `app/` y `from app.components import …`
# revienta con "ModuleNotFoundError: 'app' is not a package". Anteponemos la
# raíz del proyecto (el padre de `app/`) para que `import app` resuelva el
# PAQUETE, no este módulo. Imprescindible bajo `streamlit run` (no bajo AppTest,
# que ya corre con la raíz en el path — por eso no lo detectaba).
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


# url_path explícito por página — evita que Streamlit "adivine" la URL a
# partir del título y entre en conflicto con el router legacy de pages/.
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


# IMPORTANTE: st.navigation() debe llamarse ANTES que cualquier st.page_link()
# (que vive dentro del sidebar). De lo contrario el router legacy de Streamlit
# intenta resolver la URL contra pages/ y emite "Page not found" antes de que
# st.navigation tome el control.
pg = st.navigation(_PAGES, position="hidden")
render_sidebar(_PAGES)
pg.run()
