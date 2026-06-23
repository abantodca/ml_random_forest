"""Inyección del CSS global de la app."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "css" / "style.css"

_FALLBACK_CSS = (
    "html,body,[class*='css']{font-family:Inter,system-ui,-apple-system,"
    "BlinkMacSystemFont,'Segoe UI',sans-serif;background:#F8FAFC;}"
)

# Pre-CSS que pinta el fondo correcto en el PRIMER paint, antes de que
# la hoja completa se aplique. Sumado a `.streamlit/config.toml`, evita
# el flash entre el tema por defecto y el oficial.
_PREPAINT = (
    "html,body,.stApp,[data-testid='stAppViewContainer']"
    "{background:#F4F5FB !important;}"
    "[data-testid='stHeader']{background:transparent !important;}"
    "section[data-testid='stSidebar']{"
    "background:linear-gradient(180deg,#1E1B4B 0%,#312E81 100%) !important;}"
    ".stMainBlockContainer{padding-top:1.1rem !important;}"
)


def _load_css() -> str:
    """Lee el CSS sin cache (es un archivo pequeño y cambia poco)."""
    if _CSS_PATH.exists():
        return _CSS_PATH.read_text(encoding="utf-8")
    return _FALLBACK_CSS


def _css_version() -> str:
    """Hash del archivo CSS para forzar invalidación al cambiar."""
    if not _CSS_PATH.exists():
        return "0"
    try:
        return str(int(_CSS_PATH.stat().st_mtime))
    except OSError:
        return "0"


def inject_css() -> None:
    """Inyecta el CSS personalizado desde assets/css/style.css."""
    css = _load_css()
    version = _css_version()
    # El comentario con la versión obliga al browser/Streamlit a tratar
    # cada cambio del archivo como una hoja distinta — ningún cache viejo
    # persiste tras un edit.
    st.markdown(
        f"<style data-rnd-css='{version}'>/* rnd-forest css v{version} */{_PREPAINT}{css}</style>",
        unsafe_allow_html=True,
    )
