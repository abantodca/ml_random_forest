"""
ui - Fachada raíz de los componentes Streamlit (presentación)
==============================================================
Solo re-exporta los archivos directos del paquete `ui/`. Cada
subcarpeta (`layout`, `forms`, `charts`,
`forecast_tabs`) tiene su propio barrel para que las
modificaciones internas de un grupo no afecten al resto.
"""

from app.components.css import inject_css
from app.components.sidebar import render_sidebar
from app.components.theme import THEME

__all__ = [
    "render_sidebar",
    "inject_css",
    "THEME",
]
