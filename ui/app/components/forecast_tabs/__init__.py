"""app.components.forecast_tabs - Historial de pronósticos (tabla + editar/borrar).

La creación individual y por lote se unificó en la página "Pronosticar"
(grilla editable). Aquí solo queda el listado/CRUD, reutilizado como
pestaña "Historial".
"""

from app.components.forecast_tabs.tab_list import render_list_tab

__all__ = ["render_list_tab"]
