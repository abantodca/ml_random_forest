"""app.components.forms - Formularios reutilizables (filtros de búsqueda).

El formulario de entrada individual se retiró: la captura (individual y por
lote) vive ahora en la grilla editable de la página "Pronosticar".
"""

from app.components.forms.search_filters import SearchCriteria, render_search_filters

__all__ = [
    "SearchCriteria",
    "render_search_filters",
]
