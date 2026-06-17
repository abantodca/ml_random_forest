"""Reporte gerencial del modelo (Winner_<VARIETY>.html).

Embebe el reporte HTML generado por el servicio de entrenamiento
(`ml_training`) para que el público pueda consultarlo sin abrir MLflow.
El HTML viene del backend (`GET /api/varieties/{variety}/dashboard`) y
se renderiza en un iframe mediante `streamlit.components.v1.html`.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
from app.components.layout import empty_state, page_header, section_title
from app.core import ApiConnectionError, ApiResponseError
from app.dependencies import (
    get_all_variety_names,
    get_cached_dashboard_html,
    get_cached_health,
)

page_header(
    "Reporte del Modelo",
    "Tarjeta gerencial: veredicto, KPIs y comparativos por variedad",
    "🏆",
)

# Solo variedades con modelo entrenado: el reporte (Winner_<VARIEDAD>.html) es
# un artifact que se genera al entrenar, así que las no entrenadas no aplican.
_all_names = get_all_variety_names()
if not _all_names:
    # Lista vacía ⇒ backend caído vs. registry sin modelos (p. ej. entrenando).
    # `get_cached_health()` solo es None si el backend no responde.
    if get_cached_health() is None:
        st.error("No se puede conectar al backend. Verifica que el servicio esté corriendo.")
    else:
        empty_state(
            "Aún no hay modelos entrenados",
            "El backend está conectado pero todavía no hay variedades con modelo "
            "en el registry. Entrená al menos una variedad (o esperá a que termine "
            "el entrenamiento y recargá modelos) para ver su reporte.",
            icon="🏆",
        )
    st.stop()

section_title("🌿 Selecciona una variedad")
_selected = st.selectbox("Variedad", _all_names, label_visibility="collapsed")

with st.spinner(f"Cargando reporte de {_selected}..."):
    try:
        _html = get_cached_dashboard_html(_selected)
    except ApiResponseError as exc:
        st.error(f"❌ Error del backend: {exc.detail}")
        st.stop()
    except ApiConnectionError as exc:
        st.error(f"❌ Error de conexión: {exc}")
        st.stop()

if not _html:
    empty_state(
        f"Sin reporte disponible para {_selected}",
        "El reporte (Winner_<VARIEDAD>.html) se genera al entrenar el modelo. "
        "Solicitá a Data Science entrenar esta variedad.",
        icon="🏆",
    )
    st.stop()

_tb1, _tb2, _tb3 = st.columns([2, 1, 1])
with _tb1:
    st.markdown(
        f"**Variedad:** `{_selected}`  ·  "
        f"**Tamaño:** `{len(_html) / 1024:.0f} KB`  ·  "
        f"**Origen:** MLflow artifact"
    )
with _tb2:
    st.download_button(
        "⬇️ Descargar HTML",
        _html.encode("utf-8"),
        file_name=f"Winner_{_selected}.html",
        mime="text/html",
        type="secondary",
        use_container_width=True,
    )
with _tb3:
    if st.button("🔄 Recargar", use_container_width=True):
        get_cached_dashboard_html.clear()
        st.rerun()

section_title("📊 Reporte")
components.html(_html, height=3200, scrolling=True)

st.caption(
    "Este reporte se genera automáticamente por el servicio de entrenamiento "
    "(`ml_training`) tras cada nuevo entrenamiento y se versiona como artifact de MLflow."
)
