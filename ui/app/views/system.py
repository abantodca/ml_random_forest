"""Estado del sistema (configuración + health del backend)."""

from __future__ import annotations

import streamlit as st
from app.components.layout import kpi_card, page_header, section_title
from app.dependencies import get_cached_health, get_config

page_header("Sistema", "Diagnóstico y estado del backend", "⚙️")

_cfg = get_config()
_health = get_cached_health()

section_title("💡 Estado del backend")
if not _health:
    st.error("No se puede conectar al backend. Verifica que el servicio esté corriendo.")
    st.stop()

_c1, _c2, _c3, _c4 = st.columns(4)
with _c1:
    kpi_card(
        "Status",
        _health.status.upper(),
        icon="🖥️",
        variant="success" if _health.is_healthy else "danger",
    )
with _c2:
    kpi_card(
        "MLflow",
        "Conectado" if _health.mlflow_connected else "Desconectado",
        icon="🧪",
        variant="success" if _health.mlflow_connected else "danger",
    )
with _c3:
    kpi_card(
        "Base de datos",
        "Conectada" if _health.database_connected else "Desconectada",
        icon="🗄️",
        variant="success" if _health.database_connected else "danger",
    )
with _c4:
    kpi_card(
        "Modelos",
        f"{_health.models_loaded} / {_health.models_available}",
        icon="🤖",
        variant="primary",
    )

section_title("🔧 Configuración")
st.json({"API_URL": _cfg.api_url, "Cache TTL health": _cfg.cache_ttl_health})
