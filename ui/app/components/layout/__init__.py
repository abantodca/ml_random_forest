"""app.components.layout - Bloques de layout reutilizables."""

from app.components.layout.drift_panel import (
    render_batch_drift_panel,
    render_drift_panel,
)
from app.components.layout.empty_state import empty_state, empty_state_annotation
from app.components.layout.header import page_header
from app.components.layout.insight_card import insight_card
from app.components.layout.kpi_card import kpi_card
from app.components.layout.section_title import section_title

__all__ = [
    "page_header",
    "empty_state",
    "empty_state_annotation",
    "insight_card",
    "kpi_card",
    "render_batch_drift_panel",
    "render_drift_panel",
    "section_title",
]
