"""Estados vacíos: anotación para gráficos Plotly y card HTML ejecutiva."""

from __future__ import annotations

import streamlit as st

from app.core import TEMA


def empty_state_annotation(message: str) -> dict:
    return dict(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=TEMA["text_tertiary"]),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=TEMA["border"],
        borderpad=10,
        borderwidth=1,
    )


def empty_state(title: str, help: str = "", icon: str = "📭") -> None:
    """Card de "sin datos / sin selección" — alternativa ejecutiva a `st.info`.

    Args:
        title: encabezado breve (qué falta).
        help: ayuda accionable; admite HTML inline (p. ej. ``<strong>``).
        icon: emoji en la burbuja de marca.
    """
    help_html = f'<div class="empty-help">{help}</div>' if help else ""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">{icon}</div>
            <div class="empty-title">{title}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
