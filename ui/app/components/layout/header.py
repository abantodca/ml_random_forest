"""Encabezado de página — un único componente `.page-hero` (CSS tokens)."""

from __future__ import annotations

import streamlit as st


def page_header(title: str, subtitle: str, icon: str = "") -> None:
    """Renderiza el hero card de la página actual."""
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="hero-icon">{icon}</div>
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
