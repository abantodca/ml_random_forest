"""Tarjeta de insight (mejor variedad, R² promedio, etc.)."""

from __future__ import annotations

import streamlit as st


def insight_card(
    label: str,
    value: str,
    meta: str = "",
    variant: str = "primary",
) -> None:
    """Renderiza una insight card. `variant` ∈ {success,warning,danger,primary,accent,info}."""
    st.markdown(
        f"""
        <div class="insight-card {variant}" data-accent>
            <div class="insight-label">{label}</div>
            <div class="insight-value">{value}</div>
            <div class="insight-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
