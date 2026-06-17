"""Tarjeta KPI — variantes semánticas (success/warning/danger/primary/accent/info).

API:
    kpi_card("Backend", "Online", icon="🟢", variant="success")
    kpi_card("Modelos", "12/16", icon="🤖", variant="primary")
"""

from __future__ import annotations

import streamlit as st


def kpi_card(
    title: str,
    value: str,
    icon: str = "",
    variant: str = "primary",
) -> None:
    st.markdown(
        f"""
        <div class="kpi-card {variant}" data-accent>
            <div class="kpi-head">
                <span class="kpi-icon">{icon}</span>
                <span>{title}</span>
            </div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
