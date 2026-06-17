"""Etiqueta de sección con accent line lateral (CSS tokens)."""

from __future__ import annotations

import streamlit as st


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)
