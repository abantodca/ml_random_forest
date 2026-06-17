"""Indicator simple para mostrar un valor único (predicción individual)."""

from __future__ import annotations

import plotly.graph_objects as go

from app.core import TEMA


def build_kghora_indicator(value: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="number",
            value=value,
            title={
                "text": "KGHORA Predicho",
                "font": {"size": 13, "color": TEMA["text_secondary"]},
            },
            number={
                "suffix": " kg/h",
                "valueformat": ".2f",
                "font": {"size": 40, "color": TEMA["primary"]},
            },
        )
    )
    fig.update_layout(
        height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif", color=TEMA["text_body"]),
    )
    return fig
