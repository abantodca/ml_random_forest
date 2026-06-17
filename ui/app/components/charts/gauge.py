"""Gauge (medidor) de cobertura de modelos y de drift."""

from __future__ import annotations

import plotly.graph_objects as go

from app.components.charts._common import AXIS_TEXT
from app.core import TEMA

# Umbrales del medidor de drift (espejan los del backend en
# `app/services/drift_service.py`). El gauge muestra `score` ∈ [0,1] donde
# 0=ok perfecto y 1=todo el batch fuera de distribución.
_DRIFT_OK_LIMIT: float = 0.33
_DRIFT_WARN_LIMIT: float = 0.66


def build_drift_gauge(score: float, status: str) -> go.Figure:
    """Medidor del score de drift (0 = sano, 1 = totalmente fuera).

    Semáforo INVERTIDO respecto al de cobertura: bajo es bueno. El color
    de la aguja cambia según `status` para reforzar el mensaje.
    """
    color = {
        "ok": TEMA["success"],
        "warning": TEMA["warning"],
        "alert": TEMA["danger"],
    }.get(status, TEMA["primary"])

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={
                "text": "Score de Drift",
                "font": {"size": 13, "color": TEMA["text"]},
            },
            number={
                "valueformat": ".2f",
                "font": {"size": 32, "color": color},
            },
            gauge={
                "axis": {
                    "range": [0, 1],
                    "tickvals": [0, _DRIFT_OK_LIMIT, _DRIFT_WARN_LIMIT, 1],
                    "ticktext": ["0", "0.33", "0.66", "1"],
                    "tickcolor": AXIS_TEXT,
                    "tickfont": {"size": 10, "color": AXIS_TEXT},
                },
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": TEMA["bg_alt"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, _DRIFT_OK_LIMIT], "color": "#D1FAE5"},
                    {"range": [_DRIFT_OK_LIMIT, _DRIFT_WARN_LIMIT], "color": "#FEF3C7"},
                    {"range": [_DRIFT_WARN_LIMIT, 1], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.85,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        font=dict(family="Inter, sans-serif", color=TEMA["text"]),
    )
    return fig


def build_simple_gauge(percentage: float) -> go.Figure:
    """Variante minimal usada en la página de Modelos."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentage,
            title={"text": "Cobertura de Modelos", "font": {"size": 13}},
            number={"suffix": "%", "font": {"size": 30, "color": "#4F46E5"}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#4F46E5"},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#FEE2E2"},
                    {"range": [50, 80], "color": "#FEF3C7"},
                    {"range": [80, 100], "color": "#D1FAE5"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=40, b=10),
        font=dict(family="Inter, sans-serif", color=TEMA["text"]),
    )
    return fig
