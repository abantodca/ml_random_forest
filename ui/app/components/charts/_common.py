"""Template Plotly registrado para todos los gráficos del frontend.

Importar este módulo registra el template "rnd_forest" como default global
de Plotly, así cada `go.Figure()` hereda paleta, fuente y márgenes sin
spread manual de un dict.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from app.core import TEMA

GRID_COLOR = "rgba(15,23,42,0.06)"
AXIS_TEXT = TEMA["text_tertiary"]
TITLE_TEXT = TEMA["text"]


pio.templates["rnd_forest"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color=TEMA["text_body"], size=12),
        margin=dict(l=52, r=24, t=58, b=40),
        title=dict(
            font=dict(family="Inter, sans-serif", size=14, color=TITLE_TEXT),
            x=0.0,
            xref="paper",
            xanchor="left",
            y=0.97,
            yanchor="top",
            pad=dict(l=4, b=8),
        ),
        legend=dict(
            font=dict(size=11, color=TEMA["text_secondary"]),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            griddash="solid",
            zeroline=False,
            showline=False,
            linecolor=GRID_COLOR,
            ticks="",
            tickfont=dict(size=10, color=AXIS_TEXT),
            title=dict(font=dict(size=11, color=AXIS_TEXT)),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            griddash="solid",
            zeroline=False,
            showline=False,
            linecolor=GRID_COLOR,
            ticks="",
            tickfont=dict(size=10, color=AXIS_TEXT),
            title=dict(font=dict(size=11, color=AXIS_TEXT)),
        ),
        hoverlabel=dict(
            bgcolor=TEMA["text"],
            font_size=12,
            font_color="white",
            bordercolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
        ),
        colorway=[
            TEMA["primary"],
            TEMA["accent"],
            TEMA["info"],
            TEMA["success"],
            TEMA["warning"],
        ],
    )
)
pio.templates.default = "rnd_forest"


def hex_to_rgb(hex_color: str) -> str:
    """Convierte '#4F46E5' -> '79,70,229' (formato para `rgba(...,alpha)`)."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"
