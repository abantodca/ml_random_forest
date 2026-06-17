"""Template Plotly registrado para todos los gráficos del frontend.

Importar este módulo registra el template "rnd_forest" como default global
de Plotly, así cada `go.Figure()` hereda paleta, fuente y márgenes sin
spread manual de un dict.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from app.core import TEMA

# Colores derivados de TEMA para garantizar contraste AA.
GRID_COLOR = "rgba(15,23,42,0.06)"     # casi imperceptible pero visible
AXIS_TEXT = TEMA["text_tertiary"]      # #475569 — slate-600
TITLE_TEXT = TEMA["text"]              # #0F172A — slate-900


pio.templates["rnd_forest"] = go.layout.Template(
    layout=dict(
        # Tile blanco a nivel de Plotly: el chart es una superficie blanca
        # propia (estilo Power BI), no hereda el gris del lienzo. El CSS
        # del wrapper [data-testid="stPlotlyChart"] aporta borde/sombra/radio.
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color=TEMA["text_body"], size=12),
        # Márgenes consistentes y aireados; el título vive arriba a la izquierda.
        margin=dict(l=52, r=24, t=58, b=40),
        # Título de reporte: alineado a la izquierda, negrita, color de marca.
        title=dict(
            font=dict(family="Inter, sans-serif", size=14, color=TITLE_TEXT),
            x=0.0,
            xref="paper",
            xanchor="left",
            y=0.97,
            yanchor="top",
            pad=dict(l=4, b=8),
        ),
        # Leyenda limpia: horizontal arriba, sin marco, fuente tenue.
        legend=dict(
            font=dict(size=11, color=TEMA["text_secondary"]),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        # Ejes: gridlines tenues, sin línea de eje ni zeroline (data-ink alto).
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
