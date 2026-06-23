"""Heatmap MAE / R² por variedad."""

from __future__ import annotations

import plotly.graph_objects as go

from app.components.charts._common import AXIS_TEXT, TITLE_TEXT
from app.components.layout.empty_state import empty_state_annotation
from app.core import TEMA


def build_metrics_heatmap(
    names: list[str],
    mae_vals: list[float],
    r2_vals: list[float],
    *,
    mape_vals: list[float] | None = None,
    empty_message: str | None = None,
) -> go.Figure:
    is_empty = not names
    # MAPE es opcional (3ra fila). Las escalas difieren entre filas; el color
    # es indicativo y el valor exacto va en la etiqueta de cada celda.
    if not is_empty:
        z = [mae_vals, r2_vals] + ([mape_vals] if mape_vals is not None else [])
        yl = ["MAE", "R²"] + (["MAPE %"] if mape_vals is not None else [])
    else:
        z = [[0], [0]]
        yl = ["MAE", "R²"]
    xl = names if not is_empty else ["—"]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=xl,
            y=yl,
            # Escala continua perceptual (claro→oscuro) con buen contraste
            # de los números negros en cualquier celda.
            colorscale=[
                [0.00, "#EEF2FF"],  # indigo-50
                [0.25, "#C7D2FE"],  # indigo-200
                [0.50, "#A5B4FC"],  # indigo-300
                [0.75, "#818CF8"],  # indigo-400
                [1.00, "#6366F1"],  # indigo-500
            ],
            text=z,
            texttemplate="<b>%{text:.2f}</b>" if not is_empty else "",
            textfont=dict(size=13, color=TEMA["text"], family="Inter, sans-serif"),
            hovertemplate=("Variedad: %{x}<br>Métrica: %{y}<br>Valor: %{z:.2f}<extra></extra>"),
            colorbar=dict(
                thickness=10,
                len=0.7,
                outlinewidth=0,
                tickfont=dict(size=10, color=AXIS_TEXT),
            ),
            xgap=3,
            ygap=3,
        )
    )
    fig.update_layout(
        title=dict(
            text="Heatmap de Métricas",
            font=dict(size=14, color=TITLE_TEXT, weight=600),
            x=0.01,
        ),
        height=380,
    )
    fig.update_xaxes(
        title_text="",
        tickfont=dict(size=10, color=AXIS_TEXT),
        side="bottom",
    )
    fig.update_yaxes(tickfont=dict(size=11, color=AXIS_TEXT))
    if empty_message:
        fig.add_annotation(**empty_state_annotation(empty_message))
    return fig
