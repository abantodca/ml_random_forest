"""Subplot de tabla + barras (detalle de variedades del Dashboard)."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.components.charts._common import AXIS_TEXT, GRID_COLOR, TITLE_TEXT
from app.components.layout.empty_state import empty_state_annotation
from app.core import TEMA


def _make_table(values: dict[str, list[str]]) -> go.Table:
    headers = ["<b>Variedad</b>", "<b>Modelo</b>", "<b>MAE</b>", "<b>R²</b>"]
    cells = [values["names"], values["status"], values["mae"], values["r2"]]
    n = len(values["names"])
    return go.Table(
        header=dict(
            values=headers,
            fill_color="#EEF2FF",  # indigo-50 — sutil, sobre fondo blanco
            font=dict(
                color=TEMA["primary"],
                size=12,
                family="Inter, sans-serif",
            ),
            align="center",
            height=36,
            line=dict(color=TEMA["border"], width=1),
        ),
        cells=dict(
            values=cells,
            fill_color=[
                [TEMA["bg"] if i % 2 == 0 else TEMA["card"] for i in range(n)]
            ],
            font=dict(size=11, color=TEMA["text_body"]),
            align="center",
            height=30,
            line=dict(color=TEMA["border"], width=0.5),
        ),
    )


def _make_bar(
    names: list[str],
    values: list[float],
    label: str,
    color: str,
    text_color: str,
) -> go.Bar:
    return go.Bar(
        x=names,
        y=values,
        name=label,
        marker=dict(color=color, line=dict(width=0)),
        text=[f"<b>{v:.2f}</b>" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=text_color, family="Inter, sans-serif"),
        cliponaxis=False,
    )


def build_table_with_bars(
    names: list[str],
    mae_vals: list[float],
    r2_vals: list[float],
    table_values: dict[str, list[str]],
    *,
    empty_message: str | None = None,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.45, 0.55],
        specs=[[{"type": "table"}], [{"type": "bar"}]],
        vertical_spacing=0.06,
    )
    fig.add_trace(_make_table(table_values), row=1, col=1)
    fig.add_trace(
        _make_bar(
            names, mae_vals, "MAE",
            "rgba(79,70,229,0.85)", TEMA["primary_dark"],
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        _make_bar(
            names, r2_vals, "R²",
            "rgba(124,58,237,0.85)", "#5B21B6",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=640,
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        showlegend=True,
        legend=dict(
            orientation="v",
            y=0.46,
            x=0.98,
            xanchor="right",
            yanchor="top",
            font=dict(size=11, color=TEMA["text_body"]),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor=TEMA["border"],
            borderwidth=1,
        ),
        title=dict(
            text="Tabla y Comparativa de Métricas",
            font=dict(size=14, color=TITLE_TEXT, weight=600),
        ),
    )
    fig.update_yaxes(
        title_text="",
        row=2,
        col=1,
        gridcolor=GRID_COLOR,
        zeroline=False,
        showticklabels=False,
        range=[0, max(mae_vals + r2_vals, default=1) * 1.2],
    )
    fig.update_xaxes(
        title_text="",
        row=2,
        col=1,
        tickfont=dict(size=11, color=AXIS_TEXT),
    )
    if empty_message:
        fig.add_annotation(**empty_state_annotation(empty_message))
    return fig
