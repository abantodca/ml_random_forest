"""Histograma de KGHORA por variedad (resultados batch)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from app.components.charts._common import AXIS_TEXT, GRID_COLOR, TITLE_TEXT
from app.components.layout.empty_state import empty_state_annotation
from app.core import PALETA_SERIES, TEMA


def build_kghora_histogram(
    preds_df: pd.DataFrame,
    *,
    empty_message: str | None = None,
) -> go.Figure:
    """Distribución de predicciones KGHORA por variedad.

    `empty_message`: texto de estado vacío para mostrar si no hay datos
    (parámetro opcional → backward-compatible).
    """
    fig = go.Figure()
    varieties = preds_df["variedad"].unique() if not preds_df.empty else []
    for i, variety in enumerate(varieties):
        subset = preds_df[preds_df["variedad"] == variety]
        vals = subset["kghora_pred"].tolist()
        n = len(vals)
        mean_v = sum(vals) / n if n else 0.0
        fig.add_trace(
            go.Histogram(
                x=vals,
                nbinsx=15,
                opacity=0.75,
                name=variety,
                marker=dict(
                    color=PALETA_SERIES[i % len(PALETA_SERIES)],
                    line=dict(color="white", width=1),
                ),
                hovertemplate=(
                    f"<b>{variety}</b><br>"
                    "KGHORA: %{x:.2f} kg/h<br>"
                    "Frecuencia: <b>%{y}</b><br>"
                    f"Media variedad: <b>{mean_v:.2f}</b> kg/h"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=dict(
            text="Distribución de Predicciones KGHORA por Variedad",
            font=dict(size=14, color=TITLE_TEXT, weight=600),
            x=0.01,
        ),
        xaxis_title="KGHORA (kg/h)",
        yaxis_title="Frecuencia",
        barmode="overlay",
        height=360,
        legend=dict(
            font=dict(size=11, color=TEMA["text_secondary"]),
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        tickfont=dict(size=10, color=AXIS_TEXT),
        title_font=dict(size=11, color=AXIS_TEXT),
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        tickfont=dict(size=10, color=AXIS_TEXT),
        title_font=dict(size=11, color=AXIS_TEXT),
    )
    if empty_message or preds_df.empty:
        msg = empty_message or "Sin predicciones para mostrar"
        fig.add_annotation(**empty_state_annotation(msg))
    return fig
