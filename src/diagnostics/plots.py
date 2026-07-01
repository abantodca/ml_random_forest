"""Plotly figures reutilizables para el reporte EDA.

Cada funcion devuelve un `plotly.graph_objects.Figure` ya configurado.
El `html_renderer` hace el `to_html(include_plotlyjs=False, full_html=False)`
para embeber inline sin duplicar plotly.js (se carga UNA vez en el head).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

_AXIS_TITLE_FONT = dict(size=11, color="#475569")
_FIG_FONT = dict(family="Inter, system-ui, sans-serif", size=11, color="#1e293b")


def style_fig(fig: go.Figure, title: str = "") -> go.Figure:
    """Aplica el tema visual base (consistente con el dashboard ejecutivo)."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#0f172a")) if title else None,
        margin=dict(l=40, r=20, t=40 if title else 20, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        font=_FIG_FONT,
        showlegend=False,
        height=320,
    )
    fig.update_xaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", title_font=_AXIS_TITLE_FONT)
    fig.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", title_font=_AXIS_TITLE_FONT)
    return fig


def histogram_with_kde(x: pd.Series, name: str) -> go.Figure:
    """Histograma + densidad KDE superpuesta + lineas en mean/median."""
    x_clean = pd.to_numeric(x, errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=x_clean,
            nbinsx=40,
            name="freq",
            marker_color="#3b82f6",
            opacity=0.65,
            hovertemplate="bin=%{x}<br>n=%{y}<extra></extra>",
        )
    )
    if x_clean.std() > 0 and len(x_clean) > 30:
        # KDE manual (gaussian) sobre 200 puntos
        from scipy.stats import gaussian_kde

        try:
            kde = gaussian_kde(x_clean)
            xs = np.linspace(x_clean.min(), x_clean.max(), 200)
            ys = kde(xs)
            # Reescalar a freq*bin_width (aprox)
            bin_w = (x_clean.max() - x_clean.min()) / 40
            ys_scaled = ys * len(x_clean) * bin_w
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys_scaled,
                    mode="lines",
                    line=dict(color="#dc2626", width=2),
                    hoverinfo="skip",
                    name="kde",
                )
            )
        except Exception as exc:
            # El overlay KDE es decorativo: si scipy falla (matriz singular,
            # data degenerada) el histograma sigue siendo valido.
            logger.debug("KDE overlay omitido del histograma: %s", exc)
    fig.add_vline(
        x=float(x_clean.mean()),
        line_dash="dash",
        line_color="#16a34a",
        annotation_text="mean",
        annotation_position="top",
    )
    fig.add_vline(
        x=float(x_clean.median()),
        line_dash="dot",
        line_color="#9333ea",
        annotation_text="median",
        annotation_position="bottom",
    )
    return style_fig(fig, title=f"{name} — distribucion")


def qq_plot(x: pd.Series, name: str) -> go.Figure:
    """Q-Q plot vs distribucion normal teorica."""
    from scipy.stats import probplot

    x_clean = pd.to_numeric(x, errors="coerce").dropna()
    fig = go.Figure()
    if len(x_clean) < 10:
        fig.add_annotation(text="n insuficiente", showarrow=False, x=0.5, y=0.5)
        return style_fig(fig, title=f"{name} — Q-Q normal")
    (osm, osr), _ = probplot(x_clean, dist="norm")
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker=dict(color="#3b82f6", size=4, opacity=0.6),
            hovertemplate="theoretical=%{x:.2f}<br>sample=%{y:.2f}<extra></extra>",
        )
    )
    # Linea referencia y=x escalada
    lo, hi = osm.min(), osm.max()
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo * x_clean.std() + x_clean.mean(), hi * x_clean.std() + x_clean.mean()],
            mode="lines",
            line=dict(color="#dc2626", dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title="cuantiles teoricos (normal)")
    fig.update_yaxes(title="cuantiles muestra")
    return style_fig(fig, title=f"{name} — Q-Q normal")


def boxplot_by_group(df: pd.DataFrame, value_col: str, group_col: str, name: str) -> go.Figure:
    """Boxplot del value_col agrupado por group_col."""
    fig = go.Figure()
    if group_col not in df.columns:
        fig.add_annotation(text=f"sin columna {group_col}", showarrow=False, x=0.5, y=0.5)
        return style_fig(fig, title=name)
    groups = sorted(df[group_col].dropna().astype(str).unique().tolist())
    for g in groups[:30]:  # cap a 30 para no romper layout
        vals = pd.to_numeric(
            df.loc[df[group_col].astype(str) == g, value_col], errors="coerce"
        ).dropna()
        if len(vals) < 3:
            continue
        fig.add_trace(
            go.Box(
                y=vals,
                name=g,
                boxpoints="suspectedoutliers",
                marker=dict(color="#3b82f6", outliercolor="#dc2626", size=3),
            )
        )
    fig.update_layout(showlegend=False, xaxis=dict(tickangle=-45))
    return style_fig(fig, title=f"{name} por {group_col}")


def acf_pacf_bars(acf: list[float], pacf: list[float], n: int, name: str) -> go.Figure:
    """Barplot dual ACF + PACF con bandas de significancia (95%)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"), horizontal_spacing=0.12)
    if not acf:
        fig.add_annotation(text="ACF/PACF no disponibles", showarrow=False, x=0.5, y=0.5)
        return style_fig(fig, title=name)
    threshold = 1.96 / np.sqrt(n) if n > 0 else 0.0
    lags = list(range(len(acf)))
    fig.add_trace(go.Bar(x=lags, y=acf, marker_color="#3b82f6"), row=1, col=1)
    fig.add_trace(go.Bar(x=lags, y=pacf, marker_color="#7c3aed"), row=1, col=2)
    for col in (1, 2):
        fig.add_hline(y=threshold, line=dict(color="#dc2626", dash="dash", width=1), row=1, col=col)
        fig.add_hline(
            y=-threshold, line=dict(color="#dc2626", dash="dash", width=1), row=1, col=col
        )
        fig.add_hline(y=0, line=dict(color="#94a3b8", width=1), row=1, col=col)
    fig.update_xaxes(title="lag", row=1, col=1)
    fig.update_xaxes(title="lag", row=1, col=2)
    return style_fig(fig, title=f"{name} — autocorrelacion")


def correlation_heatmap(
    corr_cols: list[str], corr_matrix: list[list[float]], method: str
) -> go.Figure:
    """Heatmap de correlation. Trim de >40 columnas para legibilidad."""
    if not corr_cols or not corr_matrix:
        fig = go.Figure()
        fig.add_annotation(text="sin features numericas suficientes", showarrow=False, x=0.5, y=0.5)
        return style_fig(fig, title="Correlation matrix")
    # Trim si mas de 40
    cols = corr_cols[:40]
    mat = [row[:40] for row in corr_matrix[:40]]
    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=cols,
            y=cols,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate="x=%{x}<br>y=%{y}<br>r=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=max(360, 24 + 16 * len(cols)), xaxis=dict(tickangle=-45))
    return style_fig(fig, title=f"Correlation matrix ({method})")


def vif_bars(vif_results) -> go.Figure:
    """Bar chart horizontal de VIF, coloreado por severidad."""
    fig = go.Figure()
    if not vif_results:
        fig.add_annotation(text="VIF no disponible", showarrow=False, x=0.5, y=0.5)
        return style_fig(fig, title="VIF")
    sev_color = {"high": "#dc2626", "watch": "#f59e0b", "ok": "#16a34a"}
    items = sorted(vif_results, key=lambda r: r.vif, reverse=True)[:25]
    fig.add_trace(
        go.Bar(
            x=[r.vif for r in items],
            y=[r.feature for r in items],
            orientation="h",
            marker_color=[sev_color[r.severity] for r in items],
            hovertemplate="<b>%{y}</b><br>VIF=%{x:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=5, line_dash="dot", line_color="#94a3b8")
    fig.add_vline(x=10, line_dash="dash", line_color="#dc2626")
    fig.update_xaxes(title="VIF (>5 watch, >10 high)")
    fig.update_layout(height=max(280, 20 * len(items)))
    return style_fig(fig, title="Variance Inflation Factor")


def mi_bars(mi_results) -> go.Figure:
    """Bar chart horizontal de MI con target (top-25)."""
    fig = go.Figure()
    if not mi_results:
        fig.add_annotation(text="MI no disponible", showarrow=False, x=0.5, y=0.5)
        return style_fig(fig, title="Mutual Information")
    items = mi_results[:25]
    fig.add_trace(
        go.Bar(
            x=[r.mi for r in items],
            y=[r.feature for r in items],
            orientation="h",
            marker_color="#3b82f6",
            hovertemplate="<b>%{y}</b><br>MI=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="MI con target (mayor = mas informativa)")
    fig.update_layout(height=max(280, 20 * len(items)))
    return style_fig(fig, title="Mutual Information vs target (top 25)")


def psi_heatmap(drift_reports) -> go.Figure:
    """Heatmap PSI: filas=variables, cols=transiciones de anios."""
    if not drift_reports:
        fig = go.Figure()
        fig.add_annotation(text="sin drift reports", showarrow=False, x=0.5, y=0.5)
        return style_fig(fig, title="PSI por anio")

    transitions = sorted({p for r in drift_reports for p in r.year_pairs})
    if not transitions:
        fig = go.Figure()
        fig.add_annotation(text="<2 anios en data", showarrow=False, x=0.5, y=0.5)
        return style_fig(fig, title="PSI por anio")

    x_labels = [f"{a}->{b}" for a, b in transitions]
    y_labels = [r.variable for r in drift_reports]
    z = []
    for r in drift_reports:
        row = []
        for p in transitions:
            v = r.psi_values.get(p, float("nan"))
            row.append(v)
        z.append(row)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale="YlOrRd",
            zmin=0,
            zmax=0.5,
            hovertemplate="var=%{y}<br>%{x}<br>PSI=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(height=max(280, 22 * len(y_labels)))
    return style_fig(fig, title="Population Stability Index (>0.25 severo)")
