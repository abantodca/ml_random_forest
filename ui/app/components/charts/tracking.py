"""Gráficos del seguimiento de precisión (proyectado vs real).

Pocos y con propósito: cada figura responde una pregunta concreta del día a
día con pronósticos. Funciones puras (reciben listas, devuelven `go.Figure`).
"""

from __future__ import annotations

import plotly.graph_objects as go

from app.components.charts._common import AXIS_TEXT, GRID_COLOR, TITLE_TEXT, hex_to_rgb
from app.components.layout.empty_state import empty_state_annotation
from app.core import TEMA


def _title(text: str) -> dict:
    return dict(text=text, font=dict(size=14, color=TITLE_TEXT, weight=600), x=0.01)


def _axes(fig: go.Figure, *, ytitle: str = "", xtitle: str = "") -> None:
    """Aplica estilo de cuadrícula/ticks a todos los ejes.

    `ytitle`/`xtitle`: título de eje a preservar (default="" → sin título,
    igual que antes). Los builders que necesitan etiqueta de eje lo pasan
    aquí en lugar de vía `update_layout` para garantizar que _axes no lo borre.
    """
    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
        tickfont=dict(size=10, color=AXIS_TEXT),
        title_text=ytitle,
        title_font=dict(size=11, color=AXIS_TEXT),
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        tickfont=dict(size=10, color=AXIS_TEXT),
        title_text=xtitle,
        title_font=dict(size=11, color=AXIS_TEXT),
    )


def _legend() -> dict:
    return dict(
        orientation="h",
        y=-0.18,
        x=0.5,
        xanchor="center",
        font=dict(size=11, color=TEMA["text_secondary"]),
        bgcolor="rgba(0,0,0,0)",
    )


def _moving_average(values: list[float], window: int) -> list[float | None]:
    """Media móvil centrada con ventana `window` (None en los extremos sin datos)."""
    n = len(values)
    half = window // 2
    result: list[float | None] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        segment = values[lo:hi]
        result.append(sum(segment) / len(segment) if segment else None)
    return result


def build_pred_vs_real_line(
    fechas: list[str],
    proyectado: list[float],
    real: list[float],
    *,
    empty_message: str | None = None,
) -> go.Figure:
    """¿Le atina el pronóstico en el tiempo? Dos líneas: proyectado vs real.

    El hover unificado muestra también el error absoluto (proyectado − real)
    para que el usuario no tenga que calcularlo mentalmente.
    """
    # Calcular error para enriquecer el hover (customdata[0])
    errors = (
        [p - r for p, r in zip(proyectado, real, strict=True)]
        if len(proyectado) == len(real)
        else []
    )
    has_errors = bool(errors)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fechas,
            y=proyectado,
            mode="lines+markers",
            name="Proyectado",
            line=dict(color=TEMA["primary"], width=2.5),
            marker=dict(size=7),
            customdata=[[e] for e in errors] if has_errors else None,
            hovertemplate=(
                "%{x}<br>Proyectado: <b>%{y:.2f}</b> kg/h"
                + ("<br>Error: <b>%{customdata[0]:+.2f}</b>" if has_errors else "")
                + "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fechas,
            y=real,
            mode="lines+markers",
            name="Real",
            line=dict(color=TEMA["success"], width=2.5),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate="%{x}<br>Real: <b>%{y:.2f}</b> kg/h<extra></extra>",
        )
    )
    fig.update_layout(
        title=_title("Proyectado vs Real (KGHORA)"),
        height=380,
        hovermode="x unified",
        legend=_legend(),
    )
    _axes(fig, ytitle="KGHORA (kg/h)")
    if empty_message:
        fig.add_annotation(**empty_state_annotation(empty_message))
    return fig


def build_parity_plot(
    real: list[float],
    proyectado: list[float],
    *,
    empty_message: str | None = None,
) -> go.Figure:
    """¿Qué tan preciso es? Paridad real (x) vs proyectado (y) con línea y=x.

    Puntos sobre la diagonal = pronóstico perfecto; dispersión = error. El
    título anota R² (varianza explicada) y MAPE (error porcentual medio): el
    par de métricas estándar para juzgar un pronóstico de un vistazo.
    La banda ±MAE alrededor de y=x muestra qué tan lejos es "demasiado lejos"
    para el error típico de este modelo.
    """
    n = len(real)
    if n:
        mean_r = sum(real) / n
        ss_tot = sum((r - mean_r) ** 2 for r in real) or 1e-9
        ss_res = sum((r - p) ** 2 for r, p in zip(real, proyectado, strict=True))
        r2 = 1.0 - ss_res / ss_tot
        pcts = [abs(r - p) / r for r, p in zip(real, proyectado, strict=True) if r]
        mape = (sum(pcts) / len(pcts) * 100.0) if pcts else 0.0
        mae = sum(abs(r - p) for r, p in zip(real, proyectado, strict=True)) / n
        lo = min(min(real), min(proyectado))
        hi = max(max(real), max(proyectado))
    else:
        r2 = mape = mae = 0.0
        lo, hi = 0.0, 1.0

    # Errors for customdata hover
    errors = [p - r for r, p in zip(real, proyectado, strict=True)]
    pct_errors = [(p - r) / r * 100.0 if r else None for r, p in zip(real, proyectado, strict=True)]
    customdata = [[e, pe] for e, pe in zip(errors, pct_errors, strict=True)]

    fig = go.Figure()

    # Banda ±MAE alrededor de y=x: zona de "error típico aceptable"
    if n and mae > 0:
        band_rgb = hex_to_rgb(TEMA["success"])
        fig.add_trace(
            go.Scatter(
                x=[lo, hi, hi, lo, lo],
                y=[lo + mae, hi + mae, hi - mae, lo - mae, lo + mae],
                fill="toself",
                fillcolor=f"rgba({band_rgb},0.08)",
                line=dict(width=0),
                name=f"±MAE ({mae:.2f})",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Perfecto (y=x)",
            line=dict(color=TEMA["success"], dash="dash", width=2),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=real,
            y=proyectado,
            mode="markers",
            name="Pronósticos",
            marker=dict(
                size=9, color=TEMA["primary"], line=dict(width=1, color="white"), opacity=0.85
            ),
            customdata=customdata,
            hovertemplate=(
                "Real: <b>%{x:.2f}</b> kg/h<br>"
                "Proyectado: <b>%{y:.2f}</b> kg/h<br>"
                "Error: <b>%{customdata[0]:+.2f}</b> kg/h"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=_title(f"Paridad real vs proyectado · R²={r2:.2f} · MAPE={mape:.1f}%"),
        height=420,
        legend=_legend(),
    )
    _axes(fig, xtitle="Real (KGHORA · kg/h)", ytitle="Proyectado (KGHORA · kg/h)")
    if empty_message:
        fig.add_annotation(**empty_state_annotation(empty_message))
    return fig


def build_residual_bars(
    fechas: list[str],
    residuals: list[float],
    *,
    empty_message: str | None = None,
    rolling_window: int | None = None,
) -> go.Figure:
    """¿Está sesgado / empeora? Barras de error (pred−real) + sesgo + banda ±1σ.

    `rolling_window`: ventana de la media móvil superpuesta (None → automático:
    min(7, n//3) si n≥6; sin línea si hay menos puntos). La media móvil revela
    si los errores tienen tendencia sistemática que las barras individuales
    ocultan. Parámetro opcional → backward-compatible.
    """
    colors = [TEMA["warning"] if r > 0 else TEMA["info"] for r in residuals]
    fig = go.Figure(
        go.Bar(
            x=fechas,
            y=residuals,
            marker_color=colors,
            hovertemplate="%{x}<br>Error: <b>%{y:+.2f}</b> kg/h<extra></extra>",
            name="Error diario",
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_color=AXIS_TEXT, line_width=1)

    if residuals:
        bias = sum(residuals) / len(residuals)
        sd = (sum((r - bias) ** 2 for r in residuals) / len(residuals)) ** 0.5

        # Banda ±1σ alrededor del sesgo: dónde cae el ~68% de los errores.
        fig.add_hrect(
            y0=bias - sd, y1=bias + sd, fillcolor=TEMA["accent"], opacity=0.08, line_width=0
        )
        fig.add_hline(
            y=bias,
            line_dash="dash",
            line_color=TEMA["accent"],
            annotation_text=f"sesgo {bias:+.2f} · σ {sd:.2f}",
            annotation_font=dict(size=10, color=TEMA["accent"]),
        )

        # Media móvil: revela tendencia sistemática de los errores
        n = len(residuals)
        win = rolling_window if rolling_window is not None else (min(7, n // 3) if n >= 6 else 0)
        if win >= 3:
            ma = _moving_average(residuals, win)
            # Filtrar None (no deberían aparecer con _moving_average actual)
            ma_clean = [v if v is not None else float("nan") for v in ma]
            fig.add_trace(
                go.Scatter(
                    x=fechas,
                    y=ma_clean,
                    mode="lines",
                    name=f"Media móvil ({win}p)",
                    line=dict(color=TEMA["text"], width=2, dash="dot"),
                    hovertemplate="%{x}<br>Media móvil: <b>%{y:+.2f}</b><extra></extra>",
                )
            )
            fig.update_layout(showlegend=True, legend=_legend())

    fig.update_layout(
        title=_title("Error en el tiempo (sobreestima ▲ / subestima ▼)"),
        height=360,
    )
    _axes(fig, ytitle="Error (kg/h)")
    if empty_message:
        fig.add_annotation(**empty_state_annotation(empty_message))
    return fig


def build_decomp_scatter(
    err_data: list[float],
    err_model: list[float],
    labels: list[str],
    *,
    empty_message: str | None = None,
) -> go.Figure:
    """¿Culpa de los datos o del modelo? Scatter por cuadrantes.

    x = error atribuible a la proyección (KG/HA); y = error del modelo dado el
    input real. Lejos del eje X → datos malos; lejos del eje Y → modelo malo.

    Los cuadrantes están sombreados para que el usuario ubique los puntos
    sin necesidad de interpretar signos: Q1 (++, ambos sobreestiman),
    Q2 (−+, datos subestiman/modelo sobreestima), etc.
    """
    # Magnitudes para calcular error total en hover (customdata)
    err_total = [d + m for d, m in zip(err_data, err_model, strict=True)]
    customdata = [[et] for et in err_total]

    fig = go.Figure()

    # Sombreado de cuadrantes (muy tenue)
    if err_data and err_model:
        x_range = max(abs(v) for v in err_data) * 1.2 or 1.0
        y_range = max(abs(v) for v in err_model) * 1.2 or 1.0
        warning_rgb = hex_to_rgb(TEMA["warning"])
        info_rgb = hex_to_rgb(TEMA["info"])
        # Q1(++)/Q3(--): ambos sesgados en el mismo sentido → más grave (warning)
        for x0, x1, y0, y1 in [
            (0, x_range, 0, y_range),  # Q1
            (-x_range, 0, -y_range, 0),  # Q3
        ]:
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                fillcolor=f"rgba({warning_rgb},0.04)",
                line_width=0,
                layer="below",
            )
        # Q2/Q4: sesgos opuestos se cancelan parcialmente → info
        for x0, x1, y0, y1 in [
            (-x_range, 0, 0, y_range),  # Q2
            (0, x_range, -y_range, 0),  # Q4
        ]:
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                fillcolor=f"rgba({info_rgb},0.04)",
                line_width=0,
                layer="below",
            )

    fig.add_trace(
        go.Scatter(
            x=err_data,
            y=err_model,
            mode="markers",
            text=labels,
            marker=dict(
                size=10, color=TEMA["primary"], line=dict(width=1, color="white"), opacity=0.85
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Error datos: <b>%{x:+.2f}</b> kg/h<br>"
                "Error modelo: <b>%{y:+.2f}</b> kg/h<br>"
                "Error total: <b>%{customdata[0]:+.2f}</b> kg/h"
                "<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_color=AXIS_TEXT, line_width=1)
    fig.add_vline(x=0, line_color=AXIS_TEXT, line_width=1)
    fig.update_layout(
        title=_title("Diagnóstico: error de datos vs error de modelo"),
        height=420,
        showlegend=False,
    )
    _axes(
        fig,
        xtitle="Error de datos (proyección de KG/HA · kg/h)",
        ytitle="Error de modelo (input real · kg/h)",
    )
    if empty_message:
        fig.add_annotation(**empty_state_annotation(empty_message))
    return fig


def build_weekly_bars(
    weeks: list[str],
    proyectado: list[float],
    real: list[float],
    *,
    empty_message: str | None = None,
) -> go.Figure:
    """¿Cómo cierra la semana? Barras agrupadas proyectado vs real por semana ISO.

    El hover incluye el Δ% (desvío porcentual de la semana) para que el
    responsable no tenga que calcularlo fuera del gráfico.
    """
    # Δ% por semana para customdata
    pct = [((p - r) / r * 100.0) if r else None for p, r in zip(proyectado, real, strict=True)]
    cd_proj = [[p, pt] for p, pt in zip(proyectado, pct, strict=True)]
    cd_real = [[r, pt] for r, pt in zip(real, pct, strict=True)]

    # Peor semana = mayor |Δ%| entre las que tienen real → contorno rojo en
    # ambas barras para que salte a la vista cuál cerró peor.
    _valid = [(i, abs(p)) for i, p in enumerate(pct) if p is not None]
    _worst = max(_valid, key=lambda t: t[1])[0] if _valid else -1
    _line_w = [2.8 if i == _worst else 0 for i in range(len(weeks))]
    _line_c = [TEMA["danger"]] * len(weeks)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=weeks,
            y=proyectado,
            name="Proyectado",
            marker=dict(color=TEMA["primary"], line=dict(color=_line_c, width=_line_w)),
            customdata=cd_proj,
            hovertemplate=(
                "<b>%{x}</b><br>Proyectado: <b>%{customdata[0]:.1f}</b><br>"
                "Δ%: <b>%{customdata[1]:+.1f}%</b><extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=weeks,
            y=real,
            name="Real",
            marker=dict(color=TEMA["success"], line=dict(color=_line_c, width=_line_w)),
            customdata=cd_real,
            hovertemplate=(
                "<b>%{x}</b><br>Real: <b>%{customdata[0]:.1f}</b><br>"
                "Δ%: <b>%{customdata[1]:+.1f}%</b><extra></extra>"
            ),
        )
    )
    fig.update_layout(
        barmode="group",
        title=_title("Cierre semanal (ISO): proyectado vs real"),
        height=360,
        legend=_legend(),
    )
    _axes(fig, ytitle="KGHORA acumulada")
    if empty_message:
        fig.add_annotation(**empty_state_annotation(empty_message))
    return fig
