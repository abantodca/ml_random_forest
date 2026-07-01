"""Diagnosticos y graficos del pipeline final (Plotly).

Funciones vivas (todas devuelven un <div> HTML listo para embeber):
    - plot_pred_vs_actual_plotly      : scatter Predicho vs Real (OOF / refit).
    - plot_calibration_plotly         : curva de calibracion por bins.
    - plot_error_histogram_plotly     : histograma de errores absolutos / signados.
    - plot_residuals_vs_predicted_plotly : residuales vs prediccion (patrones
      de sesgo, heteroscedasticidad).
    - plot_error_over_time_plotly     : error a lo largo del tiempo (drift).
    - plot_partial_dependence_plotly  : PDP/ICE-style para una feature.

Gauges, importancias y boxplots se removieron cuando el dashboard ejecutivo
paso a renderizarlos inline en `html/winner_dashboard.py`. Si vuelven a
hacer falta, recuperar desde git historico antes que reescribir.

El script plotly.js se carga UNA sola vez desde el <head> del HTML (ver
`html.styles._PLOTLY_JS_TAG`); por eso `plotly_div` (importado de
`html.helpers`) siempre pasa `include_plotlyjs=False`.
"""

from __future__ import annotations

import numpy as np

from src.step_05_evaluate.html.helpers import plotly_div

# Paleta corporativa
_PRIMARY = "#0F766E"
_DANGER = "#B91C1C"

# Tope de puntos por scatter embebido en el HTML. Por encima de esto el
# scatter no gana legibilidad y si infla el archivo (cada punto son ~2
# floats en JSON); se muestrea de forma DETERMINISTA (paso uniforme, sin
# RNG) para que el mismo run produzca el mismo HTML.
_SCATTER_MAX_POINTS = 2500


def _sample_step(n: int, cap: int = _SCATTER_MAX_POINTS) -> np.ndarray:
    """Indices con paso uniforme para muestrear n puntos a <= cap."""
    if n <= cap:
        return np.arange(n)
    return np.linspace(0, n - 1, cap).astype(int)


_PLOTLY_LAYOUT_DEFAULTS = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif", size=11),
    margin=dict(l=60, r=20, t=50, b=50),
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=_PRIMARY),
)

# Alias retro-compatible para callers internos de este modulo. La fuente
# de verdad es `html.helpers.plotly_div`.



def plot_pred_vs_actual_plotly(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicho vs Real (out-of-fold)",
    x_label: str = "Valor real",
    y_label: str = "Prediccion (OOF)",
    color: str = _PRIMARY,
) -> str:
    """Scatter interactivo con linea ideal y = x. Usa scattergl si n>5000."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return ""

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    idx = _sample_step(y_true.size)
    y_true = np.round(y_true[idx], 3)
    y_pred = np.round(y_pred[idx], 3)

    Trace = go.Scattergl if y_true.size > 5000 else go.Scatter
    fig = go.Figure()
    fig.add_trace(
        Trace(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="observaciones",
            marker=dict(color=color, size=5, opacity=0.45, line=dict(width=0)),
            hovertemplate=f"{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="y = x (ideal)",
            line=dict(color=_DANGER, dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=_PRIMARY)),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=380,
        showlegend=True,
        legend=dict(font=dict(size=10), x=0.02, y=0.98, bgcolor="rgba(255,255,255,.7)"),
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    fig.update_yaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    return plotly_div(fig)


def plot_calibration_plotly(calibration_df) -> str:
    """Calibration plot para regresion: bin de pred vs media de real.

    Modelo perfectamente calibrado -> puntos sobre la diagonal y=x.
    Puntos arriba de la diagonal = modelo subestima en ese bin.
    Puntos abajo = sobreestima.
    """
    if calibration_df is None or len(calibration_df) == 0:
        return ""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""

    bin_pred = calibration_df["bin_pred_mean"].to_numpy(dtype=float)
    bin_real = calibration_df["bin_real_mean"].to_numpy(dtype=float)
    counts = calibration_df["bin_count"].to_numpy(dtype=int)

    lo = float(min(bin_pred.min(), bin_real.min()))
    hi = float(max(bin_pred.max(), bin_real.max()))

    fig = go.Figure()
    # Diagonal ideal
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Calibrado perfecto (y=x)",
            line=dict(color=_DANGER, dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )
    # Puntos por bin (size proporcional a count)
    fig.add_trace(
        go.Scatter(
            x=bin_pred,
            y=bin_real,
            mode="markers+lines",
            name="Bins de prediccion (10)",
            marker=dict(
                color=_PRIMARY,
                size=8 + counts / counts.max() * 14,
                opacity=0.75,
                line=dict(color="white", width=1),
            ),
            line=dict(color=_PRIMARY, width=1, dash="dot"),
            hovertemplate=(
                "Predicho promedio: %{x:.2f}<br>"
                "Real promedio: %{y:.2f}<br>"
                "n filas: %{text}<extra></extra>"
            ),
            text=counts,
        )
    )
    fig.update_layout(
        title=dict(text="Calibracion: predicho vs real (bins)", font=dict(size=14, color=_PRIMARY)),
        xaxis_title="Prediccion media (bin)",
        yaxis_title="Real media (bin)",
        height=350,
        showlegend=True,
        legend=dict(font=dict(size=10), x=0.02, y=0.98, bgcolor="rgba(255,255,255,.7)"),
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    fig.update_yaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    return plotly_div(fig, div_id="calibration_plot")


def plot_error_histogram_plotly(
    abs_errors: np.ndarray,
    p50: float,
    p90: float,
    p99: float,
) -> str:
    """Histograma del error absoluto con marcadores p50/p90/p99.

    Una distribucion ancha o con cola larga indica que el modelo se
    equivoca mucho en algunos casos (worst-case scenario peor que el
    promedio). El p99 muestra el tope realista de error.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""
    if abs_errors.size == 0:
        return ""

    # Pre-binnear en Python (np.histogram) en vez de embeber el array
    # completo en el HTML: 60 barras pesan ~1 KB; el array crudo, cientos.
    counts, edges = np.histogram(abs_errors, bins=60)
    centers = np.round((edges[:-1] + edges[1:]) / 2, 3)
    width = float(edges[1] - edges[0])
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=width,
            marker=dict(color=_PRIMARY, opacity=0.78),
            hovertemplate="Error ~%{x:.2f}<br>n: %{y}<extra></extra>",
            name="Distribución",
        )
    )
    # Lineas verticales para percentiles clave
    for _pct, val, color, label in [
        (50, p50, "#2ca02c", f"p50: {p50:.2f}"),
        (90, p90, "#f59e0b", f"p90: {p90:.2f}"),
        (99, p99, _DANGER, f"p99: {p99:.2f}"),
    ]:
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color=color,
            line_width=2,
            annotation_text=label,
            annotation_position="top",
            annotation_font=dict(size=10, color=color),
        )
    fig.update_layout(
        title=dict(
            text="Distribución del error absoluto (OOF · KG/JR)", font=dict(size=14, color=_PRIMARY)
        ),
        xaxis_title="Error absoluto (kg/jornal)",
        yaxis_title="Cantidad de cosechas",
        height=320,
        showlegend=False,
        bargap=0.04,
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    fig.update_yaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    return plotly_div(fig, div_id="error_hist")


def plot_residuals_vs_predicted_plotly(
    y_pred: np.ndarray,
    residuals: np.ndarray,
) -> str:
    """Scatter de residuos (real - predicho) vs prediccion.

    Lectura:
      - Nube alrededor de y=0 sin patron -> modelo OK.
      - Banda que se ensancha al subir x -> heteroscedasticidad
        (errores mas grandes para predicciones grandes; el modelo es
        menos confiable en extremos).
      - Tendencia (linea de regresion no plana) -> sesgo sistematico:
        el modelo sub o sobreestima en cierto rango.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""
    if y_pred.size == 0:
        return ""
    idx = _sample_step(np.asarray(y_pred).size)
    y_pred = np.round(np.asarray(y_pred, dtype=float)[idx], 3)
    residuals = np.round(np.asarray(residuals, dtype=float)[idx], 3)
    Trace = go.Scattergl if y_pred.size > 5000 else go.Scatter

    fig = go.Figure()
    fig.add_trace(
        Trace(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(color=_PRIMARY, size=4, opacity=0.4),
            name="residuos",
            hovertemplate=("Predicción: %{x:.2f}<br>Residuo: %{y:+.2f}<extra></extra>"),
        )
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=_DANGER,
        line_width=1.5,
        annotation_text="ideal (residuo = 0)",
        annotation_position="top right",
        annotation_font=dict(size=10, color=_DANGER),
    )
    fig.update_layout(
        title=dict(text="Residuos vs predicción (KG/JR)", font=dict(size=14, color=_PRIMARY)),
        xaxis_title="Predicción del modelo (kg/jornal)",
        yaxis_title="Residuo = real - predicho (kg/jornal)",
        height=340,
        showlegend=False,
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    fig.update_yaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    return plotly_div(fig, div_id="residuals_plot")


def plot_error_over_time_plotly(
    dates,
    abs_errors: np.ndarray,
    rolling_window: int = 30,
) -> str:
    """Error absoluto a lo largo del tiempo + media movil 30d.

    Si la media movil tiene pendiente positiva al final del periodo, el
    modelo empeoro en el tiempo (drift): puede ser senal de que el
    re-entrenamiento esta tardando o que la realidad cambio.
    """
    try:
        import pandas as _pd
        import plotly.graph_objects as go
    except ImportError:
        return ""
    try:
        s = _pd.Series(abs_errors, index=_pd.to_datetime(dates))
        s = s.dropna().sort_index()
        if s.empty:
            return ""
        rolling = s.rolling(f"{rolling_window}D", min_periods=5).mean()
    except Exception:
        return ""

    # La media movil se calcula sobre TODA la serie; solo los marcadores
    # individuales se muestrean para no inflar el HTML. Las fechas van como
    # 'YYYY-MM-DD' (los Timestamp se serializan como ISO completo con hora,
    # el doble de bytes sin informacion extra: el dato es diario).
    idx = _sample_step(s.size)
    s_pts = s.iloc[idx].round(3)

    fig = go.Figure()
    Trace = go.Scattergl if s_pts.size > 5000 else go.Scatter
    fig.add_trace(
        Trace(
            x=s_pts.index.strftime("%Y-%m-%d"),
            y=s_pts.values,
            mode="markers",
            marker=dict(color=_PRIMARY, size=3, opacity=0.35),
            name="Error por cosecha",
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Error: %{y:.2f} kg/jornal<extra></extra>",
        )
    )
    rolling = rolling.iloc[_sample_step(rolling.size)].round(3)
    fig.add_trace(
        go.Scatter(
            x=rolling.index.strftime("%Y-%m-%d"),
            y=rolling.values,
            mode="lines",
            line=dict(color=_DANGER, width=2.2),
            name=f"Media móvil {rolling_window}d",
            hovertemplate="Fecha: %{x|%Y-%m-%d}<br>Media móvil: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Error absoluto en el tiempo (KG/JR · ventana {rolling_window}d)",
            font=dict(size=14, color=_PRIMARY),
        ),
        xaxis_title="Fecha de cosecha",
        yaxis_title="Error absoluto (kg/jornal)",
        height=320,
        showlegend=True,
        legend=dict(font=dict(size=10), x=0.02, y=0.98, bgcolor="rgba(255,255,255,.7)"),
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    fig.update_yaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    return plotly_div(fig, div_id="error_time")


def plot_partial_dependence_plotly(
    pipeline,
    X_sample,
    feature_names: list,
    top_k: int = 5,
) -> str:
    """Partial Dependence Plot (PDP) para las top-k features mas importantes.

    PDP estadistico (Friedman 2001): efecto marginal promedio de UNA feature
    sobre la prediccion, integrando el resto. Mas honesto que feature
    importance bruta (que solo dice cuanto se usa, no como afecta).

    Estrategia:
      1. Selecciona top-k features por feature_importances_ del modelo final
         (LGB/XGB exponen este atributo nativamente).
      2. Para cada una, sklearn.inspection.partial_dependence sobre 50 puntos
         del rango observado.
      3. Plotly con UN trace por feature (lineas).

    Tolera fallos: si pipeline no expone el modelo final o sklearn falla,
    devuelve string vacio (la seccion del dashboard se omite).
    """
    try:
        import plotly.graph_objects as go
        from sklearn.inspection import partial_dependence
    except ImportError:
        return ""

    # Resolver el modelo final dentro del wrapper (OOFEnsemble -> Pipeline -> TTR -> regressor)
    try:
        if hasattr(pipeline, "models_") and len(pipeline.models_) > 0:
            inner = pipeline.models_[0]  # primero del ensemble
        else:
            inner = pipeline
        # Top-k features por importance (LGB/XGB lo exponen)
        regressor = inner.named_steps.get("regressor") if hasattr(inner, "named_steps") else inner
        # TransformedTargetRegressor wrapping
        actual = getattr(regressor, "regressor_", regressor)
        importances = getattr(actual, "feature_importances_", None)
        if importances is None or len(importances) == 0:
            return ""
        top_idx = list(np.argsort(importances)[-top_k:][::-1])
    except Exception:
        return ""

    # X_sample debe estar PRE-PROCESADO (mismo space que vio el regressor).
    # Si X_sample es raw, sklearn.inspection.partial_dependence puede fallar.
    # Aqui asumimos que el caller ya pasa X transformado.
    fig = go.Figure()
    palette = ["#1f4e8a", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    for plot_idx, feat_idx in enumerate(top_idx):
        try:
            pdp = partial_dependence(
                actual,
                X_sample,
                features=[feat_idx],
                kind="average",
                grid_resolution=30,
            )
            grid_x = pdp["grid_values"][0]
            avg_y = pdp["average"][0]
        except Exception:
            continue

        feat_label = (
            feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
        )
        fig.add_trace(
            go.Scatter(
                x=grid_x,
                y=avg_y,
                mode="lines",
                name=feat_label,
                line=dict(color=palette[plot_idx % len(palette)], width=2),
                hovertemplate=f"{feat_label}: %{{x:.2f}}<br>Predicción: %{{y:.3f}}<extra></extra>",
            )
        )

    if len(fig.data) == 0:
        return ""

    fig.update_layout(
        title=dict(
            text=f"Top-{top_k} features: efecto marginal sobre la predicción (KG/JR_H)",
            font=dict(size=14, color=_PRIMARY),
        ),
        xaxis_title="Valor de la feature (rango observado)",
        yaxis_title="Predicción promedio",
        height=380,
        showlegend=True,
        legend=dict(
            font=dict(size=10), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    fig.update_yaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    return plotly_div(fig, div_id="pdp_plot")
