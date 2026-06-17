"""Diagnosticos POST-FIT sobre residuos del modelo final (OOF).

Distinto de `eda.py` (que opera sobre data raw, antes de fit). Este modulo
toma `(y_true, y_pred)` del nested CV y produce un reporte HTML con:

    - Durbin-Watson + Ljung-Box sobre residuos (autocorrelacion residual)
    - Breusch-Pagan / White sobre (residuos²) ~ y_pred (heteroscedasticidad)
    - Shapiro / Anderson-Darling / Jarque-Bera sobre residuos (normalidad)
    - Q-Q plot, residuos vs predicho, |residuos| vs predicho (cono?), histograma

Este reporte responde a:
    "El modelo capturo todo el patron disponible o quedaron senales temporales
     o de varianza variable que indican feature engineering faltante?"

Se invoca al final de single_run.train_model y se sube como artifact MLflow
en `residuals/` del run.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.diagnostics.html_renderer import (
    BASE_CSS,
    fig_to_html_div,
    format_pvalue,
    render_test_row,
)
from src.diagnostics.plots import style_fig
from src.diagnostics.statistical_tests import (
    anderson_darling,
    breusch_pagan,
    durbin_watson,
    jarque_bera,
    ljung_box,
    shapiro_wilk,
    white_test,
)


def _residuals_vs_pred_fig(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    res = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=res, mode="markers",
        marker=dict(color="#3b82f6", size=4, opacity=0.5),
        hovertemplate="pred=%{x:.3f}<br>residual=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#dc2626", dash="dash", width=1))
    fig.update_xaxes(title="prediccion (y_pred)")
    fig.update_yaxes(title="residual (y_true - y_pred)")
    return style_fig(fig, title="Residuos vs prediccion (banda horizontal => homocedasticidad)")


def _abs_res_vs_pred_fig(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    res = np.abs(y_true - y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=res, mode="markers",
        marker=dict(color="#7c3aed", size=4, opacity=0.5),
        hovertemplate="pred=%{x:.3f}<br>|residual|=%{y:.3f}<extra></extra>",
    ))
    # Suavizado lowess para visualizar tendencia
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(res, y_pred, frac=0.3, return_sorted=True)
        fig.add_trace(go.Scatter(
            x=smoothed[:, 0], y=smoothed[:, 1],
            mode="lines", line=dict(color="#dc2626", width=2),
            hoverinfo="skip",
        ))
    except Exception:
        pass
    fig.update_xaxes(title="prediccion (y_pred)")
    fig.update_yaxes(title="|residual|")
    return style_fig(fig, title="|Residuos| vs prediccion (cono ascendente => heteroscedasticidad)")


def _residuals_hist_fig(residuals: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40,
        marker_color="#3b82f6", opacity=0.7,
        hovertemplate="bin=%{x:.3f}<br>n=%{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#dc2626", dash="dash"))
    fig.update_xaxes(title="residual")
    fig.update_yaxes(title="frecuencia")
    return style_fig(fig, title="Distribucion de residuos")


def _residuals_qq_fig(residuals: np.ndarray) -> go.Figure:
    from scipy.stats import probplot
    fig = go.Figure()
    if len(residuals) < 10:
        fig.add_annotation(text="n insuficiente", showarrow=False, x=0.5, y=0.5)
        return style_fig(fig, title="Q-Q plot residuos")
    (osm, osr), _ = probplot(residuals, dist="norm")
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(color="#3b82f6", size=4, opacity=0.6),
    ))
    sigma = float(np.std(residuals))
    mu = float(np.mean(residuals))
    lo, hi = float(osm.min()), float(osm.max())
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo * sigma + mu, hi * sigma + mu],
        mode="lines", line=dict(color="#dc2626", dash="dash"),
    ))
    fig.update_xaxes(title="cuantiles teoricos (normal)")
    fig.update_yaxes(title="cuantiles residuos")
    return style_fig(fig, title="Q-Q plot residuos vs normal")


def _run_residual_tests(residuals: np.ndarray, y_pred: np.ndarray) -> dict:
    """Ejecuta la bateria de tests estadisticos sobre los residuos.

    Devuelve un dict con keys: dw, lb, sw, ad, jb, bp, wt.
    Los except defensivos en los tests subyacentes son intencionales: muestras
    pequeñas pueden hacer fallar Shapiro/Anderson/etc.
    """
    res_series = pd.Series(residuals)

    dw = durbin_watson(res_series)
    lb = ljung_box(res_series, lags=10)
    sw = shapiro_wilk(res_series)
    ad = anderson_darling(res_series)
    jb = jarque_bera(res_series)

    # Heteroscedasticidad: regresion residuals² ~ y_pred
    pred_series = pd.Series(y_pred, name="y_pred")
    resid_sq = pd.Series(residuals ** 2, name="resid_sq")
    bp = breusch_pagan(resid_sq, pd.DataFrame({"y_pred": pred_series}))
    wt = white_test(resid_sq, pd.DataFrame({"y_pred": pred_series}))

    return {"dw": dw, "lb": lb, "sw": sw, "ad": ad, "jb": jb, "bp": bp, "wt": wt}


def _build_residual_findings(test_results: dict) -> list[tuple[str, str]]:
    """Interpreta los resultados de los tests y produce findings (severity, msg)."""
    dw = test_results["dw"]
    lb = test_results["lb"]
    bp = test_results["bp"]
    wt = test_results["wt"]
    jb = test_results["jb"]

    summary_findings: list[tuple[str, str]] = []
    if dw.rejects_h0:
        summary_findings.append(("severity-high",
            f"Autocorrelacion residual detectada (DW={dw.statistic:.2f}). "
            "Modelo dejo patron temporal sin capturar — considerar mas lags o STL features."))
    if lb.rejects_h0:
        summary_findings.append(("severity-high",
            f"Ljung-Box rechaza no-autocorrelacion (p={format_pvalue(lb.p_value)})."))
    if bp.rejects_h0 or wt.rejects_h0:
        summary_findings.append(("severity-medium",
            "Heteroscedasticidad residual: varianza no constante con la prediccion. "
            "Considerar log-target o regresion Gamma."))
    if jb.rejects_h0:
        summary_findings.append(("severity-low",
            "Residuos no-normales (skew/kurt). Aceptable para boosted trees, pero los "
            "intervalos asintoticos no son validos — usar conformal prediction."))
    if not summary_findings:
        summary_findings.append(("severity-good",
            "Sin hallazgos diagnosticos sobre residuos: autocorrelacion ausente, "
            "varianza homogenea, distribucion razonablemente normal."))
    return summary_findings


def _render_test_blocks(test_results: dict) -> str:
    """Renderiza los bloques HTML de tabla de tests agrupados por categoria."""
    tests = [
        ("Autocorrelacion temporal", [test_results["dw"], test_results["lb"]]),
        ("Heteroscedasticidad", [test_results["bp"], test_results["wt"]]),
        ("Normalidad", [test_results["sw"], test_results["ad"], test_results["jb"]]),
    ]

    test_blocks = []
    for group_title, tests_in_group in tests:
        rows = "".join(render_test_row(t) for t in tests_in_group)
        test_blocks.append(f"""
        <h3>{escape(group_title)}</h3>
        <table class="summary">
          <thead><tr><th>Test</th><th class='num'>Statistic</th><th class='num'>p-value</th><th>Notas</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """)
    return "".join(test_blocks)


def render_residual_report(
    *,
    variety: str,
    model_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_id: str | None = None,
) -> str:
    """Construye el HTML del residual diagnostic report.

    `run_id`: si se provee, agrega un link al run de MLflow en el header.
    Trazabilidad cuando el HTML se comparte fuera del filesystem (S3, mail).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    # Tests estadisticos
    test_results = _run_residual_tests(residuals, y_pred)

    # Plots
    fig_rvp = _residuals_vs_pred_fig(y_true, y_pred)
    fig_arvp = _abs_res_vs_pred_fig(y_true, y_pred)
    fig_hist = _residuals_hist_fig(residuals)
    fig_qq = _residuals_qq_fig(residuals)

    # Bloques HTML de tests
    test_blocks_html = _render_test_blocks(test_results)

    # Findings / verdict
    summary_findings = _build_residual_findings(test_results)
    findings_html = "".join(
        f'<li class="{cls}">{escape(msg)}</li>'
        for cls, msg in summary_findings
    )

    n = len(residuals)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(residuals))

    # Reusa el tag canonico del proyecto (offline vs CDN segun config).
    from src.utils.html_assets import PLOTLY_JS_TAG as plotly_cdn

    # Run identification para trazabilidad MLflow.
    run_meta = ""
    if run_id:
        run_id_short = run_id[:12]
        run_meta = (
            f' &middot; <a href="http://localhost:5000/#/experiments/0/runs/{escape(run_id)}" '
            f'target="_blank" style="color:#dbeafe;text-decoration:underline;'
            f'font-family:monospace;font-size:11px;" '
            f'title="Abrir en MLflow UI">run {escape(run_id_short)} &#x2197;</a>'
        )

    body = f"""
    <div class="container">
      <header class="hero">
        <h1>Residual diagnostics — {escape(variety)} / {escape(model_type)}</h1>
        <div class="meta">{n:,} OOF predictions · MAE={mae:.4f} · RMSE={rmse:.4f} · bias={bias:+.4f}{run_meta}</div>
      </header>

      <section class="card">
        <h2>Resumen</h2>
        <ul class="findings-list">{findings_html}</ul>
      </section>

      <section class="card">
        <h2>Plots diagnosticos</h2>
        <div class="grid-2">
          {fig_to_html_div(fig_rvp, 'rvp')}
          {fig_to_html_div(fig_arvp, 'arvp')}
          {fig_to_html_div(fig_hist, 'hist')}
          {fig_to_html_div(fig_qq, 'qq')}
        </div>
      </section>

      <section class="card">
        <h2>Tests estadisticos</h2>
        {test_blocks_html}
      </section>

      <footer class="fineprint">
        residual diagnostics · {escape(variety)} / {escape(model_type)} ·
        generado {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
      </footer>
    </div>
    """

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Residual diagnostics — {escape(variety)} / {escape(model_type)}</title>
  {plotly_cdn}
  {BASE_CSS}
</head>
<body>{body}</body>
</html>"""


def write_residual_report(
    *,
    variety: str,
    model_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    run_id: str | None = None,
) -> Path:
    """Renderiza y persiste el HTML. Devuelve out_path.

    Enmascara pares con NaN/inf antes de los tests: con CV temporal
    (expanding window) las filas de los anios warmup no tienen prediccion
    OOF (y_pred=NaN) y romperian statsmodels/plotly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    html = render_residual_report(
        variety=variety, model_type=model_type,
        y_true=y_true[finite], y_pred=y_pred[finite], run_id=run_id,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
