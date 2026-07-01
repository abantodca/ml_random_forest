"""Ensambla el HTML del reporte EDA.

Estructura:
    1. Hero / Resumen ejecutivo (5 hallazgos top con badges)
    2. Calidad de datos (n, missing patterns, duplicados)
    3. Distribuciones univariadas (1 tarjeta por variable: hist+QQ+box+tests)
    4. Analisis temporal (ACF/PACF + DW/LB + ADF/KPSS + STL strengths)
    5. Multivariado (correlation heatmap + VIF + MI)
    6. Drift entre anios (PSI heatmap)
    7. Recomendaciones automaticas (regla-based)

El HTML es self-contained: plotly.js inline (UNA sola vez), CSS embebido,
sin imagenes externas. Tamaño objetivo < 5MB por variedad.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from html import escape
from pathlib import Path

import plotly.graph_objects as go

#: API publica del modulo. Otros modulos (residuals.py, dashboard_index)
#: importan estos symbols, asi que NO llevan underscore prefix.
BASE_CSS = """
<style>
  :root {
    --primary: #2563eb;
    --success: #16a34a;
    --warning: #f59e0b;
    --danger: #dc2626;
    --gray-50: #f8fafc;
    --gray-100: #f1f5f9;
    --gray-200: #e2e8f0;
    --gray-500: #64748b;
    --gray-700: #334155;
    --gray-900: #0f172a;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    color: var(--gray-900);
    background: var(--gray-50);
    margin: 0; padding: 24px 16px;
    line-height: 1.55;
  }
  .container { max-width: 1280px; margin: 0 auto; }
  header.hero {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: white; padding: 28px 32px; border-radius: 12px;
    margin-bottom: 24px; box-shadow: 0 4px 16px rgba(37,99,235,.15);
  }
  header.hero h1 { margin: 0; font-size: 22px; font-weight: 600; }
  header.hero .meta { font-size: 13px; opacity: .85; margin-top: 6px; }
  section.card {
    background: white; border-radius: 10px; padding: 20px 24px;
    margin-bottom: 18px; box-shadow: 0 1px 3px rgba(15,23,42,.06);
    border: 1px solid var(--gray-200);
  }
  section.card h2 {
    font-size: 16px; margin: 0 0 16px;
    padding-bottom: 10px; border-bottom: 1px solid var(--gray-200);
    color: var(--gray-900);
  }
  section.card h3 {
    font-size: 14px; margin: 16px 0 8px; color: var(--gray-700);
  }
  table.summary { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.summary th, table.summary td {
    text-align: left; padding: 8px 10px;
    border-bottom: 1px solid var(--gray-100);
  }
  table.summary th {
    background: var(--gray-50); font-weight: 600;
    color: var(--gray-700); text-transform: uppercase;
    font-size: 10px; letter-spacing: .5px;
  }
  table.summary td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 500; line-height: 1.4;
  }
  .badge-ok { background: #dcfce7; color: #166534; }
  .badge-warn { background: #fef3c7; color: #92400e; }
  .badge-danger { background: #fee2e2; color: #991b1b; }
  .badge-info { background: #dbeafe; color: #1e40af; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  .var-card {
    border: 1px solid var(--gray-200); border-radius: 8px;
    padding: 14px 16px; margin-bottom: 12px;
    background: var(--gray-50);
  }
  .var-card .var-name { font-weight: 600; font-size: 14px; color: var(--gray-900); }
  .var-card .var-stats {
    display: flex; flex-wrap: wrap; gap: 12px; margin: 6px 0 10px;
    font-size: 11px; color: var(--gray-500);
  }
  .var-card .var-stats span b {
    color: var(--gray-700); font-weight: 500;
  }
  .findings-list {
    list-style: none; padding: 0; margin: 0;
  }
  .findings-list li {
    padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;
    border-left: 3px solid var(--gray-200); background: var(--gray-50);
    font-size: 13px;
  }
  .findings-list li.severity-high { border-left-color: var(--danger); }
  .findings-list li.severity-medium { border-left-color: var(--warning); }
  .findings-list li.severity-low { border-left-color: var(--primary); }
  .findings-list li.severity-good { border-left-color: var(--success); }
  footer.fineprint {
    text-align: center; color: var(--gray-500); font-size: 11px;
    margin-top: 24px; padding-bottom: 8px;
  }
  @media print {
    body { background: white; padding: 0; }
    section.card { box-shadow: none; page-break-inside: avoid; }
    header.hero { background: var(--gray-900); -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  }
</style>
"""


def render_badge(text: str, kind: str = "info") -> str:
    """Span con badge tipado (kind: info/ok/warn/danger)."""
    return f'<span class="badge badge-{kind}">{escape(text)}</span>'


def fig_to_html_div(fig: go.Figure, div_id: str) -> str:
    """Plotly fig -> HTML div sin redundar plotly.js (se carga UNA vez en head)."""
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def format_pvalue(p) -> str:
    """Formatea p-value: notacion cientifica si <0.001, decimal si no."""
    if p is None:
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def render_test_row(test) -> str:
    """Fila <tr> para una tabla de tests estadisticos. Ver TestResult."""
    stat = "—" if test.statistic is None else f"{test.statistic:.4f}"
    return (
        f"<tr><td>{test.status_emoji()} {escape(test.name)}</td>"
        f"<td class='num'>{stat}</td>"
        f"<td class='num'>{format_pvalue(test.p_value)}</td>"
        f"<td>{escape(test.notes)}</td></tr>"
    )


def _eda_history_links(variety: str, current_path: Path | None = None, max_links: int = 5) -> str:
    """Lista los EDAs anteriores de la misma variedad (excluye el actual).

    Util para comparar drift entre EDAs sin abrir el directorio. Devuelve
    HTML inline con timestamps clicables al pie del header. Si no hay
    historicos (o solo el actual), devuelve cadena vacia.
    """
    from src.config import REPORTS_DIR

    if not REPORTS_DIR.exists():
        return ""
    candidates = sorted(
        REPORTS_DIR.glob(f"EDA_{variety}_*.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if current_path is not None:
        candidates = [c for c in candidates if c.name != Path(current_path).name]
    if not candidates:
        return ""
    items = []
    for p in candidates[:max_links]:
        # Extrae timestamp del filename si tiene patron EDA_<v>_<ts>.html
        ts_part = p.stem.replace(f"EDA_{variety}_", "")
        items.append(
            f'<a href="{escape(p.name)}" '
            f'style="color:#dbeafe;text-decoration:underline;font-size:11px;'
            f'margin-right:10px;font-family:monospace;">{escape(ts_part)}</a>'
        )
    return (
        '<div style="margin-top:10px;font-size:11px;opacity:.85;">'
        '<span style="opacity:.7;">EDAs anteriores:</span> ' + "".join(items) + "</div>"
    )


def _hero(variety: str, n_rows: int, n_cols: int, current_path: Path | None = None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    history_html = _eda_history_links(variety, current_path)
    return f"""
    <header class="hero">
      <h1>EDA Diagnostico — {escape(variety)}</h1>
      <div class="meta">{n_rows:,} filas × {n_cols} columnas raw · generado {ts}</div>
      {history_html}
    </header>
    """


def _data_quality_section(quality: dict) -> str:
    rows = [
        f"<tr><td>{escape(k)}</td><td class='num'>{escape(str(v))}</td></tr>"
        for k, v in quality.items()
    ]
    return f"""
    <section class="card">
      <h2>1. Calidad de datos</h2>
      <table class="summary">
        <thead><tr><th>Metrica</th><th class='num'>Valor</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>
    """


def _variable_card(
    profile, fig_hist: go.Figure, fig_qq: go.Figure, fig_box: go.Figure, idx: int
) -> str:
    """Tarjeta por variable: hist + qq + box + tests."""
    stats = (
        f"<span><b>n</b>={profile.n:,}</span>"
        f"<span><b>miss</b>={profile.miss_ratio:.1%}</span>"
        f"<span><b>μ</b>={profile.mean:.3f}</span>"
        f"<span><b>med</b>={profile.median:.3f}</span>"
        f"<span><b>σ</b>={profile.std:.3f}</span>"
        f"<span><b>skew</b>={profile.skew:+.2f}</span>"
        f"<span><b>kurt</b>={profile.kurtosis:+.2f}</span>"
    )
    bc = (
        f"{render_badge('Box-Cox: ' + profile.boxcox_recommendation, 'info')}"
        if profile.boxcox_lambda is not None
        else f"{render_badge('Box-Cox: ' + profile.boxcox_recommendation, 'warn')}"
    )
    outliers = (
        f"<span><b>outliers</b> IQR={profile.n_outliers_iqr} "
        f"· Z={profile.n_outliers_zscore} "
        f"· MAD={profile.n_outliers_mad}</span>"
    )
    test_rows = "".join(render_test_row(t) for t in profile.normality_tests)
    return f"""
    <div class="var-card">
      <div class="var-name">{escape(profile.name)}</div>
      <div class="var-stats">{stats}{outliers}</div>
      <div>{bc}</div>
      <div class="grid-3">
        {fig_to_html_div(fig_hist, f"hist_{idx}")}
        {fig_to_html_div(fig_qq, f"qq_{idx}")}
        {fig_to_html_div(fig_box, f"box_{idx}")}
      </div>
      <h3>Tests de normalidad</h3>
      <table class="summary">
        <thead><tr><th>Test</th><th class='num'>Statistic</th><th class='num'>p-value</th><th>Notas</th></tr></thead>
        <tbody>{test_rows}</tbody>
      </table>
    </div>
    """


def _temporal_card(profile, fig_acf: go.Figure, idx: int) -> str:
    tests = [profile.durbin_watson, profile.ljung_box_10, profile.adf, profile.kpss]
    test_rows = "".join(render_test_row(t) for t in tests)
    stl = ""
    if profile.stl_trend_strength is not None:
        stl = (
            f'<div class="var-stats">'
            f"<span><b>STL trend</b>={profile.stl_trend_strength:.2f}</span>"
            f"<span><b>STL seasonal</b>={profile.stl_seasonal_strength:.2f}</span>"
            f"<span><b>significant lags</b>={profile.significant_lags[:5]}"
            f"{'...' if len(profile.significant_lags) > 5 else ''}</span>"
            f"</div>"
        )
    return f"""
    <div class="var-card">
      <div class="var-name">{escape(profile.name)} — temporal</div>
      {stl}
      {fig_to_html_div(fig_acf, f"acf_{idx}")}
      <table class="summary">
        <thead><tr><th>Test</th><th class='num'>Statistic</th><th class='num'>p-value</th><th>Notas</th></tr></thead>
        <tbody>{test_rows}</tbody>
      </table>
    </div>
    """


def _categorical_section(report) -> str:
    """Seccion de variables categoricas: cardinality, top, target stats, V."""
    if not report or not report.profiles:
        return ""

    cards = []
    for p in report.profiles:
        # Top categorias table
        rows = []
        for tc in p.top_categories:
            tmean = f"{tc.target_mean:.2f}" if tc.target_mean is not None else "—"
            tstd = f"{tc.target_std:.2f}" if tc.target_std is not None else "—"
            rows.append(
                f"<tr><td>{escape(tc.value)}</td>"
                f"<td class='num'>{tc.count:,}</td>"
                f"<td class='num'>{tc.pct:.1%}</td>"
                f"<td class='num'>{tc.cum_pct:.1%}</td>"
                f"<td class='num'>{tmean}</td>"
                f"<td class='num'>{tstd}</td></tr>"
            )
        top_table = (
            f"<table class='summary'>"
            f"<thead><tr>"
            f"<th>Categoria</th><th class='num'>n</th>"
            f"<th class='num'>%</th><th class='num'>cum%</th>"
            f"<th class='num'>target μ</th><th class='num'>target σ</th>"
            f"</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

        # Stats line
        v_str = f"V={p.cramers_v_target:.2f}" if p.cramers_v_target is not None else "V=—"
        chi2_str = (
            f"χ²={p.chi2_statistic:.1f}, p={format_pvalue(p.chi2_p_value)}"
            if p.chi2_statistic is not None
            else "χ²=—"
        )
        stats_line = (
            f"<span><b>n</b>={p.n:,}</span>"
            f"<span><b>miss</b>={p.miss_ratio:.1%}</span>"
            f"<span><b>cardinality</b>={p.cardinality:,}</span>"
            f"<span><b>singletons</b>={p.n_singletons}</span>"
            f"<span><b>cobertura top10</b>={p.coverage_top10_pct:.1%}</span>"
            f"<span><b>{v_str}</b></span>"
            f"<span>{chi2_str}</span>"
        )

        # Recomendacion badge
        rec_kind = (
            "warn"
            if (
                "agrupar" in p.target_encoding_recommendation
                or "drop" in p.target_encoding_recommendation
            )
            else "info"
        )
        rec_badge = render_badge(
            "FE: " + p.target_encoding_recommendation,
            rec_kind,
        )

        cards.append(
            f'<div class="var-card">'
            f'<div class="var-name">{escape(p.name)}</div>'
            f'<div class="var-stats">{stats_line}</div>'
            f"<div>{rec_badge}</div>"
            f"<h3>Top {len(p.top_categories)} categorias (vs target)</h3>"
            f"{top_table}"
            f"</div>"
        )

    # Asociaciones entre categoricas
    assoc_html = ""
    if report.associations:
        rows = "".join(
            f"<tr><td>{escape(a.feature_a)}</td><td>{escape(a.feature_b)}</td>"
            f"<td class='num'>{a.cramers_v:.3f}</td>"
            f"<td class='num'>{format_pvalue(a.chi2_p_value)}</td>"
            f"<td>{render_badge(a.severity, 'danger' if a.severity == 'high' else 'warn' if a.severity == 'watch' else 'ok')}</td></tr>"
            for a in report.associations
        )
        assoc_html = f"""
        <h3>Asociaciones entre categoricas (Cramer's V)</h3>
        <p style="font-size:11px; color:var(--gray-500); margin:0 0 8px;">
          V &lt; 0.10 sin asociacion · 0.10-0.30 debil/moderada · &gt; 0.30 fuerte
          (candidatas a redundancia).
        </p>
        <table class="summary">
          <thead><tr><th>A</th><th>B</th><th class='num'>V</th>
                 <th class='num'>p</th><th>Severidad</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    return f"""
    <section class="card">
      <h2>3-bis. Variables categoricas (frecuencia + asociacion con target)</h2>
      {"".join(cards)}
      {assoc_html}
    </section>
    """


def _findings_section(findings: list[tuple]) -> str:
    """Lista de hallazgos top. `findings` = [(severity, message), ...]"""
    if not findings:
        return ""
    items = "".join(f'<li class="severity-{sev}">{escape(msg)}</li>' for sev, msg in findings)
    return f"""
    <section class="card">
      <h2>Resumen ejecutivo — top hallazgos</h2>
      <ul class="findings-list">{items}</ul>
    </section>
    """


def render_eda_html(
    *,
    variety: str,
    n_rows: int,
    n_cols: int,
    quality_metrics: dict,
    findings: list[tuple],
    var_profiles_with_figs: Iterable[tuple],  # [(profile, hist, qq, box), ...]
    temporal_profiles_with_figs: Iterable[tuple],  # [(profile, acf_fig), ...]
    corr_fig: go.Figure,
    vif_fig: go.Figure,
    mi_fig: go.Figure,
    psi_fig: go.Figure,
    high_corr_pairs: list[tuple],
    categorical_report=None,
    out_path: Path | None = None,
) -> str:
    """Construye el HTML completo y lo devuelve como string."""
    # Var cards
    var_cards = []
    for i, (profile, h, q, b) in enumerate(var_profiles_with_figs):
        var_cards.append(_variable_card(profile, h, q, b, idx=i))

    temporal_cards = []
    for i, (profile, fig) in enumerate(temporal_profiles_with_figs):
        temporal_cards.append(_temporal_card(profile, fig, idx=i))

    high_pairs_table = ""
    if high_corr_pairs:
        rows = "".join(
            f"<tr><td>{escape(a)}</td><td>{escape(b)}</td><td class='num'>{r:+.3f}</td></tr>"
            for a, b, r in high_corr_pairs[:30]
        )
        high_pairs_table = f"""
        <h3>Pares con |corr| ≥ 0.85</h3>
        <table class="summary">
          <thead><tr><th>Variable A</th><th>Variable B</th><th class='num'>r</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    # Reusa el tag canonico (offline vs CDN segun config) para tamaño consistente.
    from src.utils.html_assets import plotly_js_tag

    plotly_cdn = plotly_js_tag()

    body = f"""
    <div class="container">
      {_hero(variety, n_rows, n_cols, current_path=out_path)}
      {_findings_section(findings)}
      {_data_quality_section(quality_metrics)}

      <section class="card">
        <h2>2. Distribuciones univariadas</h2>
        {"".join(var_cards) or "<p>sin variables numericas</p>"}
      </section>

      <section class="card">
        <h2>3. Analisis temporal (autocorrelacion + estacionariedad + STL)</h2>
        {"".join(temporal_cards) or "<p>sin perfil temporal disponible</p>"}
      </section>

      {_categorical_section(categorical_report)}

      <section class="card">
        <h2>4. Multivariado</h2>
        <div class="grid-2">
          {fig_to_html_div(corr_fig, "corr_heatmap")}
          {fig_to_html_div(vif_fig, "vif_bars")}
        </div>
        {fig_to_html_div(mi_fig, "mi_bars")}
        {high_pairs_table}
      </section>

      <section class="card">
        <h2>5. Drift entre anios (Population Stability Index)</h2>
        {fig_to_html_div(psi_fig, "psi_heatmap")}
        <p style="font-size:11px; color:var(--gray-500); margin-top:8px;">
          PSI &lt; 0.10 sin drift · 0.10-0.25 moderado · &gt; 0.25 severo.
        </p>
      </section>

      <footer class="fineprint">
        ml_training EDA · variedad {escape(variety)} ·
        generado {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
      </footer>
    </div>
    """

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EDA — {escape(variety)}</title>
  {plotly_cdn}
  {BASE_CSS}
</head>
<body>{body}</body>
</html>"""


def write_eda_html(html: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
