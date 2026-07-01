"""Dashboard Ejecutivo del modelo ganador (un solo HTML por variedad).

Orquestador thin: arma el `WinnerKit` (kit ejecutivo derivado del campeon)
y ensambla las 6 secciones del dashboard en un HTML estandalone:

  1. Hero        : veredicto + descarga Excel.
  2. Context     : ¿Que datos uso?
  3. Mega KPIs   : ¿Que tan preciso? ¿Cuanto explica? ¿Vale la pena?
  4. Guide       : ¿Como leer este reporte? + glosario.
  5. Actions     : ¿Que hacer hoy? (auto-generado).
  6. Technical   : detalle DS (colapsable).

CSS / plotly bundle viven en `styles.py`; helpers de presentacion en
`helpers.py`; secciones en `sections.py` y `technical.py`. El kit
ejecutivo vive en `step_05_evaluate.explainability.build_winner_kit`
para que el Excel ejecutivo lo reuse sin duplicar logica.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd

from src.config import REPORT_PROJECT_NAME, REPORTS_DIR
from src.step_05_evaluate.champion import ModelResult
from src.step_05_evaluate.explainability import build_winner_kit
from src.step_05_evaluate.html.sections import (
    build_actions_section,
    build_backends_comparison_section,
    build_bias_section,
    build_context_section,
    build_diagnostic_links_section,
    build_errors_detail_section,
    build_fundo_formato_section,
    build_guide_section,
    build_hero,
    build_mega_kpis,
    build_pdp_section,
    build_statistical_diagnostic_section,
)
from src.step_05_evaluate.html.styles import DASHBOARD_CSS
from src.step_05_evaluate.html.technical import build_technical_section
from src.utils.html_assets import plotly_js_tag


def render_winner_dashboard(
    *,
    variety: str,
    results: list[ModelResult],
    champion: ModelResult,
    decision: dict,
    output_dir: Path | None = None,
    excel_path: str | None = None,
    X_raw: pd.DataFrame | None = None,
    run_label: str | None = None,
) -> Path:
    """Genera `reports/Winner_{variety}_{run_label}.html` y devuelve la ruta.

    `run_label` es un identificador estable del run usado en el filename
    (recomendado: timestamp `YYYY-MM-DD_HH-MM-SS`, con segundos para evitar
    colisiones cuando dos runs corren en el mismo minuto, ej. smoke tests).
    Acumular un Winner por run permite revisar el historico desde el
    dashboard global. Si `run_label` es None, cae al patron viejo
    `Winner_{variety}.html` (sobrescribe).
    """
    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_label:
        out_path = out_dir / f"Winner_{variety}_{run_label}.html"
    else:
        out_path = out_dir / f"Winner_{variety}.html"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    kit = build_winner_kit(variety=variety, champion=champion, X_raw=X_raw)

    hero = build_hero(
        variety=variety,
        champion=champion,
        verdict=kit.verdict,
        excel_path=excel_path,
        timestamp=ts,
        oof_mape=kit.oof_mape,
        oof_r2=kit.oof_r2,
    )
    context_html = build_context_section(kit.context, champion)
    mega_kpis = build_mega_kpis(
        kit.abs_err,
        kit.real,
        kit.pred,
        kit.oof_mape,
        kit.oof_r2,
    )
    fundo_formato_html = build_fundo_formato_section(
        kit.X_aligned,
        kit.abs_err,
        kit.real,
    )
    guide = build_guide_section()
    backends_compare = build_backends_comparison_section(results, champion)
    stat_diag = build_statistical_diagnostic_section(
        mae_ci=kit.mae_oof_ci,
        mape_ci=kit.mape_oof_ci,
        r2_ci=kit.r2_oof_ci,
        heteroscedasticity=kit.heteroscedasticity,
        calibration_df=kit.calibration,
    )
    bias_html = build_bias_section(kit.fundo_bias)
    errors_detail_html = build_errors_detail_section(
        business_validation=champion.business_validation,
        X_aligned=kit.X_aligned,
        excel_path=excel_path,
    )
    actions_html = build_actions_section(kit.actions)

    # PDP (Partial Dependence Plots): defensivo, requiere cargar pipeline +
    # transformar X_sample. Si cualquier paso falla, la seccion se omite.
    pdp_html = ""
    if X_raw is not None and champion.pipeline_path:
        try:
            import joblib

            ensemble = joblib.load(champion.pipeline_path)
            # Tomamos el primer pipeline del ensemble para preprocessor + features
            inner_pipe = ensemble.models_[0] if hasattr(ensemble, "models_") else ensemble
            preprocessor = inner_pipe.named_steps["preprocessor"]
            X_sample = X_raw.head(min(500, len(X_raw)))
            X_transformed = preprocessor.transform(X_sample)
            # `preprocessor.transform()` puede devolver ndarray (default
            # sklearn) o DataFrame. Preferimos `get_feature_names_out()` y
            # caemos a `.columns` o a nombres sinteticos si no esta disponible.
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except (AttributeError, ValueError):
                try:
                    feature_names = list(X_transformed.columns)
                except AttributeError:
                    feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]
            from src.step_05_evaluate.diagnostics import plot_partial_dependence_plotly

            pdp_html = plot_partial_dependence_plotly(
                ensemble,
                X_transformed,
                feature_names=feature_names,
                top_k=5,
            )
        except Exception:
            pdp_html = ""
    pdp_section = build_pdp_section(pdp_html)
    diagnostic_links = build_diagnostic_links_section(variety, out_dir)
    technical = build_technical_section(
        results=results,
        champion=champion,
        decision=decision,
        X_aligned=kit.X_aligned,
        abs_errors=kit.abs_err,
        real=kit.real,
    )

    # Run identification para trazabilidad: run_id MLflow truncado + link al
    # tracking server. Permite ir del HTML al run de MLflow sin buscar manual.
    run_id = champion.mlflow_run_id or ""
    run_id_short = run_id[:12] if run_id else ""
    mlflow_link = (
        (
            f'<a href="http://localhost:5000/#/experiments/0/runs/{escape(run_id)}" '
            f'style="color:#93c5fd;text-decoration:none;font-family:monospace;" '
            f'target="_blank" title="Abrir en MLflow UI">'
            f"run {escape(run_id_short)} &#x2197;</a>"
        )
        if run_id
        else '<span style="opacity:.4;">run sin id</span>'
    )

    nav_back = f"""
        <nav style="background:#0f172a;color:#e2e8f0;padding:8px 16px;
                    font:13px/1 'Inter',system-ui,sans-serif;
                    display:flex;justify-content:space-between;align-items:center;
                    gap:16px;">
          <a href="./index.html" style="color:#93c5fd;text-decoration:none;">
            &#x21A9; Reports Dashboard
          </a>
          <div style="display:flex;gap:16px;align-items:center;font-size:11px;">
            <span style="opacity:.6;">{escape(variety)} &middot; {escape(champion.model_type.upper())}</span>
            {mlflow_link}
          </div>
        </nav>
    """
    html = f"""<!doctype html>
        <html lang="es"><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Winner · {escape(variety)} — {escape(REPORT_PROJECT_NAME)}</title>
        {plotly_js_tag()}
        <style>{DASHBOARD_CSS}</style></head>
        <body>
        {nav_back}
        <div class="wrap">
        {hero}
        {context_html}
        {mega_kpis}
        {fundo_formato_html}
        {guide}
        {backends_compare}
        {stat_diag}
        {pdp_section}
        {bias_html}
        {errors_detail_html}
        {diagnostic_links}
        {actions_html}
        {technical}
        <footer>Generado automáticamente por el pipeline de entrenamiento ML · {escape(ts)}</footer>
        </div>
        </body></html>
    """

    out_path.write_text(html, encoding="utf-8")
    return out_path
