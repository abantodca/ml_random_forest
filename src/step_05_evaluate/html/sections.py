"""Secciones ejecutivas del dashboard (top-down):

  1. Hero        : veredicto + descarga Excel.
  2. Context     : contexto del dataset (filas, fechas, fundos, formatos).
  3. Mega KPIs   : las 3 preguntas que importan en lenguaje simple.
  4. Guide       : ¿Como leer este reporte? + glosario.
  5. Actions     : recomendaciones auto-generadas.

Cada `build_*` recibe primitivos / dataclasses listos (no hace I/O ni
calculos pesados) y devuelve un fragmento HTML escapado.
"""

from __future__ import annotations

import math
from html import escape

import numpy as np
import pandas as pd

from src.config import (
    CHAMPION_MAX_GAP_REL,
    KPI_PRECISION_HIGH_MAPE_PCT,
    KPI_PRECISION_MEDIUM_MAPE_PCT,
    REPORT_BUSINESS_UNIT,
    REPORT_MODEL_DESCRIPTION,
    REPORT_PROJECT_NAME,
)
from src.step_05_evaluate.champion import ModelResult
from src.step_05_evaluate.explainability import (
    Action,
    GroupBias,
    PlainKPI,
    TrainingContext,
    Verdict,
    glossary_terms,
    kpi_explanatory_power,
    kpi_precision,
    kpi_vs_baseline,
)
from src.step_05_evaluate.html.helpers import compute_error_percentiles, download_button


def _hero_kpi_chip(label: str, value: str, status: str, sub: str = "") -> str:
    """Chip KPI del hero con semaforo (ok / warn / bad / neutral)."""
    sub_html = f'<div class="chip-sub">{escape(sub)}</div>' if sub else ""
    return (
        f'<div class="hero-chip {escape(status)}">'
        f'<div class="chip-label">{escape(label)}</div>'
        f'<div class="chip-value">{escape(value)}</div>'
        f"{sub_html}</div>"
    )


def _hero_kpi_strip(
    champion: ModelResult,
    oof_mape: float | None,
    oof_r2: float | None,
) -> str:
    """Strip de KPIs clave en el hero: el numero que gerencia busca, arriba.

    Semaforo con los mismos umbrales que los Mega KPIs
    (KPI_PRECISION_*_MAPE_PCT) y el gate de gap relativo del campeon
    (CHAMPION_MAX_GAP_REL) para que hero y detalle nunca se contradigan.
    """
    chips = []
    if oof_mape is not None and math.isfinite(oof_mape):
        status = (
            "ok"
            if oof_mape <= KPI_PRECISION_HIGH_MAPE_PCT
            else "warn"
            if oof_mape <= KPI_PRECISION_MEDIUM_MAPE_PCT
            else "bad"
        )
        chips.append(
            _hero_kpi_chip(
                "Error de predicción",
                f"{oof_mape:.1f}%",
                status,
                "MAPE validado (OOF)",
            )
        )
    if oof_r2 is not None and math.isfinite(oof_r2):
        status = "ok" if oof_r2 >= 0.75 else "warn" if oof_r2 >= 0.5 else "bad"
        chips.append(
            _hero_kpi_chip(
                "Variabilidad explicada",
                f"{oof_r2 * 100:.0f}%",
                status,
                "R² (OOF)",
            )
        )
    gap_rel = champion.gap_rel
    if math.isfinite(gap_rel):
        passed = gap_rel <= CHAMPION_MAX_GAP_REL
        chips.append(
            _hero_kpi_chip(
                "Control de sobreajuste",
                f"{'✓' if passed else '✗'} {gap_rel:.2f}",
                "ok" if passed else "bad",
                f"gap relativo · límite {CHAMPION_MAX_GAP_REL:.2f}",
            )
        )
    if not chips:
        return ""
    return f'<div class="hero-chips">{"".join(chips)}</div>'


def build_hero(
    *,
    variety: str,
    champion: ModelResult,
    verdict: Verdict,
    excel_path: str | None,
    timestamp: str,
    oof_mape: float | None = None,
    oof_r2: float | None = None,
) -> str:
    """Hero ejecutivo del dashboard."""
    model_label = champion.model_type.upper()
    kpi_strip = _hero_kpi_strip(champion, oof_mape, oof_r2)
    return f"""
    <div class="hero {verdict.color_key}">
      <div class="hero-text">
        <div class="eyebrow">Dashboard ejecutivo · Modelo de productividad</div>
        <h1>{escape(REPORT_PROJECT_NAME)}</h1>
        <div class="meta">
          Variedad <b>{escape(variety)}</b> · Modelo <b>{escape(model_label)}</b>
          · {escape(REPORT_BUSINESS_UNIT)} · {escape(timestamp)}
        </div>
        <div class="verdict-badge">
          <span class="icon">{verdict.icon}</span>
          <span>{escape(verdict.title)}</span>
        </div>
        <div class="verdict-headline">{escape(verdict.headline)}</div>
        <div class="verdict-body">{escape(verdict.body)}</div>
        {kpi_strip}
      </div>
      <div class="hero-side">
        {download_button(excel_path, variety)}
      </div>
    </div>
    """


def build_context_section(
    ctx: TrainingContext,
    champion: ModelResult,
) -> str:
    date_range = "—"
    if ctx.date_min and ctx.date_max:
        date_range = f"{ctx.date_min} a {ctx.date_max}"

    fundos_str = ", ".join(ctx.fundos_top) if ctx.fundos_top else "—"
    formatos_str = ", ".join(ctx.formatos_top[:3]) if ctx.formatos_top else "—"
    if ctx.n_formatos > 3:
        formatos_str += f", +{ctx.n_formatos - 3} más"

    model_label = champion.model_type.upper()
    model_sub = f"de {champion.elapsed_seconds:.0f}s entrenamiento"

    cards = "".join(
        [
            f'<div class="ctx-card"><div class="label">Cosechas analizadas</div>'
            f'<div class="value">{ctx.n_rows:,}</div>'
            f'<div class="sub">filas del histórico</div></div>',
            f'<div class="ctx-card"><div class="label">Período cubierto</div>'
            f'<div class="value">{escape(date_range)}</div>'
            f'<div class="sub">rango de fechas</div></div>',
            f'<div class="ctx-card"><div class="label">Fundos</div>'
            f'<div class="value">{ctx.n_fundos}</div>'
            f'<div class="sub">{escape(fundos_str)}</div></div>',
            f'<div class="ctx-card"><div class="label">Formatos</div>'
            f'<div class="value">{ctx.n_formatos}</div>'
            f'<div class="sub">{escape(formatos_str)}</div></div>',
            f'<div class="ctx-card"><div class="label">Modelo elegido</div>'
            f'<div class="value">{escape(model_label)}</div>'
            f'<div class="sub">{escape(model_sub)}</div></div>',
        ]
    )
    return f"""
    <section>
      <div class="eyebrow">Contexto del entrenamiento</div>
      <h2>¿Qué datos usó este modelo?</h2>
      <p class="lead">{escape(REPORT_MODEL_DESCRIPTION)}</p>
      <div class="ctx-grid">{cards}</div>
    </section>
    """


def _kpi_mega_card(kpi: PlainKPI, icon: str) -> str:
    score_pill = ""
    if kpi.score_label not in ("—", ""):
        score_pill = (
            f'<span class="score-pill {escape(kpi.score_label)}">{escape(kpi.score_label)}</span>'
        )
    return f"""
    <div class="kpi-mega">
      <div class="question">
        <span>{icon}</span>
        <span>{escape(kpi.question)}</span>
        {score_pill}
      </div>
      <div class="headline">{escape(kpi.headline)}</div>
      <div class="detail">{escape(kpi.detail)}</div>
      <div class="technical">{escape(kpi.technical)}</div>
    </div>
    """


def build_mega_kpis(
    abs_errors: np.ndarray,
    real: np.ndarray,
    pred: np.ndarray,
    full_mape: float,
    full_r2: float | None,
) -> str:
    k1 = kpi_precision(abs_errors, full_mape)
    k2 = kpi_explanatory_power(full_r2)
    k3 = kpi_vs_baseline(real, pred)
    cards = _kpi_mega_card(k1, "🎯") + _kpi_mega_card(k2, "📊") + _kpi_mega_card(k3, "📈")
    return f"""
    <section>
      <div class="eyebrow">Las preguntas que importan</div>
      <h2>¿Qué tan bueno es este modelo?</h2>
      <p class="lead">Tres respuestas en lenguaje simple. Si tienes 30 segundos para entender el modelo, lee esto.</p>
      <div class="kpi-mega-grid">{cards}</div>
    </section>
    """


def build_guide_section() -> str:
    rows = "".join(
        f'<div class="gloss-row"><div class="term">{escape(t)}</div>'
        f'<div class="def">{escape(d)}</div></div>'
        for t, d in glossary_terms()
    )
    return f"""
    <details class="help">
      <summary>📖 ¿Cómo leer este reporte?</summary>
      <div class="body">
        <p>Este dashboard responde 3 preguntas sobre un modelo de predicción
        de productividad por jornal. Está organizado de lo más simple a lo
        más detallado:</p>
        <ol>
          <li><b>Veredicto ejecutivo (arriba)</b>: una respuesta clara — ¿usar este modelo o no?</li>
          <li><b>Contexto</b>: con qué datos se entrenó.</li>
          <li><b>Las 3 preguntas que importan</b>: precisión, capacidad explicativa, valor vs no usar modelo.</li>
          <li><b>Acciones recomendadas</b>: qué hacer hoy con esta información.</li>
          <li><b>Detalle técnico</b> (colapsado): para el equipo de Data Science — comparación de modelos, gráficos de error, distribución por subgrupo.</li>
        </ol>
        <p style="margin-top:14px"><b>Glosario de términos técnicos:</b></p>
        <div class="glossary-grid">{rows}</div>
      </div>
    </details>
    """


def build_bias_section(fundo_bias: list[GroupBias]) -> str:
    """Sesgo direccional por FUNDO. Vacio si no hay sesgos significativos.

    Diferencia clave vs `actions`: aqui el problema NO es el tamano del
    error, es la DIRECCION (sobreestima vs subestima). Un fundo con MAPE
    razonable puede igual tener un sesgo del +8% sostenido que pasa
    invisible en el filtro por magnitud y le cuesta dinero al negocio.
    """
    if not fundo_bias:
        return ""
    rows = "".join(
        f"<tr>"
        f"<td>{escape(b.group_value)}</td>"
        f"<td>{b.n}</td>"
        f'<td class="{"sub" if b.direction == "subestima" else "sobre"}">'
        f"{b.direction.upper()} {abs(b.bias_pct_of_real_mean):.1f}%"
        f"</td>"
        f"<td>{b.mean_signed_bias:+.2f} kg/jornal</td>"
        f"</tr>"
        for b in fundo_bias
    )
    return f"""
    <section>
      <div class="eyebrow">Diagnostico estructural</div>
      <h2>Sesgo direccional por FUNDO</h2>
      <p class="lead">
        Fundos donde el modelo se desvia consistentemente hacia un lado
        (sobre o subestima). El error puede estar en rango aceptable, pero
        la direccion es sistematica: revisar la causa antes de automatizar
        decisiones operativas en estos fundos.
      </p>
      <table class="bias-table">
        <thead><tr>
          <th>FUNDO</th><th>n</th><th>Sesgo</th><th>Diferencia promedio</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
    """


def _oof_mape_of(r: ModelResult) -> float:
    """MAPE OOF de negocio del modelo (la metrica que decide), NaN si falta."""
    if r.business_metrics_oof:
        v = r.business_metrics_oof.get("mape")
        if v is not None and math.isfinite(float(v)):
            return float(v)
    return float("nan")


_FF_MIN_N = 5


def _group_error_rows(
    groups: pd.Series,
    abs_err: np.ndarray,
    real: np.ndarray,
) -> list[dict]:
    """[{label, n, mape, ratio}] por categoria, ordenado por MAPE desc."""
    nz_all = real != 0
    mape_g = (
        float(np.mean(abs_err[nz_all] / np.abs(real[nz_all])) * 100)
        if nz_all.any()
        else float("nan")
    )
    rows: list[dict] = []
    for cat in groups.dropna().unique():
        if str(cat) in ("", "nan", "<NA>", "None"):
            continue
        mask = (groups == cat).to_numpy()
        if mask.sum() < _FF_MIN_N:
            continue
        nz = mask & nz_all
        if not nz.any():
            continue
        mape = float(np.mean(abs_err[nz] / np.abs(real[nz])) * 100)
        ratio = mape / mape_g if math.isfinite(mape_g) and mape_g > 0 else float("nan")
        rows.append({"label": str(cat), "n": int(mask.sum()), "mape": mape, "ratio": ratio})
    rows.sort(key=lambda r: r["mape"], reverse=True)
    return rows


def _group_error_table(title: str, rows: list[dict]) -> str:
    if not rows:
        return ""

    def pill(ratio: float) -> str:
        if not math.isfinite(ratio):
            return '<span class="ratio-pill neutral">—</span>'
        if ratio < 0.85:
            return f'<span class="ratio-pill good">{(ratio - 1) * 100:+.0f}%</span>'
        if ratio <= 1.15:
            return f'<span class="ratio-pill neutral">{(ratio - 1) * 100:+.0f}%</span>'
        if ratio <= 1.5:
            return f'<span class="ratio-pill warn">{(ratio - 1) * 100:+.0f}%</span>'
        return f'<span class="ratio-pill bad">×{ratio:.1f}</span>'

    body = "".join(
        f'<tr class="subgrp-row">'
        f'<td><span class="grp-name">{escape(r["label"])}</span></td>'
        f'<td class="num">{r["n"]:,}</td>'
        f'<td class="num">{r["mape"]:.1f}%</td>'
        f'<td class="num">{pill(r["ratio"])}</td>'
        f"</tr>"
        for r in rows
    )
    return f"""
    <div class="subgrp-table-wrap">
      <div class="subgrp-table-title">{escape(title)}
        <span class="subgrp-table-hint">ordenado por error · vs promedio global</span>
      </div>
      <table class="subgrp-table">
        <thead><tr><th>Grupo</th><th class="num">n</th>
        <th class="num">MAPE</th><th class="num">vs global</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    """


def build_fundo_formato_section(
    X_aligned: pd.DataFrame | None,
    abs_err: np.ndarray,
    real: np.ndarray,
) -> str:
    """Seccion ejecutiva 'Error por Fundo y Formato'.

    FUNDO y FORMATO son las dimensiones con mayor heterogeneidad de error
    medida (dolor de dominio: no todos los fundos/formatos existen en todas
    las variedades y los grupos jovenes parten en frio). Esta vista los
    saca del detalle tecnico colapsado y los pone a la vista de gerencia
    con el ratio vs el promedio global.
    """
    if (
        X_aligned is None
        or not hasattr(X_aligned, "columns")
        or len(X_aligned) != abs_err.size
        or real.size != abs_err.size
        or abs_err.size == 0
    ):
        return ""

    tables = []
    if "FUNDO" in X_aligned.columns:
        rows = _group_error_rows(
            X_aligned["FUNDO"].astype(str).reset_index(drop=True),
            abs_err,
            real,
        )
        tables.append(_group_error_table("Error por FUNDO", rows))
    if "FORMATO" in X_aligned.columns:
        rows = _group_error_rows(
            X_aligned["FORMATO"].astype(str).reset_index(drop=True),
            abs_err,
            real,
        )
        tables.append(_group_error_table("Error por FORMATO", rows))
    tables = [t for t in tables if t]
    if not tables:
        return ""

    return f"""
    <section>
      <div class="eyebrow">¿Dónde confiar más y dónde menos?</div>
      <h2>Error por Fundo y Formato</h2>
      <p class="lead">
        El error promedio global esconde diferencias grandes entre fundos y
        formatos: los grupos con poca historia (fundos jóvenes, formatos
        nuevos) son sistemáticamente más difíciles de predecir. Usa esta
        tabla para calibrar cuánta confianza darle al pronóstico según el
        grupo — los rojos requieren revisión manual.
      </p>
      <div class="ff-grid">{"".join(tables)}</div>
    </section>
    """


def _backend_row_html(r: ModelResult, champion: ModelResult) -> str:
    """Renderiza una fila <tr> de la tabla comparativa de backends."""
    is_champ = r.model_type == champion.model_type
    gap_rel = r.gap_rel
    gate_ok = math.isfinite(gap_rel) and gap_rel <= CHAMPION_MAX_GAP_REL
    gate_pill = (
        '<span class="ratio-pill good">✓ pasa</span>'
        if gate_ok
        else '<span class="ratio-pill bad">✗ falla</span>'
    )
    oof_mape = _oof_mape_of(r)
    champ_mape = _oof_mape_of(champion)
    delta_mape = oof_mape - champ_mape
    delta_time = r.elapsed_seconds - champion.elapsed_seconds
    crown = "👑 " if is_champ else ""
    klass = "champ-row" if is_champ else ""
    gap_rel_str = f"{gap_rel:.2f}" if math.isfinite(gap_rel) else "—"
    oof_mape_str = f"{oof_mape:.2f}%" if math.isfinite(oof_mape) else "—"
    delta_mape_str = (
        "—" if is_champ else (f"{delta_mape:+.2f} pp" if math.isfinite(delta_mape) else "—")
    )
    delta_time_str = "—" if is_champ else f"{delta_time:+.0f}s"
    return (
        f'<tr class="{klass}">'
        f"<td>{crown}{escape(r.model_type.upper())}</td>"
        f"<td>{gap_rel_str} {gate_pill}</td>"
        f"<td>{oof_mape_str}</td>"
        f"<td>{delta_mape_str}</td>"
        f"<td>{r.elapsed_seconds:.0f}s</td>"
        f"<td>{delta_time_str}</td>"
        f"</tr>"
    )


def _temporal_check_note(champion: ModelResult) -> str:
    """Nota de transparencia: chequeo de honestidad con CV temporal por año.

    Solo se muestra si el run lo calculo (DUAL_CV_REPORT). El MAPE temporal
    responde otra pregunta (¿forecast de un año NUNCA visto?) y suele ser
    bastante peor que el OOF estratificado (interpolacion dentro de años
    conocidos): publicar ambos evita que el numero bueno se lea como el
    escenario pesimista.
    """
    t_mape = champion.metrics.get("temporal_mape_oof")
    if t_mape is None or not math.isfinite(float(t_mape)):
        return ""
    t_r2 = champion.metrics.get("temporal_r2_oof")
    r2_str = f" · R² temporal: {float(t_r2):.2f}" if t_r2 is not None else ""
    return (
        f'<p class="lead" style="margin-top:12px;font-size:12.5px;">'
        f"<b>Chequeo de honestidad temporal:</b> si el modelo tuviera que "
        f"pronosticar un año completo nunca visto, el error subiría a "
        f"<b>{float(t_mape):.1f}%</b>{r2_str}. El MAPE principal mide el uso "
        f"real (operar dentro de la temporada con historia disponible).</p>"
    )


def build_backends_comparison_section(
    results: list[ModelResult],
    champion: ModelResult,
) -> str:
    """Tabla comparativa de los backends que compitieron.

    Solo se renderiza si len(results) >= 2. Refleja el criterio REAL de
    `select_champion` (gate de gap relativo → MAPE OOF → tiempo); la fila
    del campeon va resaltada y las otras muestran deltas vs campeon.
    """
    if len(results) < 2:
        return ""
    rows_html = "".join(
        _backend_row_html(r, champion)
        for r in sorted(
            results,
            key=lambda x: _oof_mape_of(x) if math.isfinite(_oof_mape_of(x)) else float("inf"),
        )
    )
    return f"""
    <section>
      <div class="eyebrow">Comparativo de modelos</div>
      <h2>¿Por qué ganó este modelo?</h2>
      <p class="lead">
        El campeón se elige en orden estricto: <b>(1)</b> pasar el control de
        sobreajuste (gap relativo Train→Test ≤ {CHAMPION_MAX_GAP_REL:.2f}),
        <b>(2)</b> menor error validado (MAPE OOF: cada cosecha predicha por
        un modelo que no la vio), <b>(3)</b> a igual error (±0.5 pp), menor
        tiempo de entrenamiento. Cada candidato entrenó su propio estudio
        Optuna sobre la misma data y CV.
      </p>
      <table class="backends-table">
        <thead><tr>
          <th>Modelo</th>
          <th>Gap relativo (gate ≤ {CHAMPION_MAX_GAP_REL:.2f})</th>
          <th>MAPE OOF</th><th>Δ vs campeón</th>
          <th>Tiempo</th><th>Δ vs campeón</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
      {_temporal_check_note(champion)}
    </section>
    """


def _ci_table_html(mae_ci, mape_ci, r2_ci) -> str:
    """Tabla de intervalos de confianza bootstrap. Vacio si todas las CI son None."""
    ci_rows = ""
    if mae_ci is not None:
        ci_rows += (
            f"<tr><td>Error absoluto medio (KG/JR)</td>"
            f"<td>{mae_ci.point:.4f}</td>"
            f"<td>[{mae_ci.ci_low:.4f}, {mae_ci.ci_high:.4f}]</td></tr>"
        )
    if mape_ci is not None:
        ci_rows += (
            f"<tr><td>Error porcentual medio (%)</td>"
            f"<td>{mape_ci.point:.2f}</td>"
            f"<td>[{mape_ci.ci_low:.2f}, {mape_ci.ci_high:.2f}]</td></tr>"
        )
    if r2_ci is not None:
        ci_rows += (
            f"<tr><td>R² (variabilidad explicada)</td>"
            f"<td>{r2_ci.point:.4f}</td>"
            f"<td>[{r2_ci.ci_low:.4f}, {r2_ci.ci_high:.4f}]</td></tr>"
        )
    if not ci_rows:
        return ""
    return (
        f"<table class='backends-table'>"
        f"<thead><tr><th>Métrica</th><th>Valor</th><th>IC 95%</th></tr></thead>"
        f"<tbody>{ci_rows}</tbody></table>"
    )


def _heteroscedasticity_block_html(heteroscedasticity) -> str:
    """Bloque visual para el test de heteroscedasticidad.

    Vacio si el objeto es None o si su p_value es NaN (test no aplicable).
    """
    if heteroscedasticity is None or math.isnan(heteroscedasticity.p_value):
        return ""
    klass = "action warning" if heteroscedasticity.is_heteroscedastic else "action info"
    icon = "⚠" if heteroscedasticity.is_heteroscedastic else "✅"
    title = (
        "Variabilidad del error NO uniforme"
        if heteroscedasticity.is_heteroscedastic
        else "Variabilidad del error uniforme"
    )
    return (
        f'<div class="{klass}" style="margin-top:12px;">'
        f'<div class="icon">{icon}</div>'
        f'<div class="body-wrap">'
        f'<div class="title">{escape(title)}</div>'
        f'<div class="body">{escape(heteroscedasticity.note)}</div>'
        f"</div></div>"
    )


def _calibration_block_html(calibration_df) -> str:
    """Renderiza el plot de calibracion (Plotly). Vacio si no aplica o falla."""
    if calibration_df is None or len(calibration_df) == 0:
        return ""
    try:
        from src.step_05_evaluate.diagnostics import plot_calibration_plotly

        return plot_calibration_plotly(calibration_df)
    except Exception:
        return ""


def build_statistical_diagnostic_section(
    *,
    mae_ci,
    mape_ci,
    r2_ci,
    heteroscedasticity,
    calibration_df,
) -> str:
    """Diagnostico estadistico riguroso: IC bootstrap + heteroscedasticity + calibration.

    Solo se renderiza si hay al menos UNA pieza disponible (todas pueden venir
    None si la muestra fue insuficiente).
    """
    if all(x is None for x in [mae_ci, mape_ci, r2_ci, heteroscedasticity, calibration_df]):
        return ""

    ci_block = _ci_table_html(mae_ci, mape_ci, r2_ci)
    hetero_block = _heteroscedasticity_block_html(heteroscedasticity)
    calib_block = _calibration_block_html(calibration_df)

    return f"""
    <section>
      <div class="eyebrow">Diagnóstico estadístico</div>
      <h2>¿Qué tan confiables son estos números?</h2>
      <p class="lead">
        Tres respuestas con sustento estadístico: (1) cuánto puede variar
        cada métrica si el dataset hubiera sido ligeramente distinto
        (intervalos de confianza al 95% via bootstrap), (2) si el modelo
        falla más en algunos rangos que en otros (heteroscedasticidad),
        (3) si las predicciones promedio coinciden con los reales por bin
        (calibración).
      </p>
      {ci_block}
      {hetero_block}
      {calib_block}
    </section>
    """


def build_pdp_section(pdp_html: str) -> str:
    """Renderiza el PDP (Partial Dependence Plot) si fue generado.

    `pdp_html` ya viene como `<div>` plotly desde
    `diagnostics.plot_partial_dependence_plotly`. Si llega vacio (modelo no
    expone feature_importances_, sklearn fallo, etc.), la seccion se omite.
    """
    if not pdp_html:
        return ""
    return f"""
    <section>
      <div class="eyebrow">¿Qué mueve la predicción?</div>
      <h2>Efecto marginal de las features clave</h2>
      <p class="lead">
        Para cada una de las features más importantes, este gráfico muestra
        cómo cambia la predicción promedio cuando esa feature varía,
        manteniendo el resto constante. Pendiente positiva = la feature
        eleva el pronóstico al subir; pendiente plana = el modelo no
        depende fuerte de ella en ese rango.
      </p>
      {pdp_html}
    </section>
    """


def _compute_oof_error_arrays(business_validation):
    """Extrae y prepara los arrays OOF (real, pred, abs_err, residuals, pct_err).

    Devuelve None si no hay datos validos.
    """
    if business_validation is None or business_validation.is_empty():
        return None
    real = getattr(business_validation, "kg_jr_real_oof", None)
    pred = getattr(business_validation, "kg_jr_pred_oof", None)
    if real is None or pred is None or len(real) == 0:
        return None

    real_arr = np.asarray(real, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    abs_err = np.abs(real_arr - pred_arr)
    residuals = real_arr - pred_arr
    nz = real_arr != 0
    pct_err = np.where(nz, abs_err / np.abs(real_arr) * 100.0, np.nan)
    return real_arr, pred_arr, abs_err, residuals, pct_err, nz


def _compute_oof_kpis(abs_err: np.ndarray, pct_err: np.ndarray, nz: np.ndarray) -> dict:
    """Calcula KPIs globales OOF: n, mae, mape, p50, p90, p99."""
    pcts = compute_error_percentiles(abs_err)
    return {
        "n": abs_err.size,
        "mae": float(abs_err.mean()),
        "mape": float(np.nanmean(pct_err)) if np.any(nz) else float("nan"),
        "p50": pcts["p50"],
        "p90": pcts["p90"],
        "p99": pcts["p99"],
    }


def _metrics_cards_html(kpis: dict) -> str:
    """Renderiza la grilla de KPI cards OOF."""
    n = kpis["n"]
    mae = kpis["mae"]
    mape = kpis["mape"]
    p50 = kpis["p50"]
    p90 = kpis["p90"]
    p99 = kpis["p99"]
    return (
        '<div class="ctx-grid">'
        f'<div class="ctx-card"><div class="label">N filas evaluadas (OOF)</div>'
        f'<div class="value">{n:,}</div>'
        f'<div class="sub">cada predicción de un modelo que NO la vio en train</div></div>'
        f'<div class="ctx-card"><div class="label">MAE OOF</div>'
        f'<div class="value">{mae:.2f}</div>'
        f'<div class="sub">kg/jornal · error promedio</div></div>'
        f'<div class="ctx-card"><div class="label">MAPE OOF</div>'
        f'<div class="value">{mape:.1f}%</div>'
        f'<div class="sub">error porcentual promedio</div></div>'
        f'<div class="ctx-card"><div class="label">Mediana (p50)</div>'
        f'<div class="value">{p50:.2f}</div>'
        f'<div class="sub">kg/jornal · 50% del error está debajo</div></div>'
        f'<div class="ctx-card"><div class="label">p90 (severos)</div>'
        f'<div class="value">{p90:.2f}</div>'
        f'<div class="sub">kg/jornal · 10% peor</div></div>'
        f'<div class="ctx-card"><div class="label">p99 (cola)</div>'
        f'<div class="value">{p99:.2f}</div>'
        f'<div class="sub">kg/jornal · casos extremos</div></div>'
        "</div>"
    )


def _extract_row_context(X_aligned, i: int, has_X: bool) -> tuple:
    """Devuelve (fecha, fundo, formato) para la fila i de X_aligned.

    Si X_aligned no esta alineado o falta una columna, retorna "—".
    """
    fecha = "—"
    fundo = "—"
    formato = "—"
    if has_X:
        try:
            row = X_aligned.iloc[i]
            if "FECHA" in row.index and pd.notna(row["FECHA"]):
                fecha = pd.to_datetime(row["FECHA"]).strftime("%Y-%m-%d")
            if "FUNDO" in row.index and pd.notna(row["FUNDO"]):
                fundo = str(row["FUNDO"])
            if "FORMATO" in row.index and pd.notna(row["FORMATO"]):
                formato = str(row["FORMATO"])
        except Exception:
            pass
    return fecha, fundo, formato


def _compute_top_worst_errors(abs_err: np.ndarray, top_n: int) -> np.ndarray:
    """Indices de las top_n filas con mayor error absoluto (desc)."""
    return np.argsort(abs_err)[::-1][:top_n]


def _top_errors_table_html(
    order: np.ndarray,
    real_arr: np.ndarray,
    pred_arr: np.ndarray,
    abs_err: np.ndarray,
    pct_err: np.ndarray,
    X_aligned,
    has_X: bool,
) -> str:
    """Tabla de las top-N peores predicciones."""
    rows_html = ""
    for rank, i in enumerate(order, start=1):
        i = int(i)
        fecha, fundo, formato = _extract_row_context(X_aligned, i, has_X)
        signo = "↑ sobreestima" if pred_arr[i] > real_arr[i] else "↓ subestima"
        rows_html += (
            f"<tr>"
            f'<td class="num">{rank}</td>'
            f"<td>{escape(fecha)}</td>"
            f"<td>{escape(fundo)}</td>"
            f"<td>{escape(formato)}</td>"
            f'<td class="num">{real_arr[i]:.2f}</td>'
            f'<td class="num">{pred_arr[i]:.2f}</td>'
            f'<td class="num">{abs_err[i]:.2f}</td>'
            f'<td class="num">{pct_err[i]:.1f}%</td>'
            f"<td>{signo}</td>"
            f"</tr>"
        )
    return f"""
    <table class="backends-table" style="margin-top:8px">
      <thead><tr>
        <th>#</th><th>Fecha</th><th>FUNDO</th><th>FORMATO</th>
        <th class="num">Real (kg/jr)</th>
        <th class="num">Predicho (kg/jr)</th>
        <th class="num">Error abs.</th>
        <th class="num">Error %</th>
        <th>Dirección</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """


def _error_plots_html(
    abs_err: np.ndarray,
    pred_arr: np.ndarray,
    residuals: np.ndarray,
    p50: float,
    p90: float,
    p99: float,
    X_aligned,
    has_X: bool,
) -> tuple:
    """Genera los plots Plotly: histograma, residuos vs predicho, serie temporal.

    Devuelve (hist_html, resid_html, time_html). Cualquiera puede ser "" si falla.
    """
    from src.step_05_evaluate.diagnostics import (
        plot_error_histogram_plotly,
        plot_error_over_time_plotly,
        plot_residuals_vs_predicted_plotly,
    )

    hist_html = plot_error_histogram_plotly(abs_err, p50, p90, p99)
    resid_html = plot_residuals_vs_predicted_plotly(pred_arr, residuals)

    time_html = ""
    if has_X and "FECHA" in X_aligned.columns:
        try:
            time_html = plot_error_over_time_plotly(
                X_aligned["FECHA"].to_numpy(),
                abs_err,
            )
        except Exception:
            time_html = ""
    return hist_html, resid_html, time_html


def _excel_link_html(excel_path: str | None, n: int) -> str:
    """Genera el link al Excel adjunto (vacio si no hay path)."""
    if not excel_path:
        return ""
    from pathlib import Path as _P

    fname = _P(excel_path).name
    return (
        f'<p style="margin-top:8px;font-size:12px;color:#475569">'
        f"Para inspeccionar las {n:,} filas con sus errores fila por fila, "
        f"abri la hoja <b>Predicciones_OOF</b> en el Excel adjunto: "
        f'<a href="{escape(fname)}" download style="color:#1f4e8a;'
        f'text-decoration:none;border-bottom:1px dashed #1f4e8a">'
        f"{escape(fname)}</a>.</p>"
    )


def build_errors_detail_section(
    *,
    business_validation,
    X_aligned,
    excel_path: str | None = None,
    top_n: int = 20,
) -> str:
    """Sección 'Errores detallados': MAE/MAPE OOF totales + top N peores +
    histograma + residuos + serie temporal.

    Sustenta visualmente "donde se equivoca el modelo" sin tener que abrir
    el Excel. La hoja 'Predicciones_OOF' del Excel sigue siendo la fuente
    autoritativa con todas las filas (filtrable, ordenable).
    """
    arrays = _compute_oof_error_arrays(business_validation)
    if arrays is None:
        return ""
    real_arr, pred_arr, abs_err, residuals, pct_err, nz = arrays

    kpis = _compute_oof_kpis(abs_err, pct_err, nz)
    n = kpis["n"]
    p50, p90, p99 = kpis["p50"], kpis["p90"], kpis["p99"]

    metrics_cards = _metrics_cards_html(kpis)

    has_X = X_aligned is not None and hasattr(X_aligned, "iloc") and len(X_aligned) == n

    order = _compute_top_worst_errors(abs_err, top_n)
    table_html = _top_errors_table_html(
        order,
        real_arr,
        pred_arr,
        abs_err,
        pct_err,
        X_aligned,
        has_X,
    )

    hist_html, resid_html, time_html = _error_plots_html(
        abs_err,
        pred_arr,
        residuals,
        p50,
        p90,
        p99,
        X_aligned,
        has_X,
    )

    excel_link = _excel_link_html(excel_path, n)

    return f"""
    <section>
      <div class="eyebrow">Errores detallados · OOF sobre toda la data</div>
      <h2>¿Dónde se equivoca el modelo?</h2>
      <p class="lead">
        Estas son las métricas de error con TODAS las {n:,} cosechas evaluadas
        out-of-fold (cada una predicha por un modelo que NO la vio en train).
        Es la perspectiva honesta del rendimiento esperado en producción.
      </p>
      {metrics_cards}

      <h3 style="margin-top:24px">Top {top_n} peores predicciones</h3>
      <p class="lead">
        Filas con el mayor error absoluto. Investiga estas para entender
        las causas: ¿son outliers reales? ¿errores de captura? ¿segmentos
        que el modelo no aprendió?
      </p>
      {table_html}
      {excel_link}

      <h3 style="margin-top:24px">Distribución y patrón del error</h3>
      <p class="lead">
        Tres lecturas complementarias: <b>(1)</b> distribución del error
        (¿cola larga?), <b>(2)</b> residuos vs predicción (¿sesgo
        sistemático?, ¿heteroscedasticidad?), <b>(3)</b> error en el
        tiempo (¿el modelo se está degradando?).
      </p>
      <div class="charts-grid">
        {f'<div class="chart-block">{hist_html}</div>' if hist_html else ""}
        {f'<div class="chart-block">{resid_html}</div>' if resid_html else ""}
      </div>
      {f'<div class="chart-block" style="margin-top:8px">{time_html}</div>' if time_html else ""}
    </section>
    """


def build_diagnostic_links_section(
    variety: str,
    reports_dir,
) -> str:
    """Linkea reportes diagnosticos (EDA + Residuals) si existen como hermanos.

    Detecta automaticamente:
      - reports/EDA_<variety>_*.html  (mas reciente por mtime)
      - reports/residuals_<variety>_*.html (todas)

    Si ninguno existe, devuelve cadena vacia (la seccion se omite).
    """
    from pathlib import Path

    rdir = Path(reports_dir)
    if not rdir.exists():
        return ""

    eda_candidates = sorted(
        rdir.glob(f"EDA_{variety}_*.html"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    residual_candidates = sorted(
        rdir.glob(f"residuals_{variety}_*.html"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    cards = []
    if eda_candidates:
        latest_eda = eda_candidates[0].name
        cards.append(
            f'<a class="diag-link" href="{escape(latest_eda)}" target="_blank">'
            f'<div class="diag-icon">📊</div>'
            f'<div class="diag-meta">'
            f'<div class="diag-title">EDA — Analisis del dataset</div>'
            f'<div class="diag-sub">Forma de las variables · ciclos temporales · '
            f"colinealidad · informacion vs target · cambios entre años</div>"
            f"</div></a>"
        )
    if residual_candidates:
        # Mostrar todos los modelos (xgb_v1, lgb_v3, etc.)
        for r in residual_candidates[:5]:
            label = r.stem.replace(f"residuals_{variety}_", "")
            cards.append(
                f'<a class="diag-link" href="{escape(r.name)}" target="_blank">'
                f'<div class="diag-icon">🔬</div>'
                f'<div class="diag-meta">'
                f'<div class="diag-title">Diagnostico de errores — {escape(label)}</div>'
                f'<div class="diag-sub">¿El error se distribuye uniforme? '
                f"¿Quedan patrones temporales? ¿Es normal o hay sesgo?</div>"
                f"</div></a>"
            )

    if not cards:
        return ""

    return f"""
    <section>
      <div class="eyebrow">Diagnosticos profundos</div>
      <h2>Reportes vinculados</h2>
      <p class="muted">
        Reportes auxiliares con tests estadisticos detallados. Estan en
        <code>reports/</code> y como artifacts en MLflow.
      </p>
      <div class="diag-grid">
        {"".join(cards)}
      </div>
    </section>
    """


def build_actions_section(actions: list[Action]) -> str:
    items = "".join(
        f'<div class="action {escape(a.severity)}">'
        f'<div class="icon">{a.icon}</div>'
        f'<div class="body-wrap">'
        f'<div class="title">{escape(a.title)}</div>'
        f'<div class="body">{escape(a.body)}</div>'
        f"</div></div>"
        for a in actions
    )
    return f"""
    <section>
      <div class="eyebrow">Acciones recomendadas</div>
      <h2>¿Qué hacer con esto hoy?</h2>
      <p class="lead">Recomendaciones generadas automáticamente del análisis de errores y subgrupos.</p>
      {items}
    </section>
    """
