"""Seccion 'Detalle tecnico' (colapsable) del dashboard ejecutivo.

Para Data Science: justificacion del campeon, panel comparativo
Train/Test/Full por modelo, scatters Drift & Overfitting (OOF vs refit) y
panel tabular (boxplot + tabla con n/p50/p90/MAPE/desviacion) del error
por subgrupo (FORMATO, FUNDO, AÑO, MES, HA, KG/HA, %INDUS) con tabs.
"""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

from src.config import DATE_COLUMN
from src.step_05_evaluate.champion import ModelResult
from src.step_05_evaluate.diagnostics import plot_pred_vs_actual_plotly
from src.step_05_evaluate.html.helpers import (
    compute_error_percentiles,
    fmt,
    kpi_card,
    plotly_div,
)
from src.step_05_evaluate.metrics import mape_safe


def _kpi_panel_for_model(r: ModelResult, *, is_winner: bool, rank: int) -> str:
    mae_train = r.metrics.get("nested_cv_mae_train_mean")
    mae_test = r.metrics.get("nested_cv_mae_mean")
    r2_test = r.metrics.get("nested_cv_r2_mean")
    gap = r.metrics.get("nested_cv_gap_mean")
    full_h = r.full_metrics_h or {}
    business_oof = r.business_metrics_oof or {}
    business_full = r.full_metrics or {}

    badge_winner = (
        '<span class="badge winner-tag">CAMPEÓN</span>'
        if is_winner
        else '<span class="badge loser-tag">descartado</span>'
    )
    head = (
        f'<div class="head">'
        f'<div class="name">{escape(r.model_type.upper())} '
        f'<span style="color:#94a3b8;font-weight:500;font-size:12px">'
        f"· {r.elapsed_seconds:.1f}s</span></div>"
        f'<div><span class="badge rank">#{rank}</span>{badge_winner}</div>'
        f"</div>"
    )
    section_h = (
        '<div class="legend section-label">Unidad del modelo · KG/JR_H</div>'
        '<div class="kpi-row">'
        + kpi_card("Train (CV)", fmt(mae_train), "MAE", "train")
        + kpi_card("Test (CV)", fmt(mae_test), f"MAE · R²={fmt(r2_test)}")
        + kpi_card(
            "Aplicación Total", fmt(full_h.get("mae")), f"MAE · R²={fmt(full_h.get('r2'))}", "full"
        )
        + "</div>"
    )
    section_b = (
        '<div class="legend section-label">Unidad de negocio · KG/JR (MAPE %)</div>'
        '<div class="kpi-row">'
        + kpi_card("Brecha Train→Test", fmt(gap, 4), "menor = más estable", "train")
        + kpi_card(
            "Test (OOF)", fmt(business_oof.get("mape"), 2, "%"), f"R²={fmt(business_oof.get('r2'))}"
        )
        + kpi_card(
            "Aplicación Total",
            fmt(business_full.get("mape"), 2, "%"),
            f"R²={fmt(business_full.get('r2'))}",
            "full",
        )
        + "</div>"
    )
    cls_card = "model-card winner" if is_winner else "model-card"
    return f'<div class="{cls_card}">{head}{section_h}{section_b}</div>'


def _scatter_block(y_real, y_pred, title: str, color: str) -> str:
    if y_real is None or y_pred is None or len(y_real) == 0:
        body = (
            '<div style="padding:32px;text-align:center;color:var(--gray-500);'
            'font-size:13px">No hay datos disponibles</div>'
        )
    else:
        body = plot_pred_vs_actual_plotly(
            y_real,
            y_pred,
            title="",
            x_label="Real (KG/JR)",
            y_label="Predicción (KG/JR)",
            color=color,
        )
    return f'<div class="chart-block"><div class="chart-title">{escape(title)}</div>{body}</div>'


def _model_charts(r: ModelResult, *, is_winner: bool) -> str:
    bv = r.business_validation
    real_oof = getattr(bv, "kg_jr_real_oof", None) if bv else None
    pred_oof = getattr(bv, "kg_jr_pred_oof", None) if bv else None
    real_full = getattr(bv, "kg_jr_real_insample", None) if bv else None
    pred_full = getattr(bv, "kg_jr_pred_insample", None) if bv else None
    badge = (
        '<span class="badge winner-tag">CAMPEÓN</span>'
        if is_winner
        else '<span class="badge loser-tag">descartado</span>'
    )
    return (
        f'<div class="model-block">'
        f"<h4>{escape(r.model_type.upper())} {badge}</h4>"
        f'<div class="charts-grid">'
        + _scatter_block(real_oof, pred_oof, "Test · OOF (honesto)", "#1f4e8a")
        + _scatter_block(real_full, pred_full, "Aplicación Total · refit (sesgado)", "#16a34a")
        + "</div></div>"
    )


_SUBGRP_MIN_N = 5
# Tope de puntos por caja: el boxplot embebe el array completo en el HTML
# (x7 tabs). El muestreo es determinista y sobre el array ORDENADO con paso
# uniforme, asi que los cuantiles que dibuja la caja se preservan aun con
# pocos puntos; 600 basta para mediana/cuartiles/bigotes estables.
_BOX_MAX_POINTS = 600


def _box_sample(arr: np.ndarray) -> np.ndarray:
    if arr.size <= _BOX_MAX_POINTS:
        return np.round(arr, 3)
    idx = np.linspace(0, arr.size - 1, _BOX_MAX_POINTS).astype(int)
    return np.round(np.sort(arr)[idx], 3)


def _build_boxplot(
    abs_err: np.ndarray,
    groups: pd.Series,
    title: str,
    group_label: str,
    min_n: int = _SUBGRP_MIN_N,
) -> str:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""
    if abs_err.size == 0:
        return ""

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=_box_sample(abs_err),
            name="GLOBAL",
            boxmean="sd",
            marker=dict(color="#2a9d8f"),
            line=dict(color="#1f6e63"),
            boxpoints="outliers",
            hovertemplate=(
                f"Error: %{{y:.2f}} kg/jornal<br>n: {int(abs_err.size)}<extra>GLOBAL</extra>"
            ),
        )
    )
    palette = [
        "#1f4e8a",
        "#b08968",
        "#d62828",
        "#6c757d",
        "#e9c46a",
        "#9d4edd",
        "#2d6a4f",
        "#bf812d",
    ]
    cats = []
    for cat in groups.unique():
        if pd.isna(cat) or str(cat) in ("", "nan", "<NA>", "None"):
            continue
        mask = (groups == cat).to_numpy()
        if mask.sum() < min_n:
            continue
        cats.append((str(cat), abs_err[mask]))
    cats.sort(key=lambda kv: float(np.median(kv[1])), reverse=True)
    for i, (cat, err) in enumerate(cats):
        fig.add_trace(
            go.Box(
                y=_box_sample(err),
                name=cat,
                boxmean=True,
                marker=dict(color=palette[i % len(palette)]),
                boxpoints="outliers",
                hovertemplate=(
                    f"Error: %{{y:.2f}} kg/jornal<br>{group_label}: {cat}<br>"
                    f"n: {int(err.size)}<extra></extra>"
                ),
            )
        )

    p50 = float(np.percentile(abs_err, 50))
    p90 = float(np.percentile(abs_err, 90))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#0c2a4d")),
        yaxis_title="Error absoluto (kg/jornal)",
        height=380,
        margin=dict(l=60, r=20, t=44, b=72),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="-apple-system, Segoe UI, Roboto, sans-serif", size=11),
        showlegend=False,
        annotations=[
            dict(
                text=(
                    f"<b>Lectura:</b> 50% del error ≤ {p50:.2f} · 90% ≤ {p90:.2f} kg/jornal · "
                    "ordenados de peor (izq) a mejor (der)"
                ),
                xref="paper",
                yref="paper",
                x=0,
                y=-0.20,
                showarrow=False,
                align="left",
                font=dict(size=10.5, color="#555"),
            )
        ],
    )
    fig.update_yaxes(gridcolor="#e7eaf0", zerolinecolor="#cbd5e0")
    return plotly_div(fig)


def _build_grouping_options(
    X_aligned: pd.DataFrame,
) -> list[tuple[str, str, pd.Series]]:
    """Devuelve [(id, label, series_categorica)] para cada agrupacion disponible.

    Combina categoricas directas (FORMATO, FUNDO), derivadas de fecha
    (AÑO, MES) y bineadas de numericas (HA, KG/HA, %INDUS).
    """
    options: list[tuple[str, str, pd.Series]] = []

    if "FORMATO" in X_aligned.columns:
        options.append(
            ("formato", "FORMATO", X_aligned["FORMATO"].astype(str).reset_index(drop=True))
        )
    if "FUNDO" in X_aligned.columns:
        options.append(("fundo", "FUNDO", X_aligned["FUNDO"].astype(str).reset_index(drop=True)))

    if DATE_COLUMN in X_aligned.columns:
        d = pd.to_datetime(X_aligned[DATE_COLUMN], errors="coerce").reset_index(drop=True)
        if d.notna().any():
            anio = d.dt.year.astype("Int64").astype(str).replace({"<NA>": ""})
            options.append(("anio", "AÑO", anio))
            month_names = [
                "Ene",
                "Feb",
                "Mar",
                "Abr",
                "May",
                "Jun",
                "Jul",
                "Ago",
                "Sep",
                "Oct",
                "Nov",
                "Dic",
            ]
            mes = d.dt.month.map(
                lambda m: f"{int(m):02d}-{month_names[int(m) - 1]}" if pd.notna(m) else ""
            )
            options.append(("mes", "MES", mes.astype(str)))

    def _bin_numeric(
        col: str, gid: str, label: str, fixed: list[tuple[float, str]] | None = None
    ) -> None:
        if col not in X_aligned.columns:
            return
        s = pd.to_numeric(X_aligned[col], errors="coerce").reset_index(drop=True)
        if s.notna().sum() < _SUBGRP_MIN_N * 2:
            return
        try:
            if fixed is not None:
                edges_cut = [-np.inf] + [e for e, _ in fixed[:-1]] + [np.inf]
                lbls_cut = [lbl for _, lbl in fixed]
                bins = pd.cut(s, bins=edges_cut, labels=lbls_cut, include_lowest=True)
            else:
                bins = pd.qcut(s, q=4, labels=["Q1 bajo", "Q2", "Q3", "Q4 alto"], duplicates="drop")
        except Exception:
            return
        options.append((gid, label, bins.astype(str).replace({"nan": ""})))

    _bin_numeric(
        "HA",
        "ha",
        "HA (tamaño)",
        fixed=[(1, "≤1 ha"), (3, "1-3 ha"), (6, "3-6 ha"), (np.inf, ">6 ha")],
    )
    _bin_numeric("KG/HA", "kgha", "KG/HA (rendimiento)")
    _bin_numeric("%INDUS", "indus", "%INDUS (calidad)")

    return options


def _ratio_pill(ratio: float) -> str:
    if not np.isfinite(ratio):
        return '<span class="ratio-pill neutral">—</span>'
    if ratio < 0.85:
        return f'<span class="ratio-pill good">{(ratio - 1) * 100:+.0f}%</span>'
    if ratio <= 1.15:
        return f'<span class="ratio-pill neutral">{(ratio - 1) * 100:+.0f}%</span>'
    if ratio <= 1.5:
        return f'<span class="ratio-pill warn">{(ratio - 1) * 100:+.0f}%</span>'
    return f'<span class="ratio-pill bad">×{ratio:.1f}</span>'


def _build_subgroup_table(
    abs_err: np.ndarray,
    real: np.ndarray,
    groups: pd.Series,
    min_n: int = _SUBGRP_MIN_N,
) -> str:
    """Tabla por subgrupo: n / p50 / p90 / MAE / MAPE / desviacion vs global."""
    p50_g = float(np.percentile(abs_err, 50))
    p90_g = float(np.percentile(abs_err, 90))
    mae_g = float(np.mean(abs_err))
    # Defensivo: si `real` y `abs_err` tienen tamaños distintos (p.ej. pickle
    # antiguo con shape desalineado), no podemos reconstruir un `pred` valido.
    # Misma validacion que en `_build_subgroup_block` (l.348). `mape_safe`
    # recibe `(real, real - abs_err)` porque solo usa |yt-yp| (== abs_err).
    mape_g = mape_safe(real, real - abs_err) if real.size == abs_err.size else float("nan")

    rows: list[dict] = [
        {
            "label": "GLOBAL",
            "is_global": True,
            "n": int(abs_err.size),
            "p50": p50_g,
            "p90": p90_g,
            "mae": mae_g,
            "mape": mape_g,
            "ratio": 1.0,
        }
    ]
    seen = set()
    for cat in groups.unique():
        if pd.isna(cat) or str(cat) in ("", "nan", "<NA>", "None") or cat in seen:
            continue
        seen.add(cat)
        mask = (groups == cat).to_numpy()
        if mask.sum() < min_n:
            continue
        cat_err = abs_err[mask]
        cat_real = real[mask] if real.size == abs_err.size else np.array([])
        nz = cat_real != 0 if cat_real.size else np.array([], dtype=bool)
        cat_mape = (
            float(np.mean(cat_err[nz] / np.abs(cat_real[nz])) * 100) if nz.any() else float("nan")
        )
        rows.append(
            {
                "label": str(cat),
                "is_global": False,
                "n": int(mask.sum()),
                "p50": float(np.percentile(cat_err, 50)),
                "p90": float(np.percentile(cat_err, 90)),
                "mae": float(np.mean(cat_err)),
                "mape": cat_mape,
                "ratio": (cat_mape / mape_g)
                if (np.isfinite(mape_g) and mape_g > 0 and np.isfinite(cat_mape))
                else float("nan"),
            }
        )
    rows[1:] = sorted(
        rows[1:],
        key=lambda r: r["mape"] if np.isfinite(r["mape"]) else -1.0,
        reverse=True,
    )

    parts: list[str] = [
        '<table class="subgrp-table">',
        "<thead><tr>",
        "<th>Grupo</th>",
        '<th class="num">n</th>',
        '<th class="num">MAE</th>',
        '<th class="num">p50</th>',
        '<th class="num">p90</th>',
        '<th class="num">MAPE</th>',
        '<th class="num">vs global</th>',
        "</tr></thead><tbody>",
    ]
    for r in rows:
        cls = "subgrp-row global" if r["is_global"] else "subgrp-row"
        ratio_cell = (
            '<span class="ratio-pill ref">ref</span>' if r["is_global"] else _ratio_pill(r["ratio"])
        )
        mape_txt = f"{r['mape']:.1f}%" if np.isfinite(r["mape"]) else "—"
        parts.append(
            f'<tr class="{cls}">'
            f'<td><span class="grp-name" title="{escape(r["label"])}">{escape(r["label"])}</span></td>'
            f'<td class="num">{r["n"]:,}</td>'
            f'<td class="num">{r["mae"]:.2f}</td>'
            f'<td class="num">{r["p50"]:.2f}</td>'
            f'<td class="num">{r["p90"]:.2f}</td>'
            f'<td class="num">{mape_txt}</td>'
            f'<td class="num">{ratio_cell}</td>'
            f"</tr>"
        )
    parts.append("</tbody></table>")
    return "".join(parts)


_SUBGRP_TAB_JS = """
        <script>
        function showSubgrpTab(rootId, tabId) {
        var root = document.getElementById(rootId);
        if (!root) return;
        root.querySelectorAll('.subgrp-tab-btn').forEach(function(b){ b.classList.remove('active'); });
        root.querySelectorAll('.subgrp-panel').forEach(function(p){ p.classList.remove('active'); });
        var btn = document.getElementById(rootId + '-btn-' + tabId);
        var panel = document.getElementById(rootId + '-panel-' + tabId);
        if (btn) btn.classList.add('active');
        if (panel) {
            panel.classList.add('active');
            panel.querySelectorAll('.plotly-graph-div').forEach(function(g){
            if (window.Plotly && Plotly.Plots && Plotly.Plots.resize) {
                try { Plotly.Plots.resize(g); } catch(e) {}
            }
            });
        }
        }
        </script>
    """


def _build_subgroup_block(
    X_aligned: pd.DataFrame | None,
    abs_errors: np.ndarray,
    real: np.ndarray,
) -> str:
    """Panel de errores por subgrupo: stats globales + tabs (boxplot + tabla)."""
    if abs_errors.size == 0:
        return '<div class="legend">No hay datos OOF para análisis de subgrupos.</div>'

    pcts = compute_error_percentiles(abs_errors)
    p50, p90, p99 = pcts["p50"], pcts["p90"], pcts["p99"]
    mae = float(np.mean(abs_errors))
    nz = real != 0 if real.size == abs_errors.size else np.array([], dtype=bool)
    mape_g = float(np.mean(abs_errors[nz] / np.abs(real[nz])) * 100) if nz.any() else float("nan")
    mape_txt = f"{mape_g:.1f}%" if np.isfinite(mape_g) else "—"

    stats = (
        '<div class="boxplot-stats">'
        f'<div class="stat-pill"><div class="label">N total</div>'
        f'<div class="val">{int(abs_errors.size):,}</div></div>'
        f'<div class="stat-pill"><div class="label">MAE (media)</div>'
        f'<div class="val">{mae:.2f}<span class="unit"> kg/jr</span></div></div>'
        f'<div class="stat-pill"><div class="label">MAPE global</div>'
        f'<div class="val">{mape_txt}</div></div>'
        f'<div class="stat-pill"><div class="label">Mediana (p50)</div>'
        f'<div class="val">{p50:.2f}<span class="unit"> kg/jr</span></div></div>'
        f'<div class="stat-pill"><div class="label">p90 (severos)</div>'
        f'<div class="val amber">{p90:.2f}<span class="unit"> kg/jr</span></div></div>'
        f'<div class="stat-pill"><div class="label">p99 (cola)</div>'
        f'<div class="val red">{p99:.2f}<span class="unit"> kg/jr</span></div></div>'
        "</div>"
    )

    if X_aligned is None or len(X_aligned) != abs_errors.size or real.size != abs_errors.size:
        empty = pd.Series([""] * abs_errors.size)
        single = _build_boxplot(abs_errors, empty, "Distribución global del error", "GLOBAL")
        return f'{stats}<div class="chart-block">{single}</div>'

    options = _build_grouping_options(X_aligned)
    if not options:
        empty = pd.Series([""] * abs_errors.size)
        single = _build_boxplot(abs_errors, empty, "Distribución global del error", "GLOBAL")
        return f'{stats}<div class="chart-block">{single}</div>'

    root_id = "subgrp"
    btns: list[str] = []
    panels: list[str] = []
    for i, (gid, glabel, gseries) in enumerate(options):
        active = " active" if i == 0 else ""
        btns.append(
            f'<button type="button" class="subgrp-tab-btn{active}" '
            f'id="{root_id}-btn-{gid}" '
            f"onclick=\"showSubgrpTab('{root_id}', '{gid}')\">"
            f"{escape(glabel)}</button>"
        )
        boxplot = _build_boxplot(
            abs_errors,
            gseries,
            f"Error por {glabel} (con GLOBAL como referencia)",
            glabel,
        )
        table = _build_subgroup_table(abs_errors, real, gseries)
        panels.append(
            f'<div class="subgrp-panel{active}" id="{root_id}-panel-{gid}">'
            f'<div class="subgrp-grid">'
            f'<div class="subgrp-chart">{boxplot}</div>'
            f'<div class="subgrp-table-wrap">'
            f'<div class="subgrp-table-title">Detalle por {escape(glabel)}'
            f'<span class="subgrp-table-hint">ordenado por MAPE descendente · '
            f"mín {_SUBGRP_MIN_N} obs.</span></div>"
            f"{table}</div></div></div>"
        )

    nav = '<div class="subgrp-tabs-nav" role="tablist">' + "".join(btns) + "</div>"
    body = '<div class="subgrp-tabs-body">' + "".join(panels) + "</div>"
    return f'{stats}<div class="subgrp-tabs" id="{root_id}">{nav}{body}</div>{_SUBGRP_TAB_JS}'


def build_technical_section(
    *,
    results: list[ModelResult],
    champion: ModelResult,
    decision: dict,
    X_aligned: pd.DataFrame | None,
    abs_errors: np.ndarray,
    real: np.ndarray,
) -> str:
    n_models = len(results)
    grid_cls = "models-grid"
    if n_models == 2:
        grid_cls += " cols-2"
    elif n_models >= 3:
        grid_cls += " cols-3"

    ranking_lookup = {row["model"]: row["rank"] for row in decision.get("ranking", [])}
    ordered = sorted(results, key=lambda r: ranking_lookup.get(r.model_type, 999))
    cards = "".join(
        _kpi_panel_for_model(
            r,
            is_winner=(r.model_type == champion.model_type),
            rank=ranking_lookup.get(r.model_type, i + 1),
        )
        for i, r in enumerate(ordered)
    )
    # Solo los charts del CAMPEON: los scatters de los descartados duplican
    # peso del HTML sin sumar a la decision (sus KPIs ya estan en las cards).
    charts = _model_charts(champion, is_winner=True)
    subgroup_block = _build_subgroup_block(X_aligned, abs_errors, real)
    justification = decision.get("justification", "")
    criteria = " → ".join(decision.get("decision_criteria", []))

    return f"""
    <details class="technical">
      <summary>
        <span class="title">🔬 Detalle técnico (para el equipo de Data Science)</span>
        <span class="sub">Click para expandir · selección de campeón, métricas Train/Test/Full, gráficos de error</span>
      </summary>
      <div class="body">
        <div class="tech-block">
          <div class="eyebrow">Justificación técnica</div>
          <h3>¿Por qué ganó {escape(champion.model_type.upper())}?</h3>
          <p class="lead">Criterio aplicado (orden estricto): {escape(criteria)}</p>
          <div class="justify-text">{escape(justification)}</div>
        </div>

        <div class="tech-block">
          <div class="eyebrow">Panel comparativo · KPIs por modelo</div>
          <h3>Métricas Train · Test · Aplicación Total</h3>
          <p class="lead">Tarjetas por modelo en la unidad del modelo (KG/JR_H, nested CV) y la unidad de negocio (KG/JR, MAPE %).</p>
          <div class="{grid_cls}">{cards}</div>
        </div>

        <div class="tech-block">
          <div class="eyebrow">Monitorización · Drift &amp; Overfitting</div>
          <h3>Distribución predicción vs real (KG/JR)</h3>
          <p class="lead">Comparar Test (OOF, sin contaminar) con Aplicación Total (refit + predict all) revela overfitting si los puntos del segundo se pegan más a la diagonal que los del primero.</p>
          {charts}
        </div>

        <div class="tech-block">
          <div class="eyebrow">Errores por subgrupo · Campeón</div>
          <h3>¿Dónde se equivoca el modelo ganador?</h3>
          <p class="lead">Distribución del error absoluto (KG/JR, OOF) del campeón <b>{escape(champion.model_type.upper())}</b>. Cambia de pestaña para analizar cada dimensión: cada panel muestra el boxplot y una tabla con n, MAE, p50, p90, MAPE y desviación vs el global. Los grupos con MAPE más alto son los más problemáticos.</p>
          {subgroup_block}
        </div>
      </div>
    </details>
    """
