"""Recursos compartidos del dashboard HTML: CSS + plotly.js bundle.

El tag canonico de plotly.js (offline vs CDN) vive en
`src.utils.html_assets` para evitar acoplamiento cross-package entre
`diagnostics/` y `step_05_evaluate/`. Aqui solo se re-exporta como
`_PLOTLY_JS_TAG` para callers internos del paquete html. `DASHBOARD_CSS`
es la hoja de estilos completa del dashboard ejecutivo (hero, secciones,
KPIs, charts, technical details).
"""
from __future__ import annotations

# Re-export interno del paquete html. Los consumidores externos
# (diagnostics, etc.) deben importar directamente desde
# `src.utils.html_assets` para no depender de simbolos privados.
from src.utils.html_assets import PLOTLY_JS_TAG as _PLOTLY_JS_TAG

DASHBOARD_CSS = """
:root {
  --navy:#1e293b; --navy-2:#0f766e; --gold:#c9a961; --teal:#0f766e;
  --green:#15803d; --green-2:#22c55e; --amber:#b45309; --red:#b91c1c;
  --gray-50:#f8fafc; --gray-100:#f1f5f9; --gray-200:#e2e8f0;
  --gray-500:#64748b; --gray-700:#334155; --gray-900:#0f172a;
  --shadow-sm: 0 1px 3px rgba(15,23,42,.06), 0 1px 2px rgba(15,23,42,.04);
  --shadow-md: 0 4px 12px rgba(15,23,42,.08), 0 2px 4px rgba(15,23,42,.05);
}
* { box-sizing: border-box; }
body {
  margin:0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--gray-50); color: var(--gray-900); line-height:1.6;
}
.wrap { max-width: 1280px; margin: 0 auto; padding: 24px; }

/* ===== HERO ===== */
.hero {
  border-radius: 16px; padding: 32px 36px; margin-bottom: 22px;
  box-shadow: 0 10px 30px rgba(15,23,42,.22);
  display: flex; flex-wrap: wrap; gap: 22px; align-items: stretch;
}
.hero.green   { background: linear-gradient(135deg, #0f172a 0%, #0f766e 135%); color: white; }
.hero.green-2 { background: linear-gradient(135deg, #0f172a 0%, #134e4a 110%); color: white; }
.hero.amber   { background: linear-gradient(135deg, #78350f 0%, #d97706 130%); color: white; }
.hero.red     { background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 130%); color: white; }

/* KPI chips del hero (semaforo) */
.hero-chips { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 18px; }
.hero-chip {
  background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.22);
  border-radius: 12px; padding: 10px 16px; min-width: 150px;
  backdrop-filter: blur(2px);
}
.hero-chip .chip-label { font-size: 10.5px; letter-spacing: .1em;
  text-transform: uppercase; font-weight: 700; color: rgba(255,255,255,.72); }
.hero-chip .chip-value { font-size: 24px; font-weight: 800; margin-top: 2px; }
.hero-chip .chip-sub { font-size: 11px; color: rgba(255,255,255,.66); margin-top: 1px; }
.hero-chip.ok   .chip-value { color: #6ee7b7; }
.hero-chip.warn .chip-value { color: #fcd34d; }
.hero-chip.bad  .chip-value { color: #fca5a5; }
.hero-text { flex: 1 1 540px; }
.hero .eyebrow { color: rgba(255,255,255,.7); font-size: 12px;
  letter-spacing: .14em; text-transform: uppercase; font-weight: 600; }
.hero h1 { margin: 8px 0 4px; font-size: 30px; font-weight: 700; }
.hero .meta { color: rgba(255,255,255,.78); font-size: 13px; }

.verdict-badge {
  display: inline-flex; align-items: center; gap: 10px;
  background: rgba(255,255,255,.18); backdrop-filter: blur(2px);
  padding: 10px 18px; border-radius: 10px; font-size: 14px;
  font-weight: 700; margin-top: 14px; letter-spacing: .03em;
  border: 1.5px solid rgba(255,255,255,.32);
}

.verdict-badge .icon { font-size: 22px; }
.verdict-headline { font-size: 17px; margin: 14px 0 6px; font-weight: 600; }
.verdict-body { font-size: 14px; color: rgba(255,255,255,.92);
  max-width: 720px; line-height: 1.55; }

.hero-side { flex: 0 0 auto; min-width: 240px; display: flex;
  flex-direction: column; gap: 10px; justify-content: center; }
.btn-download {
  display: inline-flex; align-items: center; gap: 12px;
  background: white; color: var(--navy); font-weight: 700;
  padding: 14px 20px; border-radius: 10px; font-size: 14px;
  text-decoration: none; box-shadow: 0 4px 14px rgba(0,0,0,.18);
  transition: transform .12s ease;
}
.btn-download:hover { transform: translateY(-1px); }
.btn-download.disabled { background: rgba(255,255,255,.14);
  color: rgba(255,255,255,.7); cursor: not-allowed; box-shadow: none; }
.btn-download .icon { font-size: 18px; }
.btn-download .label-sub { display:block; font-size: 11px; font-weight: 500;
  opacity: .85; margin-top: 2px; }

/* ===== SECTIONS ===== */
section { background:white; border:1px solid var(--gray-200); border-radius:12px;
  padding:24px 28px; margin-bottom:18px; box-shadow:var(--shadow-sm);
  transition: box-shadow .15s ease; }
section:hover { box-shadow: var(--shadow-md); }
section h2 { margin:0 0 4px; font-size:19px; color: var(--navy); }

/* ===== FUNDO x FORMATO (ejecutivo) ===== */
.ff-grid { display: grid; gap: 16px; grid-template-columns: 1fr; }
@media (min-width: 920px) { .ff-grid { grid-template-columns: 1fr 1fr; } }
section .eyebrow { color: var(--gray-500); font-size: 11px;
  letter-spacing: .14em; text-transform: uppercase; font-weight: 600; }
section .lead { color: var(--gray-700); font-size: 14px; margin: 4px 0 18px; }

/* ===== CONTEXT CARDS ===== */
.ctx-grid {
  display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}
.ctx-card { background: var(--gray-50); border: 1px solid var(--gray-200);
  border-radius: 10px; padding: 12px 14px; }
.ctx-card .label { font-size: 11px; color: var(--gray-500);
  text-transform: uppercase; letter-spacing: .1em; font-weight: 700; }
.ctx-card .value { font-size: 22px; font-weight: 700; color: var(--navy);
  margin-top: 4px; }
.ctx-card .sub { font-size: 12px; color: var(--gray-500); margin-top: 4px; }

/* ===== MEGA KPIs ===== */
.kpi-mega-grid {
  display: grid; gap: 16px; grid-template-columns: 1fr; margin-top: 10px;
}
@media (min-width: 980px) { .kpi-mega-grid { grid-template-columns: repeat(3, 1fr); } }
.kpi-mega {
  border: 1px solid var(--gray-200); border-radius: 12px; padding: 20px;
  background: linear-gradient(180deg, #ffffff 0%, var(--gray-50) 100%);
  display: flex; flex-direction: column; gap: 8px;
}
.kpi-mega .question { font-size: 13px; color: var(--gray-500);
  text-transform: uppercase; letter-spacing: .08em; font-weight: 700;
  display: flex; align-items: center; gap: 8px; }
.kpi-mega .score-pill { font-size: 10px; padding: 2px 10px; border-radius: 999px;
  font-weight: 700; letter-spacing: .04em; }
.score-pill.ALTO  { background: #dcfce7; color: #166534; }
.score-pill.MEDIO { background: #fef3c7; color: #92400e; }
.score-pill.BAJO  { background: #fee2e2; color: #991b1b; }
.kpi-mega .headline { font-size: 22px; font-weight: 700; color: var(--navy);
  line-height: 1.25; margin: 4px 0 6px; }
.kpi-mega .detail { font-size: 13px; color: var(--gray-700); line-height: 1.55; }
.kpi-mega .technical { font-size: 11px; color: var(--gray-500);
  font-family: "SF Mono", Menlo, Consolas, monospace;
  border-top: 1px dashed var(--gray-200); padding-top: 8px; margin-top: 6px; }

/* ===== ACTIONS ===== */
.action {
  display: flex; gap: 14px; padding: 14px 16px; border-radius: 10px;
  margin-bottom: 10px; align-items: flex-start;
}
.action.critical { background: #fef2f2; border-left: 4px solid var(--red); }
.action.warning  { background: #fffbeb; border-left: 4px solid var(--amber); }
.action.info     { background: #f0fdf4; border-left: 4px solid var(--green); }
.action .icon { font-size: 22px; line-height: 1; flex: 0 0 auto; }
.action .body-wrap { flex: 1 1 auto; }
.action .title { font-weight: 700; color: var(--navy); margin-bottom: 4px;
  font-size: 14px; }
.action .body { font-size: 13px; color: var(--gray-700); line-height: 1.55; }

/* ===== BIAS / BACKENDS COMPARISON TABLES ===== */
.bias-table, .backends-table {
  width: 100%; border-collapse: collapse; font-size: 13px;
  margin-top: 8px;
}
.bias-table th, .bias-table td,
.backends-table th, .backends-table td {
  padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--gray-200);
}
.bias-table th, .backends-table th {
  background: var(--gray-50); color: var(--navy); font-weight: 700;
  font-size: 12px; text-transform: uppercase; letter-spacing: .03em;
}
.bias-table td.sub  { color: #b45309; font-weight: 700; }   /* subestima -> ambar oscuro */
.bias-table td.sobre { color: #1d4ed8; font-weight: 700; }  /* sobreestima -> azul */
.backends-table tr.champ-row {
  background: #f0fdf4; font-weight: 600;
}
.backends-table tr.champ-row td:first-child {
  border-left: 4px solid var(--green);
}

/* ===== HELP / GLOSSARY ===== */
details.help {
  background: var(--gray-50); border: 1px solid var(--gray-200);
  border-radius: 12px; padding: 14px 18px; margin-bottom: 18px;
}
details.help[open] { background: white; }
details.help summary {
  cursor: pointer; font-weight: 700; color: var(--navy); font-size: 14px;
  list-style: none; user-select: none; padding: 4px 0;
  display: flex; align-items: center; gap: 10px;
}
details.help summary::-webkit-details-marker { display: none; }
details.help summary::before {
  content: "▸"; font-size: 14px; color: var(--gray-500);
  transition: transform .15s ease;
}
details.help[open] summary::before { transform: rotate(90deg); }
details.help .body { padding-top: 10px; color: var(--gray-700); font-size: 14px; }
.glossary-grid {
  display: grid; gap: 8px; grid-template-columns: 1fr; margin-top: 10px;
}
@media (min-width: 800px) { .glossary-grid { grid-template-columns: 1fr 1fr; } }
.gloss-row {
  background: white; border: 1px solid var(--gray-200); border-radius: 8px;
  padding: 10px 12px;
}
.gloss-row .term { font-weight: 700; color: var(--navy); font-size: 13px;
  margin-bottom: 3px; }
.gloss-row .def { font-size: 12.5px; color: var(--gray-700); line-height: 1.5; }

/* ===== TECHNICAL DETAILS (collapsible wrapper) ===== */
details.technical {
  background: white; border: 1px solid var(--gray-200); border-radius: 14px;
  padding: 0; margin-bottom: 18px; overflow: hidden;
}
details.technical summary {
  cursor: pointer; padding: 18px 24px; background: var(--gray-50);
  border-bottom: 1px solid var(--gray-200); list-style: none;
  user-select: none; display: flex; align-items: center; gap: 12px;
}
details.technical summary::-webkit-details-marker { display: none; }
details.technical summary::before {
  content: "▸"; color: var(--gray-500); transition: transform .15s ease;
}
details.technical[open] summary::before { transform: rotate(90deg); }
details.technical summary .title { font-weight: 700; color: var(--navy);
  font-size: 15px; }
details.technical summary .sub { color: var(--gray-500); font-size: 12px;
  margin-left: auto; }
details.technical .body { padding: 22px 26px; }

.tech-block { margin-bottom: 28px; }
.tech-block:last-child { margin-bottom: 0; }
.tech-block h3 { margin: 0 0 4px; font-size: 16px; color: var(--navy); }
.tech-block .lead { color: var(--gray-700); font-size: 13.5px;
  margin: 4px 0 14px; }
.tech-block .eyebrow { color: var(--gray-500); font-size: 11px;
  letter-spacing: .14em; text-transform: uppercase; font-weight: 600; }

/* ===== JUSTIFY ===== */
.justify-text { background: var(--gray-100); border-left: 4px solid var(--gold);
  padding: 14px 18px; border-radius: 8px; font-size: 14px; color: var(--gray-700); }

/* ===== MODELS GRID + KPI CARDS (technical, reused) ===== */
.models-grid { display: grid; gap: 16px; grid-template-columns: 1fr; }
@media (min-width: 900px) { .models-grid.cols-2 { grid-template-columns: 1fr 1fr; } }
@media (min-width: 1100px) { .models-grid.cols-3 { grid-template-columns: repeat(3, 1fr); } }

.model-card {
  border:1px solid var(--gray-200); border-radius: 10px; padding: 16px;
  background: white;
}
.model-card.winner { border: 2px solid var(--gold); background: #fffdf6; }
.model-card .head { display:flex; align-items:center; justify-content:space-between;
  margin-bottom: 10px; }
.model-card .name { font-weight: 700; font-size: 15px; color: var(--navy); }
.model-card .badge {
  font-size: 10.5px; padding: 3px 10px; border-radius: 999px; font-weight: 700;
  letter-spacing: .04em; margin-left: 4px;
}
.badge.winner-tag { background: var(--gold); color: var(--navy); }
.badge.loser-tag  { background: var(--gray-200); color: var(--gray-700); }
.badge.rank { background: var(--navy); color: white; }

.kpi-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top:6px; }
.kpi {
  background: var(--gray-50); border-radius: 8px; padding: 8px 10px;
  border: 1px solid var(--gray-200);
}
.kpi .label { color: var(--gray-500); font-size: 10px; text-transform: uppercase;
  letter-spacing: .1em; font-weight: 700; }
.kpi .value { font-size: 17px; font-weight: 700; color: var(--navy); margin-top: 2px; }
.kpi .sub { color: var(--gray-500); font-size: 11px; margin-top: 2px; }
.kpi.train .value { color: var(--gray-700); }
.kpi.full  .value { color: var(--green); }

.legend { font-size: 12px; color: var(--gray-500); margin-top: 10px; line-height: 1.6; }
.legend.section-label { font-weight: 700; color: var(--gray-700); margin-top: 12px; }

/* ===== CHARTS ===== */
.charts-grid { display: grid; gap: 14px; grid-template-columns: 1fr; margin-top: 6px; }
@media (min-width: 1000px) { .charts-grid { grid-template-columns: 1fr 1fr; } }
.chart-block { border: 1px solid var(--gray-200); border-radius: 10px;
  padding: 6px; background: white; }
.chart-block .chart-title { font-size: 11.5px; color: var(--gray-500);
  text-transform: uppercase; letter-spacing: .1em; padding: 6px 8px 0; font-weight: 700; }

.model-block { margin-bottom: 16px; }
.model-block h4 { margin: 4px 0 6px; color: var(--navy); font-size: 14px;
  display: flex; align-items: center; gap: 10px; }

.boxplot-stats { display: grid; gap: 8px;
  grid-template-columns: repeat(2, 1fr); margin-bottom: 14px; }
@media (min-width: 720px) {
  .boxplot-stats { grid-template-columns: repeat(3, 1fr); }
}
@media (min-width: 1080px) {
  .boxplot-stats { grid-template-columns: repeat(6, 1fr); }
}
.stat-pill { background: var(--gray-50); border: 1px solid var(--gray-200);
  border-radius: 10px; padding: 10px 12px; text-align: center; }
.stat-pill .label { font-size: 10px; color: var(--gray-500); text-transform: uppercase;
  letter-spacing: .1em; font-weight: 700; }
.stat-pill .val { font-size: 18px; font-weight: 700; color: var(--navy);
  margin-top: 2px; }
.stat-pill .val.amber { color: var(--amber); }
.stat-pill .val.red { color: var(--red); }
.stat-pill .unit { font-size: 11px; color: var(--gray-500); font-weight: 500; }

/* ===== SUBGROUP TABS (errors by group) ===== */
.subgrp-tabs { background: white; border: 1px solid var(--gray-200);
  border-radius: 12px; overflow: hidden; }
.subgrp-tabs-nav {
  display: flex; flex-wrap: wrap; gap: 0;
  background: var(--gray-50); border-bottom: 1px solid var(--gray-200);
  padding: 0 4px;
}
.subgrp-tab-btn {
  appearance: none; background: transparent; border: 0;
  padding: 12px 16px; cursor: pointer;
  font-family: inherit; font-size: 12.5px; font-weight: 600;
  color: var(--gray-500); letter-spacing: .04em;
  border-bottom: 2.5px solid transparent;
  transition: color .12s ease, border-color .12s ease, background .12s ease;
}
.subgrp-tab-btn:hover { color: var(--navy); background: rgba(12,42,77,.04); }
.subgrp-tab-btn.active {
  color: var(--navy); border-bottom-color: var(--gold);
  background: white;
}
.subgrp-tabs-body { padding: 16px; background: white; }
.subgrp-panel { display: none; }
.subgrp-panel.active { display: block; }

.subgrp-grid { display: grid; gap: 16px; grid-template-columns: 1fr; }
@media (min-width: 1080px) {
  .subgrp-grid { grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); }
}
.subgrp-chart { border: 1px solid var(--gray-200); border-radius: 10px;
  padding: 6px; background: white; min-width: 0; }
.subgrp-table-wrap { border: 1px solid var(--gray-200); border-radius: 10px;
  background: white; padding: 10px 12px; overflow-x: auto; min-width: 0; }
.subgrp-table-title { font-size: 12px; color: var(--gray-700);
  font-weight: 700; padding: 4px 4px 10px; border-bottom: 1px solid var(--gray-100);
  display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; }
.subgrp-table-hint { font-size: 10.5px; color: var(--gray-500); font-weight: 500;
  letter-spacing: .02em; margin-left: auto; }

.subgrp-table { width: 100%; border-collapse: collapse;
  font-size: 12.5px; font-variant-numeric: tabular-nums; }
.subgrp-table thead th {
  text-align: left; padding: 8px 10px;
  font-size: 10.5px; color: var(--gray-500); font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase;
  border-bottom: 1px solid var(--gray-200); position: sticky; top: 0;
  background: white;
}
.subgrp-table thead th.num { text-align: right; }
.subgrp-table tbody td { padding: 8px 10px; border-bottom: 1px solid var(--gray-100); }
.subgrp-table tbody td.num { text-align: right; color: var(--gray-700); }
.subgrp-table tbody tr:last-child td { border-bottom: 0; }
.subgrp-table tbody tr.subgrp-row:hover { background: var(--gray-50); }
.subgrp-table tbody tr.subgrp-row.global {
  background: #fffdf6; font-weight: 600;
}
.subgrp-table tbody tr.subgrp-row.global td { color: var(--navy); }
.grp-name { display: inline-block; max-width: 220px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; vertical-align: bottom;
  color: var(--navy); font-weight: 600; }

.ratio-pill { display: inline-block; min-width: 52px; padding: 2px 9px;
  border-radius: 999px; font-size: 11px; font-weight: 700;
  letter-spacing: .02em; text-align: center; }
.ratio-pill.ref { background: var(--gray-100); color: var(--gray-700); }
.ratio-pill.good { background: #dcfce7; color: #166534; }
.ratio-pill.neutral { background: var(--gray-100); color: var(--gray-700); }
.ratio-pill.warn { background: #fef3c7; color: #92400e; }
.ratio-pill.bad { background: #fee2e2; color: #991b1b; }

footer { text-align:center; color: var(--gray-500); font-size: 12px;
  padding: 16px 0; margin-top: 8px; }

/* Diagnostic links section (EDA + Residuals) */
.diag-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px; margin-top: 12px; }
.diag-link { display: flex; align-items: center; gap: 12px;
  padding: 14px 16px; border: 1px solid var(--gray-200); border-radius: 10px;
  text-decoration: none; color: var(--navy); background: white;
  transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease; }
.diag-link:hover { transform: translateY(-1px); border-color: var(--navy);
  box-shadow: 0 4px 12px rgba(37,99,235,.12); }
.diag-icon { font-size: 24px; line-height: 1; }
.diag-meta { flex: 1; min-width: 0; }
.diag-title { font-weight: 600; font-size: 13px; margin-bottom: 2px;
  color: var(--navy); }
.diag-sub { font-size: 11px; color: var(--gray-500); }

/* Print mode: optimizado para imprimir o exportar a PDF */
@media print {
  body { background: white; padding: 0; }
  section { box-shadow: none !important; page-break-inside: avoid; }
  .diag-link { border: 1px solid #cbd5e1; }
  .diag-grid { page-break-inside: avoid; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .modebar { display: none !important; }
  /* Intencion: forzar que los <details> colapsables aparezcan abiertos al
     imprimir/exportar a PDF. No se puede modificar el atributo `open` desde
     CSS, pero podemos forzar el render del contenido (cualquier hijo que no
     sea <summary>) tanto si el details esta abierto como cerrado. */
  details[open] > * { display: block; }
  details:not([open]) > *:not(summary) { display: block !important; }
}
"""


__all__ = ["_PLOTLY_JS_TAG", "DASHBOARD_CSS"]
