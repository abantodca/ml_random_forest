"""Generador de `reports/index.html` — dashboard global de reportes.

Reemplaza al `index.html` JS-dinamico viejo (que dependia de nginx autoindex
y tenia un loop bug: nginx servia el `index.html` a `fetch('./')` en vez del
listing del directorio). Este modulo escanea `reports/` server-side y emite
HTML estatico autocontenido con:

  - Topbar con "Latest" callouts (ultimo Winner / EDA por variedad principal).
  - Sidebar 280px AGRUPADO por VARIEDAD -> TIPO (Winner/EDA/Residuals).
    Cada grupo de variedad colapsa/expande; muestra los 3 mas recientes
    por default y "Ver todos N" al pie. Search input filtra in-place.
  - Iframe central que carga el HTML seleccionado al click. Pre-selecciona
    el Winner mas reciente (caso comun: usuario abre dashboard tras un
    training y quiere ver el ultimo).
  - Boton Refresh con cache-buster (regenera la pagina con ?t=<ts>) por si
    el navegador cachea el archivo entre runs.

Categorizacion (regex sobre filename):
    EDA_<variety>_<YYYY-MM-DD_HH-MM>.html        -> EDA
    Winner_<variety>_<YYYY-MM-DD_HH-MM-SS>.html  -> Winner por-run
    Winner_<variety>.html                         -> Winner legacy (sin ts)
    residuals_<variety>_<run>.html                -> Residual diagnostics
    Winner_<variety>_*.xlsx                       -> Excel ejecutivo
    *.xlsx / *.json                               -> grupo "Sin variedad"

Funciona via http://localhost:8080/reports/ (nginx) y tambien file://.

Uso manual:
    python -m src.diagnostics.dashboard_index
    docker compose run --rm --no-deps --entrypoint python trainer \\
      -m src.diagnostics.dashboard_index

Llamado automaticamente por `variety_runner.train_variety` al final del
training (despues del register MLflow) para que el index siempre refleje
los runs nuevos sin pasos manuales.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modelo de datos
# ---------------------------------------------------------------------------
@dataclass
class ReportFile:
    filename: str
    kind: str          # "winner" | "eda" | "resid" | "excel" | "json" | "other"
    variety: str | None
    label: str         # texto principal (timestamp o nombre humano)
    sub: str           # texto secundario (modelo, etc.)
    ext: str
    mtime: datetime


# Regex compartidos. `.+?` non-greedy para tolerar variedades con `_`
# (ej. POP_HASS); el ancla del timestamp `\d{4}-\d{2}-\d{2}` desambigua.
_RE_EDA = re.compile(r"^EDA_(.+?)_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})$")
_RE_WINNER_TS = re.compile(
    r"^Winner_(.+?)_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})(?:-(\d{2}))?$"
)
_RE_WINNER_LEGACY = re.compile(r"^Winner_(.+?)$")
_RE_RESID = re.compile(r"^residuals_(.+?)_(.+)$")


def _classify(path: Path) -> ReportFile | None:
    """Devuelve un ReportFile si el archivo encaja en algun grupo, sino None."""
    name = path.name
    if name in ("index.html", "index_static.html") or name.startswith("."):
        return None
    if not path.is_file():
        return None

    base = name.rsplit(".", 1)[0]
    ext = path.suffix.lstrip(".").lower()
    mtime = datetime.fromtimestamp(path.stat().st_mtime)

    # EDA
    if name.startswith("EDA_") and ext == "html":
        m = _RE_EDA.match(base)
        if m:
            label = f"{m.group(2)} {m.group(3)}:{m.group(4)}"
            return ReportFile(name, "eda", m.group(1), label, "", ext, mtime)
        return ReportFile(name, "eda", None, base, "", ext, mtime)

    # Winner HTML (por-run o legacy)
    if name.startswith("Winner_") and ext == "html":
        m = _RE_WINNER_TS.match(base)
        if m:
            secs = m.group(5) or "00"
            label = f"{m.group(2)} {m.group(3)}:{m.group(4)}:{secs}"
            return ReportFile(name, "winner", m.group(1), label, "", ext, mtime)
        # Legacy: Winner_<variety>.html sin timestamp. Usamos mtime como
        # label asi se ordena cronologicamente junto a los Winners por-run.
        m = _RE_WINNER_LEGACY.match(base)
        if m:
            return ReportFile(
                name, "winner", m.group(1),
                mtime.strftime("%Y-%m-%d %H:%M") + " (legacy)",
                "Winner sin run-id",
                ext, mtime,
            )

    # Winner Excel
    if name.startswith("Winner_") and ext == "xlsx":
        # Reusa los mismos regex pero adapta el label
        m = _RE_WINNER_TS.match(base)
        if m:
            secs = m.group(5) or "00"
            label = f"{m.group(2)} {m.group(3)}:{m.group(4)}:{secs}"
            return ReportFile(name, "excel", m.group(1), label,
                              "Excel ejecutivo", ext, mtime)
        m = _RE_WINNER_LEGACY.match(base)
        if m:
            return ReportFile(
                name, "excel", m.group(1),
                mtime.strftime("%Y-%m-%d %H:%M") + " (legacy)",
                "Excel ejecutivo", ext, mtime,
            )

    # Residuals
    if name.startswith("residuals_") and ext == "html":
        m = _RE_RESID.match(base)
        if m:
            return ReportFile(name, "resid", m.group(1), m.group(2),
                              "Diagnostico residuales", ext, mtime)

    # Sin variedad reconocible -> descartar del indice. El dashboard solo
    # lista archivos cuya variedad sale del filename; los huerfanos
    # (JSON sidecars, xlsx sueltos, html ad-hoc) quedan accesibles
    # directamente via http://localhost:8080/reports/<file> pero no
    # ensucian el sidebar.
    return None


# ---------------------------------------------------------------------------
# Scan + organizacion
# ---------------------------------------------------------------------------
@dataclass
class VarietyBucket:
    variety: str
    winners: list[ReportFile] = field(default_factory=list)
    edas: list[ReportFile] = field(default_factory=list)
    resids: list[ReportFile] = field(default_factory=list)
    excels: list[ReportFile] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.winners) + len(self.edas) + len(self.resids) + len(self.excels)


@dataclass
class ScanResult:
    by_variety: dict[str, VarietyBucket]

    @property
    def total(self) -> int:
        return sum(b.total for b in self.by_variety.values())


def scan_reports(reports_dir: Path) -> ScanResult:
    if not reports_dir.exists():
        return ScanResult({})

    by_var: dict[str, VarietyBucket] = {}

    for p in reports_dir.iterdir():
        rf = _classify(p)
        if rf is None or not rf.variety:
            continue
        bucket = by_var.setdefault(rf.variety, VarietyBucket(rf.variety))
        if rf.kind == "winner":
            bucket.winners.append(rf)
        elif rf.kind == "eda":
            bucket.edas.append(rf)
        elif rf.kind == "resid":
            bucket.resids.append(rf)
        elif rf.kind == "excel":
            bucket.excels.append(rf)

    # Sort: mas reciente primero
    for b in by_var.values():
        for items in (b.winners, b.edas, b.resids, b.excels):
            items.sort(key=lambda x: x.mtime, reverse=True)

    return ScanResult(by_var)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
_CSS = """
:root {
  --primary: #2563eb;
  --primary-dark: #1d4ed8;
  --accent: #7c3aed;
  --gray-50: #f8fafc;
  --gray-100: #f1f5f9;
  --gray-200: #e2e8f0;
  --gray-300: #cbd5e1;
  --gray-400: #94a3b8;
  --gray-500: #64748b;
  --gray-700: #334155;
  --gray-800: #1e293b;
  --gray-900: #0f172a;
  --sidebar-width: 320px;
  --header-height: 64px;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  color: var(--gray-900); background: var(--gray-50); overflow: hidden;
}

/* Topbar */
header.topbar {
  height: var(--header-height);
  background: linear-gradient(90deg, #1e3a8a, #2563eb);
  color: white; display: flex; align-items: center; justify-content: space-between;
  padding: 0 20px; box-shadow: 0 2px 8px rgba(15,23,42,.1);
  position: relative; z-index: 10;
}
.brand { font-weight: 600; font-size: 15px; }
.brand small { font-weight: 400; opacity: .75; margin-left: 6px; font-size: 11px; }
.topbar-actions { display: flex; gap: 8px; align-items: center; }
.latest-pill {
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18);
  padding: 5px 11px; border-radius: 999px; font-size: 11px;
  display: inline-flex; gap: 6px; align-items: center; cursor: pointer;
  color: white; text-decoration: none; transition: background .12s;
  font-variant-numeric: tabular-nums;
}
.latest-pill:hover { background: rgba(255,255,255,.22); }
.latest-pill .dot { width:6px;height:6px;border-radius:50%; background:#34d399; }
.refresh-btn {
  background: rgba(255,255,255,.1); color: white; border: 1px solid rgba(255,255,255,.2);
  padding: 6px 12px; border-radius: 6px; font-size: 12px; cursor: pointer;
}
.refresh-btn:hover { background: rgba(255,255,255,.18); }
.count { font-size: 11px; opacity: .75; font-variant-numeric: tabular-nums; }

/* Layout */
main.layout { display: flex; height: calc(100% - var(--header-height)); }

/* Sidebar */
aside.sidebar {
  width: var(--sidebar-width); background: white;
  border-right: 1px solid var(--gray-200); overflow-y: auto;
  display: flex; flex-direction: column;
}
.search-box {
  position: sticky; top: 0; background: white; padding: 12px;
  border-bottom: 1px solid var(--gray-200); z-index: 2;
}
.search-box input {
  width: 100%; padding: 7px 10px; border: 1px solid var(--gray-300);
  border-radius: 6px; font-size: 13px; outline: none;
  font-family: inherit;
}
.search-box input:focus { border-color: var(--primary); }

.variety-block { border-bottom: 1px solid var(--gray-100); }
.variety-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px; cursor: pointer; user-select: none;
  background: var(--gray-50); font-size: 12px;
  border-bottom: 1px solid var(--gray-100);
  transition: background .12s;
}
.variety-header:hover { background: var(--gray-100); }
.variety-header .name {
  font-weight: 600; color: var(--gray-800);
  display: flex; gap: 6px; align-items: center;
}
.variety-header .icon-folder { font-size: 12px; }
.variety-header .badge {
  background: white; color: var(--gray-700); border: 1px solid var(--gray-200);
  padding: 1px 7px; border-radius: 999px; font-size: 10px; font-weight: 600;
}
.variety-header .arrow {
  color: var(--gray-400); font-size: 10px; transition: transform .15s;
}
.variety-block.collapsed .arrow { transform: rotate(-90deg); }
.variety-block.collapsed .variety-body { display: none; }

.kind-block { padding: 6px 0; }
.kind-header {
  padding: 4px 18px; font-size: 9px; font-weight: 600;
  color: var(--gray-500); text-transform: uppercase; letter-spacing: .08em;
  display: flex; justify-content: space-between; align-items: center;
}
.kind-header .kbadge {
  background: var(--gray-100); color: var(--gray-700);
  padding: 0 6px; border-radius: 999px; font-size: 9px;
}

.item {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 16px 7px 28px;
  color: var(--gray-700); cursor: pointer; font-size: 12px;
  border-left: 2px solid transparent;
  transition: background .1s, border-color .1s, color .1s;
}
.item:hover { background: var(--gray-50); color: var(--gray-900); }
.item.active {
  background: #eff6ff; border-left-color: var(--primary);
  color: var(--primary-dark); font-weight: 500;
}
.item .item-body { display: flex; flex-direction: column; min-width: 0; flex: 1; }
.item .item-name {
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.item .item-sub { font-size: 10px; color: var(--gray-500); margin-top: 1px; }
.item.active .item-sub { color: var(--primary); opacity: .7; }
.ext {
  font-size: 8px; padding: 1px 5px; border-radius: 3px;
  font-weight: 700; letter-spacing: .04em;
}
.ext.html { background: #dbeafe; color: #1e40af; }
.ext.xlsx { background: #dcfce7; color: #166534; }
.ext.json { background: #fef3c7; color: #92400e; }
.ext.other { background: var(--gray-100); color: var(--gray-700); }

.show-more {
  padding: 6px 16px 6px 28px; font-size: 11px;
  color: var(--primary); cursor: pointer; user-select: none;
}
.show-more:hover { background: var(--gray-50); }
.kind-block.expanded .item.hidden-extra { display: flex; }
.kind-block .item.hidden-extra { display: none; }
.kind-block.expanded .show-more.expand { display: none; }
.kind-block .show-more.collapse { display: none; }
.kind-block.expanded .show-more.collapse { display: block; }

.empty-sidebar {
  padding: 32px 16px; text-align: center;
  color: var(--gray-500); font-size: 13px;
}
.empty-sidebar code {
  background: var(--gray-100); padding: 2px 6px; border-radius: 4px; font-size: 11px;
}

/* Content */
section.content { flex: 1; position: relative; background: var(--gray-100); }
.breadcrumb {
  position: absolute; top: 0; left: 0; right: 0;
  background: white; border-bottom: 1px solid var(--gray-200);
  padding: 8px 16px; font-size: 12px; color: var(--gray-700);
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: 0 1px 2px rgba(15,23,42,.04); z-index: 5;
}
.breadcrumb .path { font-family: 'JetBrains Mono', Menlo, monospace; font-size: 11px; }
.breadcrumb .path-sep { color: var(--gray-300); margin: 0 6px; }
.breadcrumb a { color: var(--primary); text-decoration: none; font-size: 11px; }
.breadcrumb a:hover { text-decoration: underline; }
section.content iframe {
  width: 100%; height: calc(100% - 33px); border: 0; background: white;
  display: block; margin-top: 33px;
}
.placeholder {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  height: 100%; color: var(--gray-500); padding: 32px; text-align: center;
}
.placeholder .big { font-size: 48px; margin-bottom: 16px; opacity: .35; }
.placeholder h2 { color: var(--gray-700); margin: 0 0 8px; font-size: 18px; }

@media (max-width: 760px) {
  :root { --sidebar-width: 100%; }
  main.layout { flex-direction: column; }
  aside.sidebar { max-height: 280px; border-right: 0; border-bottom: 1px solid var(--gray-200); }
  .latest-pill { display: none; }
}
"""


def _render_item(rf: ReportFile, *, hidden: bool = False) -> str:
    ext_class = rf.ext if rf.ext in {"html", "xlsx", "json"} else "other"
    sub_html = ""
    if rf.sub:
        sub_html = f'<div class="item-sub">{escape(rf.sub)}</div>'
    extra_class = " hidden-extra" if hidden else ""
    return (
        f'<div class="item{extra_class}" data-href="{escape(rf.filename)}" '
        f'data-search="{escape((rf.variety or "") + " " + rf.label + " " + rf.sub).lower()}" '
        f'onclick="loadReport(this)">'
        f'<span class="ext {ext_class}">{escape(rf.ext.upper())}</span>'
        f'<div class="item-body">'
        f'<div class="item-name">{escape(rf.label)}</div>'
        f'{sub_html}</div></div>'
    )


def _render_kind_block(label: str, icon: str, items: list[ReportFile],
                       *, collapse_after: int = 3) -> str:
    if not items:
        return ""
    visible = items[:collapse_after]
    extra = items[collapse_after:]
    items_html = "".join(_render_item(it) for it in visible)
    items_html += "".join(_render_item(it, hidden=True) for it in extra)
    show_more = ""
    if extra:
        show_more = (
            f'<div class="show-more expand" onclick="toggleExtra(this)">'
            f'+ Ver {len(extra)} mas &#x25BC;</div>'
            f'<div class="show-more collapse" onclick="toggleExtra(this)">'
            f'- Colapsar &#x25B2;</div>'
        )
    return (
        f'<div class="kind-block">'
        f'<div class="kind-header">'
        f'<span>{icon} {escape(label)}</span>'
        f'<span class="kbadge">{len(items)}</span>'
        f'</div>'
        f'{items_html}'
        f'{show_more}'
        f'</div>'
    )


def _render_variety_block(b: VarietyBucket) -> str:
    body = ""
    body += _render_kind_block("Winners", "&#x1F3C6;", b.winners)
    body += _render_kind_block("EDA", "&#x1F4CA;", b.edas)
    body += _render_kind_block("Residuals", "&#x1F52C;", b.resids)
    body += _render_kind_block("Excel", "&#x1F4D1;", b.excels)
    return (
        f'<div class="variety-block" data-variety="{escape(b.variety)}">'
        f'<div class="variety-header" onclick="toggleVariety(this.parentElement)">'
        f'<span class="name"><span class="icon-folder">&#x1F4C2;</span>'
        f'{escape(b.variety)}</span>'
        f'<span><span class="badge">{b.total}</span> '
        f'<span class="arrow">&#x25BC;</span></span>'
        f'</div>'
        f'<div class="variety-body">{body}</div>'
        f'</div>'
    )


def _latest_pill(scan: ScanResult) -> str:
    """Pill 'Latest' del Winner mas reciente entre todas las variedades."""
    latest: ReportFile | None = None
    for b in scan.by_variety.values():
        if b.winners and (latest is None or b.winners[0].mtime > latest.mtime):
            latest = b.winners[0]
    if not latest:
        return ""
    return (
        f'<a class="latest-pill" href="#" data-href="{escape(latest.filename)}" '
        f'onclick="loadFromTopbar(event, this)">'
        f'<span class="dot"></span>'
        f'<span>Latest: <b>{escape(latest.variety or "?")}</b> &middot; '
        f'{escape(latest.label)}</span></a>'
    )


def _initial_href(scan: ScanResult) -> str:
    """Pre-selecciona el Winner mas reciente. Si no hay, primer EDA."""
    latest_winner: ReportFile | None = None
    for b in scan.by_variety.values():
        if b.winners and (latest_winner is None
                          or b.winners[0].mtime > latest_winner.mtime):
            latest_winner = b.winners[0]
    if latest_winner:
        return latest_winner.filename
    for b in scan.by_variety.values():
        if b.edas:
            return b.edas[0].filename
    return ""


_JS = r"""
function toggleVariety(block) {
  block.classList.toggle('collapsed');
}
function toggleExtra(el) {
  const block = el.closest('.kind-block');
  if (block) block.classList.toggle('expanded');
}
function loadReport(el) {
  const href = el.dataset.href;
  if (!href) return;
  document.querySelectorAll('.item.active').forEach(x => x.classList.remove('active'));
  el.classList.add('active');
  const content = document.getElementById('content');
  const ext = (href.match(/\.([^.]+)$/) || [])[1] || '';
  if (ext.toLowerCase() === 'html') {
    content.innerHTML = `
      <div class="breadcrumb">
        <span class="path">reports<span class="path-sep">/</span>${decodeURIComponent(href)}</span>
        <a href="${encodeURI(href)}" target="_blank">Abrir en pestana nueva &#x2197;</a>
      </div>
      <iframe src="${encodeURI(href)}" referrerpolicy="no-referrer"></iframe>`;
  } else {
    content.innerHTML = `
      <div class="placeholder">
        <div class="big">${ext.toLowerCase() === 'xlsx' ? '📑' : '🗂️'}</div>
        <h2>${decodeURIComponent(href)}</h2>
        <p>${ext.toUpperCase()} no se embebe en el navegador.</p>
        <p style="margin-top:16px;">
          <a href="${encodeURI(href)}" download
             style="color:var(--primary); text-decoration:none; font-weight:500;">
             &#x2B07; Descargar archivo</a></p>
      </div>`;
  }
}
function loadFromTopbar(ev, el) {
  ev.preventDefault();
  const target = document.querySelector(`.item[data-href="${el.dataset.href}"]`);
  if (target) {
    // Expandir variety y kind block que lo contienen
    const vblock = target.closest('.variety-block');
    if (vblock) vblock.classList.remove('collapsed');
    const kblock = target.closest('.kind-block');
    if (kblock) kblock.classList.add('expanded');
    target.scrollIntoView({block: 'center'});
    loadReport(target);
  }
}
function refreshDashboard() {
  const url = new URL(location.href);
  url.searchParams.set('t', Date.now());
  location.href = url.toString();
}
function applyFilter(q) {
  q = q.trim().toLowerCase();
  document.querySelectorAll('.item').forEach(it => {
    const match = !q || it.dataset.search.includes(q);
    it.style.display = match ? '' : 'none';
  });
  // Hide variety blocks where ALL items are filtered out
  document.querySelectorAll('.variety-block').forEach(vb => {
    const visible = Array.from(vb.querySelectorAll('.item'))
      .some(i => i.style.display !== 'none');
    vb.style.display = visible ? '' : 'none';
    if (q && visible) vb.classList.remove('collapsed'); // expandir matches
  });
}
// Init
document.addEventListener('DOMContentLoaded', () => {
  const search = document.getElementById('search');
  if (search) search.addEventListener('input', e => applyFilter(e.target.value));
  const initial = document.querySelector('.item[data-href="__INITIAL__"]');
  if (initial) {
    const vblock = initial.closest('.variety-block');
    if (vblock) vblock.classList.remove('collapsed');
    loadReport(initial);
  }
});
"""


def render_dashboard(scan: ScanResult) -> str:
    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    initial = _initial_href(scan)

    # Sidebar
    if scan.total == 0:
        sidebar_body = (
            '<div class="empty-sidebar">No hay reportes aun.<br><br>'
            'Corre <code>task train VARIETIES=POP TUNING=smoke</code> '
            'para generar.</div>'
        )
    else:
        # Variedades ordenadas por mas reciente Winner (o EDA si no hay)
        def _sort_key(b: VarietyBucket) -> datetime:
            for items in (b.winners, b.edas, b.resids, b.excels):
                if items:
                    return items[0].mtime
            return datetime.min
        ordered = sorted(scan.by_variety.values(), key=_sort_key, reverse=True)
        sidebar_body = "".join(_render_variety_block(b) for b in ordered)

    js = _JS.replace("__INITIAL__", escape(initial)) if initial else _JS

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<title>ml_training - Reports Dashboard</title>
<style>{_CSS}</style>
</head>
<body>
<header class="topbar">
  <div class="brand">ml_training <small>- Reports Dashboard</small></div>
  <div class="topbar-actions">
    {_latest_pill(scan)}
    <span class="count">{scan.total} reporte{'' if scan.total == 1 else 's'}
      &middot; {escape(ts_now)}</span>
    <button class="refresh-btn" onclick="refreshDashboard()">&#x21BB; Refresh</button>
  </div>
</header>
<main class="layout">
  <aside class="sidebar">
    <div class="search-box">
      <input id="search" type="search" placeholder="Buscar reporte...">
    </div>
    {sidebar_body}
  </aside>
  <section class="content" id="content">
    <div class="placeholder">
      <div class="big">&#x1F4CA;</div>
      <h2>Selecciona un reporte</h2>
      <p>El sidebar agrupa los reportes por <b>variedad</b> y luego por tipo
      (Winners, EDA, Residuals, Excel).</p>
      <p>Cada training acumula un Winner por-run con su timestamp.</p>
    </div>
  </section>
</main>
<script>{js}</script>
</body>
</html>"""


def write_dashboard(reports_dir: Path,
                    *, filename: str = "index.html") -> Path:
    """Escanea reports_dir y escribe reports_dir/<filename>. Devuelve el path.

    Default `filename='index.html'`: reemplaza al index.html JS-dinamico
    viejo (que estaba bugueado por nginx vs autoindex). Pasar un filename
    distinto si se quiere cohabitar con otro index (ej. 'index_static.html'
    para snapshot archivable).
    """
    scan = scan_reports(reports_dir)
    html = render_dashboard(scan)
    out = reports_dir / filename
    # Atomic write-then-rename para evitar race condition cuando multiples
    # procesos paralelos (variety_runner) regeneran el mismo index.html. El
    # tmp es per-PID para que escrituras concurrentes no se pisen entre si;
    # os.replace es atomico en POSIX y Windows.
    tmp = reports_dir / f"{filename}.tmp.{os.getpid()}"
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, out)
    logger.info(f"Dashboard regenerado: {out} ({scan.total} reportes indexados)")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> int:
    parser = argparse.ArgumentParser(description="Genera reports/index.html")
    parser.add_argument(
        "--reports-dir", default=None,
        help="Override del directorio reports/ (default: config.REPORTS_DIR)",
    )
    parser.add_argument(
        "--filename", default="index.html",
        help="Nombre del archivo de salida (default: index.html)",
    )
    args = parser.parse_args()

    if args.reports_dir:
        reports_dir = Path(args.reports_dir)
    else:
        from src.config import REPORTS_DIR  # lazy import
        reports_dir = REPORTS_DIR

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = write_dashboard(reports_dir, filename=args.filename)
    print(f"\n  Dashboard: file://{out}")
    print(f"  Via nginx: http://localhost:8080/reports/{args.filename}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
