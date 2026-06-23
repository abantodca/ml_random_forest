"""Assets estáticos (CSS + JS) del dashboard global de reportes.

Extraídos de `dashboard_index.py` para que ese módulo quede enfocado en el
modelo de datos, el scan y el render. Son strings puros sin dependencias:
`render_dashboard` los inyecta inline en el HTML (`<style>{CSS}</style>` y
`<script>{JS}</script>`). No tienen lógica.

`JS` contiene el placeholder `__INITIAL__` que `render_dashboard` reemplaza por
el filename del Winner más reciente (pre-selección al abrir el dashboard).
"""

from __future__ import annotations

CSS = """
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


JS = r"""
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
