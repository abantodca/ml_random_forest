"""Activos HTML compartidos entre capas (dashboard ejecutivo y diagnostics).

Centraliza la construccion del tag `<script>` de plotly.js para que tanto
el dashboard de `step_05_evaluate` como los reportes de `diagnostics`
consuman el mismo bundle sin acoplamientos cross-package.

`plotly_js_tag()` decide entre embeber plotly.js (offline, autocontenido)
o cargarlo desde CDN segun `REPORT_PLOTLY_OFFLINE`. Va con `lru_cache`:
comprimir el bundle (~5 MB -> ~1.9 MB gzip) cuesta ~1s y solo debe pagarse
la PRIMERA vez que se genera un reporte, no al importar el modulo (antes
era una constante de modulo y cualquier `import` de la cadena html la
pagaba aunque nunca se renderizara nada).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from src.config import REPORT_PLOTLY_OFFLINE

logger = logging.getLogger(__name__)

_CDN_URL = "https://cdn.plot.ly/plotly-3.1.0.min.js"

# Loader del bundle comprimido. Plotly inline plano pesa ~4.8 MB; gzip+base64
# lo baja a ~1.9 MB, pero la descompresion (DecompressionStream) es asincrona
# y los <script> que genera `fig.to_html` llaman a `Plotly.newPlot` de forma
# sincrona durante el parseo. El stub captura esas llamadas en una cola y las
# re-ejecuta cuando el bundle real termina de descomprimirse. Navegadores sin
# DecompressionStream (o cualquier fallo) caen al CDN.
_GZ_LOADER_JS = """
(function () {
  var q = [];
  function stub(name) { return function () { q.push([name, arguments]); }; }
  window.Plotly = { newPlot: stub('newPlot'), react: stub('react'),
                    addTraces: stub('addTraces'), update: stub('update') };
  function replay() {
    for (var i = 0; i < q.length; i++) {
      try { window.Plotly[q[i][0]].apply(null, q[i][1]); } catch (e) {}
    }
    q = [];
  }
  function boot(code) {
    try { delete window.Plotly; } catch (e) { window.Plotly = undefined; }
    (0, eval)(code);
    replay();
  }
  function cdn() {
    try { delete window.Plotly; } catch (e) { window.Plotly = undefined; }
    var s = document.createElement('script');
    s.src = '%CDN%'; s.charset = 'utf-8';
    s.onload = replay;
    document.head.appendChild(s);
  }
  function start() {
    var el = document.getElementById('plotly-gz');
    if (!el || typeof DecompressionStream === 'undefined') { cdn(); return; }
    try {
      var bin = atob(el.textContent.replace(/\\s+/g, ''));
      var bytes = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      var ds = new DecompressionStream('gzip');
      new Response(new Blob([bytes]).stream().pipeThrough(ds))
        .text().then(boot, cdn);
    } catch (e) { cdn(); }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else { start(); }
})();
""".replace("%CDN%", _CDN_URL)


@lru_cache(maxsize=1)
def plotly_js_tag() -> str:
    """Tag <script> con plotly.js. Offline (default, gzip ~1.9 MB) o CDN.

    Modo controlado por `REPORT_PLOTLY_OFFLINE` en config:
      True  -> bundle gzip+base64 inline + loader con DecompressionStream
               (HTML autocontenido, funciona sin internet; navegadores
               antiguos caen al CDN)
      False -> CDN (HTML mas liviano pero requiere internet)
    """
    cdn_tag = f'<script charset="utf-8" src="{_CDN_URL}"></script>'
    if not REPORT_PLOTLY_OFFLINE:
        return cdn_tag
    try:
        import base64
        import gzip

        from plotly.offline import get_plotlyjs

        b64 = base64.b64encode(gzip.compress(get_plotlyjs().encode("utf-8"), 9)).decode("ascii")
        return (
            f'<script id="plotly-gz" type="text/plain">{b64}</script>'
            f"<script>{_GZ_LOADER_JS}</script>"
        )
    except Exception as exc:
        # Sin bundle offline el reporte sigue funcionando via CDN (requiere
        # internet al abrirlo); dejar rastro del porque.
        logger.warning("plotly.js offline no disponible, cayendo a CDN: %s", exc)
        return cdn_tag


__all__ = ["plotly_js_tag"]
