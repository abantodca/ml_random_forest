"""Helpers de presentacion compartidos por las secciones del dashboard.

Cubren formateo numerico, tarjetas KPI generales y el boton de descarga.
Ningun helper aqui debe contener narrativa de negocio: solo wrapping HTML.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np


def plotly_div(fig, div_id: str = "") -> str:
    """Devuelve <div> embebible (sin plotly.js) o "" si Plotly no esta.

    Helper compartido entre `diagnostics.py` (charts globales del pipeline)
    y `html/technical.py` (boxplots por subgrupo). El script `plotly.js`
    se carga UNA sola vez desde el <head> via `html.styles._PLOTLY_JS_TAG`,
    por eso aqui siempre se pasa `include_plotlyjs=False`.
    """
    try:
        kwargs = {
            "include_plotlyjs": False,
            "full_html": False,
            "config": {"displaylogo": False, "responsive": True},
        }
        if div_id:
            kwargs["div_id"] = div_id
        return fig.to_html(**kwargs)
    except Exception:
        return ""


def compute_error_percentiles(abs_errors: np.ndarray) -> dict[str, float]:
    """Devuelve {p50, p90, p99} sobre `abs_errors`.

    Si el array esta vacio retorna NaN en cada percentil (evita el warning
    "percentile of empty array" y mantiene el contrato de un dict completo
    para los renderers que indexan por clave).
    """
    arr = np.asarray(abs_errors)
    if arr.size == 0:
        nan = float("nan")
        return {"p50": nan, "p90": nan, "p99": nan}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def fmt(value: float | None, digits: int = 4, suffix: str = "") -> str:
    """Formatea un float; devuelve '—' para None / NaN / Inf."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "—"
    return f"{value:.{digits}f}{suffix}"


def kpi_card(label: str, value: str, sub: str = "", flavor: str = "") -> str:
    """Tarjeta KPI generica usada en los paneles tecnicos."""
    cls = f"kpi {flavor}".strip()
    sub_html = f'<div class="sub">{escape(sub)}</div>' if sub else ""
    return (
        f'<div class="{cls}">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>'
        f'{sub_html}'
        f'</div>'
    )


def download_button(excel_path: str | None, variety: str) -> str:
    """Boton de descarga del Excel ejecutivo. Disabled si no hay archivo."""
    if not excel_path:
        return (
            '<a class="btn-download disabled">'
            '<span class="icon">⬇</span>'
            '<span><span class="label-main">Excel no disponible</span>'
            '<span class="label-sub">faltan columnas KG/JR · H-EF</span></span>'
            '</a>'
        )
    rel = Path(excel_path).name
    return (
        f'<a class="btn-download" href="{escape(rel)}" download>'
        f'<span class="icon">⬇</span>'
        f'<span><span class="label-main">Descargar Excel completo</span>'
        f'<span class="label-sub">{escape(rel)} · variedad {escape(variety)}</span></span>'
        f'</a>'
    )
