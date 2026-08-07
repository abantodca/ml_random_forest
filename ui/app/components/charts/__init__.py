"""
app.components.charts - Factory de figuras Plotly (funciones puras).

`_comun.py` queda fuera del barrel: es uso interno del paquete (layout
y constantes compartidas entre los gráficos).
"""

from app.components.charts.bar_table import build_table_with_bars
from app.components.charts.gauge import build_drift_gauge, build_simple_gauge
from app.components.charts.heatmap import build_metrics_heatmap
from app.components.charts.histogram import build_kghora_histogram
from app.components.charts.tracking import (
    build_decomp_scatter,
    build_parity_plot,
    build_pred_vs_real_line,
    build_residual_bars,
    build_weekly_bars,
)

__all__ = [
    "build_kghora_histogram",
    "build_metrics_heatmap",
    "build_drift_gauge",
    "build_simple_gauge",
    "build_table_with_bars",
    "build_pred_vs_real_line",
    "build_parity_plot",
    "build_residual_bars",
    "build_decomp_scatter",
    "build_weekly_bars",
]
