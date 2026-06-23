"""Dashboard ejecutivo HTML — package publico.

Estructura interna:
    styles.py      - CSS + plotly.js bundle.
    helpers.py     - fmt(), kpi_card(), download_button().
    sections.py    - Hero, Context, Mega KPIs, Guide, Actions.
    technical.py   - Seccion 'Detalle tecnico' (colapsable, DS-grade).
    winner_dashboard.py - Orquestador thin (arma el kit + ensambla).

API publica:
    render_winner_dashboard - genera reports/Winner_{variety}.html
"""

from src.step_05_evaluate.html.winner_dashboard import render_winner_dashboard

__all__ = ["render_winner_dashboard"]
