"""Diagnostic / EDA module — runs INDEPENDIENTEMENTE del pipeline de training.

Produce un reporte HTML por variedad con tests estadisticos sobre la data
RAW (post-load_data, pre-pipeline). Se ejecuta via:

    task eda VARIETIES=POP

El HTML se sube como artifact a MLflow si hay un run activo, o se guarda
solo en `reports/EDA_<variety>_<timestamp>.html` para inspeccion local.

Modulos:
    statistical_tests : wrappers tipados sobre statsmodels/scipy
    distributions     : analisis univariado por variable
    temporal          : autocorrelacion, estacionariedad, drift por anio
    multivariate      : VIF, mutual information, correlacion
    plots             : plotly figures reutilizables
    html_renderer     : ensamblaje del reporte HTML
    eda               : entrypoint `run_eda(variety) -> Path`
"""
from src.diagnostics.eda import run_eda  # noqa: F401

__all__ = ["run_eda"]
