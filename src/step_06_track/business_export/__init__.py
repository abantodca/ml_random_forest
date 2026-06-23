"""Exportador de Excel ejecutivo (submodulo).

Genera un workbook multi-hoja con la decision del modelo en lenguaje
natural. 8 hojas (orden de lectura):

  1. Inicio              Portada con veredicto + 3 KPIs en lenguaje natural
                         + indice de hojas. Una persona no-tecnica entiende
                         el resultado leyendo SOLO esta hoja.
  2. Acciones            Recomendaciones auto-generadas desde el analisis
                         (subgrupos problematicos, overfitting, etc.).
  3. Resumen             Metricas tecnicas globales + estado vs targets.
  4. Por_FORMATO         Agregado por FORMATO (n, error, ranking peor->mejor).
  5. Por_FUNDO           Idem por FUNDO.
  6. Predicciones_OOF    Detalle fila a fila con OOF (honesto). Hoja
                         operacional para filtros + pivots del negocio.
  7. Predicciones_Total  Idem con el modelo aplicado a TODA la data
                         (in-sample, sesgado/optimista — sanity check).
                         Incluye banda de confianza al 95% (mean +/- 1.96*std)
                         si el modelo es OOFEnsembleRegressor con K>=2.
  8. Glosario            Diccionario de terminos tecnicos en lenguaje claro.

Submodulos:
  - builders.py    : `build_*_df` (hojas como DataFrames puros).
  - formatting.py  : `apply_formatting` (estilos openpyxl + paletas color).
  - export.py      : `export_business_excel` (orquestador).

Re-export del API publico (`export_business_excel`) para que los consumidores
sigan importando `from src.step_06_track.business_export import export_business_excel`
sin cambios.
"""

from src.step_06_track.business_export.export import export_business_excel

__all__ = ["export_business_excel"]
