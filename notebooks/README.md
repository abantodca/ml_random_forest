# notebooks/

Experimentos exploratorios. **No** son parte del pipeline de producción
(`src/` + `main.py`, vía `task train`) y **no** se ejecutan en CI.

Si vas a editar el modelo de producción, hacelo en `src/step_04_train/` o
`src/step_03_features/`, **no** desde estos notebooks.

## Inventario

- **`experiment_variety_anchor_routing.ipynb`** — decide qué variedades
  entrenan modelo propio (**anclas**) y a cuál ancla rutear las de poca data.
  - No es clustering: la data no tiene estructura de clusters natural
    (silhouette negativo a todo k; los clusters reales están en k=2-3). Es
    **"asignar al donante más cercano"**.
  - Validación **robusta**: tamaño de efecto (Cliff's delta) en vez de
    p-valores (que saturan a n grande), bootstrap stability, Holm-Bonferroni,
    silhouette sobre todas las observaciones.
  - **Sección 6 — tablero de ruteo predictivo:** la decisión accionable.
    Rutea cada variedad al ancla cuyo modelo la **pronostica** mejor (menor
    MAPE OOS), no a la más parecida por distribución. No es clustering.
  - Salida: `variety_predictive_routing.csv/.yaml` (mapping `variedad →
    ancla`) + `decision_table.html` (tablero) en `../data/`.
- **`variety_routing.py`** — módulo con la lógica estable del notebook
  (config, distancias, validación, estadística). Importable y testeable.

## Cómo correrlo

```bash
# entorno con numpy/pandas/scipy/scikit-learn/statsmodels/openpyxl
cd notebooks
jupyter nbconvert --to notebook --execute --inplace experiment_variety_anchor_routing.ipynb
```

El mapping generado se enchufa al pipeline real; el notebook **no** toca
`src/`, `api/` ni `ui/`.
