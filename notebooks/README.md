# notebooks/

Esta carpeta contiene **experimentos exploratorios históricos**, no son
parte del pipeline de producción. El flujo en producción vive en `src/` +
`main.py` y se ejecuta vía `task train`.

Si vas a editar el modelo, NO trabajes desde estos notebooks: hacelo en
`src/step_04_train/` o `src/step_03_features/`. Los notebooks NO se
ejecutan en CI ni en el entrenamiento automatizado, pueden tener código
desactualizado o que no compile contra la versión actual del proyecto.

## Inventario

- **`experiment_cluster_varieties_new.ipynb`** — Análisis exploratorio
  inicial: clustering de variedades para ver si comparten patrones de
  productividad. Conclusión histórica: cada variedad entrena su propio
  modelo (decisión actual del proyecto).

- **`RFR_POP.ipynb`** — Prototipado original de Random Forest para POP
  antes de migrar a XGBoost / LightGBM con Optuna. Mantenido como
  referencia de la baseline.

## Cuándo borrar

Si en una próxima limpieza ya no necesitás el contexto histórico,
podés borrar la carpeta entera. Nada en `src/`, `scripts/`, `Taskfile.yml`
ni `main.py` los importa.
