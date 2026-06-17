# Limpieza de MLflow (2026-06-13, pedido del usuario: "limpia todo mi mlflow,
# quedar con el modelo campeon"). Conserva SOLO el linaje del campeon:
#   - run 1ed787be... : entrenamiento prod del campeon (LGB 14.27% OOF),
#     con Winner HTML/Excel/PREVIEW backfilleados.
#   - run de registro de rnd-forest-POP v1 (source del modelo servido).
#   - run eda_POP_2026-06-12_14-08 (EDA vigente, HTML+JSON).
# Borra (soft-delete; el purge fisico lo hace `mlflow gc` despues):
#   - todo otro run del experimento POP (smokes, exante, iteraciones),
#   - el experimento S3_SMOKE completo.
# Uso:
#   docker compose run --rm --entrypoint sh trainer -c \
#     "PYTHONPATH=/app python artifacts/mlflow_cleanup.py"
import os

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

KEEP_RUNS = {
    "1ed787bed358457ca79c3998bcfd36f6",  # campeon prod LGB
    "391e604b0c0844438ed902dcb5d3134b",  # EDA vigente
}
# El run de registro: lo resolvemos del registered model (no hardcodear).
for v in client.search_model_versions("name='rnd-forest-POP'"):
    KEEP_RUNS.add(v.run_id)
    print(f"conservar (registro v{v.version}): {v.run_id}")

borrados = 0
for exp in client.search_experiments():
    if exp.name == "S3_SMOKE":
        client.delete_experiment(exp.experiment_id)
        print(f"experimento eliminado: {exp.name}")
        continue
    if exp.name == "Default":
        continue
    runs = client.search_runs(
        [exp.experiment_id], max_results=1000, run_view_type=1  # ACTIVE_ONLY
    )
    for r in runs:
        if r.info.run_id in KEEP_RUNS:
            print(f"conservar: {r.info.run_id[:12]} ({r.info.run_name})")
            continue
        client.delete_run(r.info.run_id)
        borrados += 1
        print(f"eliminado: {r.info.run_id[:12]} ({r.info.run_name})")

print(f"\nTotal runs eliminados (soft): {borrados}")
print("Siguiente paso: mlflow gc en el contenedor mlflow para purgar "
      "filas de Postgres y artifacts de S3.")
