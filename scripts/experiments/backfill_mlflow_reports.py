# Backfill puntual (2026-06-13): sube los reportes HTML/Excel del run campeón
# prod al MLflow server, que en su momento no pudo recibirlos (bucket S3
# inexistente + mount de credenciales vacío — ambos ya corregidos).
# Uso: docker compose run --rm --entrypoint python trainer artifacts/backfill_mlflow_reports.py
import os

import mlflow
from mlflow.tracking import MlflowClient

CHAMPION_RUN = "1ed787bed358457ca79c3998bcfd36f6"  # run prod LGB 14.27% OOF
FILES = [
    "/app/reports/Winner_POP_2026-06-11_17-20-49.html",
    "/app/reports/Winner_POP_2026-06-11_17-20-49.xlsx",
    "/app/reports/PREVIEW_redesign.html",  # mismo run, rediseño gerencial 2026-06-12
    "/app/artifacts/PROD_GRILLA_NUEVA_2026-06-11.json",
]

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

# Run de registro del modelo (rnd-forest-POP v1) — también recibe los reportes
# para que sean visibles desde el Model Registry.
reg_runs = [
    v.run_id for v in client.search_model_versions("name='rnd-forest-POP'") if v.version == "1"
]
targets = [CHAMPION_RUN] + [r for r in reg_runs if r != CHAMPION_RUN]

for run_id in targets:
    for path in FILES:
        if not os.path.exists(path):
            print(f"  AVISO: no existe {path}, omitido")
            continue
        client.log_artifact(run_id, path, artifact_path="reports")
        print(f"  subido {os.path.basename(path)} -> run {run_id[:12]}")

print("\nVerificación:")
for run_id in targets:
    arts = client.list_artifacts(run_id, "reports")
    print(f"run {run_id[:12]} reports/: {[a.path for a in arts]}")
