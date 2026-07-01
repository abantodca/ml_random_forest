"""Reconcilia el backend de MLflow con los artifacts huerfanos en S3.

PROBLEMA QUE RESUELVE
---------------------
El backend de MLflow (Postgres) es la UNICA fuente de verdad de QUE runs y
modelos existen; S3 solo guarda los blobs (`pipeline.joblib`, JSONs, HTMLs)
indexados por `artifacts/<exp_id>/<run_id>/...`. MLflow NUNCA escanea S3 para
descubrir runs. Si el volumen `pg-data` se borra (p.ej. `docker compose down -v`),
se pierde TODO el metadata de runs/experimentos/Model Registry aunque los
artifacts sigan intactos en S3 -> el servidor muestra cero modelos y la API se
queda sin `rnd-forest-<variety>` que cargar.

QUE HACE (idempotente, "universal")
------------------------------------
- Arranca-desde-cero: el backend esta vacio pero S3 tiene runs -> reconstruye
  una sola vez el run campeon de cada variedad (metricas + params + artifacts)
  y registra `rnd-forest-<variety>` en el Model Registry.
- Ya-poblado: si la variedad ya tiene su modelo registrado -> NO hace nada.
  Los entrenamientos nuevos se registran como siempre (este script no estorba).

Asi se puede cablear como paso de arranque (one-shot, ver `mlflow-init` en
docker-compose): corre tras `mlflow` healthy y antes de `api`; es no-op barato
cuando ya hay modelos.

FIDELIDAD
---------
Las metricas/params/tags nativas del run vivian en Postgres y se perdieron; se
reconstruyen desde los JSON que el trainer dejo EN los artifacts
(`run_summary_*.json`, `champion_<variety>.json`). Es lo mejor recuperable sin
re-entrenar. Los runs reconstruidos llevan tag `reconciled_from_run_id` con el
run_id original en S3.

Corre en la imagen `mlflow` (mlflow client + boto3 + creds ~/.aws montadas).
Best-effort: un fallo por variedad se loguea y NO aborta el resto ni el arranque.

Uso:
  python scripts/mlflow_reconcile.py            # reconcilia lo que falte
  python scripts/mlflow_reconcile.py --dry-run  # solo reporta, no escribe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import boto3
import mlflow
from mlflow.tracking import MlflowClient

# Prefijo raiz dentro del bucket (== `--artifacts-destination s3://<bucket>/artifacts`).
ARTIFACTS_ROOT = "artifacts"


def log(msg: str) -> None:
    print(f"[mlflow-reconcile] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Resolucion de entorno
# ---------------------------------------------------------------------------
def resolve_config() -> tuple[str, str, str, str]:
    """Devuelve (tracking_uri, bucket, experiment_prefix, registry_prefix)."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    bucket = os.environ.get("S3_MLFLOW_BUCKET") or os.environ.get("S3_ARTIFACTS_BUCKET")
    if not bucket:
        log("ERROR: falta S3_MLFLOW_BUCKET/S3_ARTIFACTS_BUCKET; nada que reconciliar.")
        sys.exit(0)  # best-effort: no bloquear el arranque
    # Espejan al trainer: experimento = f"{MLFLOW_EXPERIMENT_PREFIX}{variety}",
    # modelo registrado = f"{MODEL_REGISTRY_PREFIX}{variety}".
    experiment_prefix = os.environ.get("MLFLOW_EXPERIMENT_PREFIX", "")
    registry_prefix = os.environ.get("MODEL_REGISTRY_PREFIX", "rnd-forest-")
    return tracking_uri, bucket, experiment_prefix, registry_prefix


# ---------------------------------------------------------------------------
# Helpers S3
# ---------------------------------------------------------------------------
def s3_list(s3, bucket: str, prefix: str) -> list[dict]:
    """Lista TODOS los objetos bajo `prefix` (paginado)."""
    out: list[dict] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        out.extend(resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


def s3_get_json(s3, bucket: str, key: str) -> dict:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def s3_download_prefix(s3, bucket: str, prefix: str, dest: Path) -> int:
    """Baja todos los objetos bajo `prefix` a `dest`, preservando estructura
    relativa al prefix. Devuelve cuantos archivos bajo."""
    n = 0
    for obj in s3_list(s3, bucket, prefix):
        key = obj["Key"]
        if key.endswith("/"):
            continue
        rel = key[len(prefix) :].lstrip("/")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(target))
        n += 1
    return n


# ---------------------------------------------------------------------------
# Descubrimiento de campeones en S3
# ---------------------------------------------------------------------------
def discover_champions(s3, bucket: str) -> dict[str, dict]:
    """Encuentra el champion_<variety>.json mas reciente por variedad.

    Devuelve {variety: {"exp_id", "run_id", "model", "champion_key"}} donde
    run_id es el `champion_run_id` (el run que tiene el pipeline + winner
    dashboard), NO necesariamente el run donde quedo el JSON.
    """
    objs = s3_list(s3, bucket, f"{ARTIFACTS_ROOT}/")
    champ_keys = [
        o for o in objs if "/champion/champion_" in o["Key"] and o["Key"].endswith(".json")
    ]
    by_variety: dict[str, dict] = {}
    for o in sorted(champ_keys, key=lambda x: x["LastModified"]):
        key = o["Key"]
        # artifacts/<exp_id>/<run_id>/artifacts/champion/champion_<VAR>.json
        parts = key.split("/")
        exp_id = parts[1]
        variety = parts[-1][len("champion_") : -len(".json")]
        try:
            doc = s3_get_json(s3, bucket, key)
        except Exception as exc:
            log(f"  aviso: no pude leer {key}: {exc}")
            continue
        run_id = doc.get("champion_run_id")
        if not run_id:
            continue
        # El sorted() asc por fecha deja el ultimo (mas reciente) como ganador.
        by_variety[variety] = {
            "exp_id": exp_id,
            "run_id": run_id,
            "model": doc.get("champion_model", "unknown"),
            "champion_key": key,
        }
    return by_variety


def find_logged_model_prefix(s3, bucket: str, exp_id: str, run_id: str) -> str | None:
    """Localiza el Logged Model (`models/m-<id>/artifacts/`) cuyo MLmodel
    referencia `run_id`. Devuelve el prefix S3 de su carpeta artifacts o None."""
    base = f"{ARTIFACTS_ROOT}/{exp_id}/models/"
    mlmodels = [o["Key"] for o in s3_list(s3, bucket, base) if o["Key"].endswith("/MLmodel")]
    for key in mlmodels:
        try:
            text = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        except Exception:
            continue
        if f"run_id: {run_id}" in text:
            return key.rsplit("/MLmodel", 1)[0] + "/"  # .../m-<id>/artifacts/
    return None


# ---------------------------------------------------------------------------
# Mapeo de metricas: run_summary JSON -> nombres que la API espera
# ---------------------------------------------------------------------------
def build_metrics(summary: dict) -> dict[str, float]:
    """Aplana el run_summary a los nombres de metrica que lee la API
    (`api/app/services/mlflow_service.py`): nested_cv_*, business_oof_mape,
    full_model_r2, business_insample_mape. Loguea un superset (inocuo)."""
    m: dict[str, float] = {}
    for k, v in (summary.get("metrics") or {}).items():
        if isinstance(v, (int, float)):
            m[k] = float(v)
    bo = summary.get("business_metrics_oof") or {}
    for k, v in bo.items():
        if isinstance(v, (int, float)):
            m[f"business_oof_{k}"] = float(v)  # -> business_oof_mape, etc.
    fm = summary.get("full_metrics_model") or {}
    for k, v in fm.items():
        if isinstance(v, (int, float)):
            m[f"full_model_{k}"] = float(v)  # -> full_model_r2, etc.
    fb = summary.get("full_metrics_business") or {}
    for k, v in fb.items():
        if isinstance(v, (int, float)):
            m[f"full_business_{k}"] = float(v)
    # Alias que la API lee como "train_mape" in-sample.
    if "mape" in fb and isinstance(fb["mape"], (int, float)):
        m["business_insample_mape"] = float(fb["mape"])
    return m


def build_params(summary: dict) -> dict[str, str]:
    params = {"model_type": str(summary.get("model_type", "unknown"))}
    for k, v in (summary.get("best_params") or {}).items():
        params[str(k)] = str(v)
    return params


# ---------------------------------------------------------------------------
# Reconstruccion de una variedad
# ---------------------------------------------------------------------------
def already_registered(client: MlflowClient, model_name: str) -> bool:
    try:
        versions = client.search_model_versions(f"name='{model_name}'", max_results=1)
        return len(versions) > 0
    except Exception:
        return False


def reconstruct_variety(
    s3,
    bucket: str,
    variety: str,
    info: dict,
    experiment_prefix: str,
    model_name: str,
    dry_run: bool,
) -> bool:
    exp_id, run_id = info["exp_id"], info["run_id"]
    run_prefix = f"{ARTIFACTS_ROOT}/{exp_id}/{run_id}/artifacts/"

    objs = s3_list(s3, bucket, run_prefix)
    if not objs:
        log(f"  {variety}: sin artifacts en {run_prefix}; salto.")
        return False

    # run_summary (metricas/params) — el del modelo campeon.
    summary_key = next(
        (o["Key"] for o in objs if "/run_summary_" in o["Key"] and o["Key"].endswith(".json")),
        None,
    )
    summary = s3_get_json(s3, bucket, summary_key) if summary_key else {}
    metrics = build_metrics(summary)
    params = build_params(summary)
    params.setdefault("model_type", info["model"])

    model_prefix = find_logged_model_prefix(s3, bucket, exp_id, run_id)

    log(
        f"  {variety}: reconstruyendo desde run {run_id[:12]} "
        f"(model={params['model_type']}, {len(metrics)} metricas, "
        f"modelo={'si' if model_prefix else 'NO ENCONTRADO'})"
    )
    if dry_run:
        return True

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        run_dir = tmpdir / "run"
        n_run = s3_download_prefix(s3, bucket, run_prefix, run_dir)
        model_dir = tmpdir / "model"
        n_model = s3_download_prefix(s3, bucket, model_prefix, model_dir) if model_prefix else 0
        log(f"    descargados {n_run} artifacts del run + {n_model} del modelo")

        mlflow.set_experiment(f"{experiment_prefix}{variety}")
        with mlflow.start_run(run_name=f"reconciled_{variety}") as run:
            new_run_id = run.info.run_id
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "variety": variety,
                    "model_type": params["model_type"],
                    "is_champion": "true",
                    "source": "s3_reconcile",
                    "reconciled_from_run_id": run_id,
                }
            )
            # Re-sube los artifacts del run original (incluye winner_dashboard/,
            # que la API lista para servir el reporte HTML).
            mlflow.log_artifacts(str(run_dir))

            model_uri = None
            if n_model:
                # Re-loguea el Logged Model como artifact del run nuevo; queda
                # como pyfunc+sklearn cargable por la API (que importa `src`).
                mlflow.log_artifacts(str(model_dir), artifact_path="model_pipeline")
                model_uri = f"runs:/{new_run_id}/model_pipeline"

        if model_uri is None:
            log(f"  {variety}: run reconstruido pero SIN modelo registrable.")
            return False

        register_model(model_name, model_uri, variety, metrics, run_id)
        return True


def register_model(
    model_name: str,
    model_uri: str,
    variety: str,
    metrics: dict[str, float],
    orig_run_id: str,
) -> None:
    """Registra y etiqueta la version (espeja src/step_06_track/mlflow_registry)."""
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient()
    description = (
        f"Modelo de productividad para variedad '{variety}' "
        f"(RECONSTRUIDO desde S3, run original {orig_run_id}).\n"
        f"R2 (Nested CV): {metrics.get('nested_cv_r2_mean', float('nan')):.4f}  |  "
        f"MAE test: {metrics.get('nested_cv_mae_mean', float('nan')):.4f}"
    )
    client.update_model_version(model_name, mv.version, description=description)
    tags = {
        "variety": variety,
        "r2_mean": f"{metrics.get('nested_cv_r2_mean', float('nan')):.4f}",
        "mae_test_mean": f"{metrics.get('nested_cv_mae_mean', float('nan')):.4f}",
        "reconciled": "true",
        "reconciled_from_run_id": orig_run_id,
    }
    for k, v in tags.items():
        client.set_model_version_tag(model_name, mv.version, k, v)
    log(f"  {variety}: registrado {model_name} v{mv.version} ✅")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="solo reporta, no escribe")
    args = p.parse_args()

    tracking_uri, bucket, experiment_prefix, registry_prefix = resolve_config()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    log(f"tracking={tracking_uri}  bucket={bucket}  registry_prefix={registry_prefix}")
    s3 = boto3.client("s3")

    champions = discover_champions(s3, bucket)
    if not champions:
        log("No hay champion_<variety>.json en S3; nada que reconciliar.")
        return
    log(f"Variedades con campeon en S3: {sorted(champions)}")

    reconciled, skipped, failed = 0, 0, 0
    for variety, info in sorted(champions.items()):
        model_name = f"{registry_prefix}{variety}".strip("_")
        if already_registered(client, model_name):
            log(f"  {variety}: {model_name} ya registrado -> skip (idempotente).")
            skipped += 1
            continue
        try:
            ok = reconstruct_variety(
                s3, bucket, variety, info, experiment_prefix, model_name, args.dry_run
            )
            reconciled += int(ok)
            failed += int(not ok)
        except Exception as exc:
            failed += 1
            log(f"  {variety}: FALLO reconstruccion: {exc}")

    verb = "reconciliaria" if args.dry_run else "reconcilio"
    log(f"Resumen: {verb} {reconciled} | skip {skipped} | fallidas {failed}")


if __name__ == "__main__":
    main()
