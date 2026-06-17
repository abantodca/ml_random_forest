"""Wrappers thin sobre MLflow para mantener `main.py` legible.

Backend unico: MLflow server con Postgres (metadata + Registry) + S3
(artifacts). En local lo sirve `docker compose up`. En produccion el
mismo URI apunta al server real. Model Registry siempre habilitado
(Postgres lo soporta nativamente; el viejo modo file://mlruns no).
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo

from src.config import MLFLOW_TRACKING_URI, MODEL_REGISTRY_PREFIX

_logger = logging.getLogger(__name__)


def _is_inactive_run_error(exc: MlflowException) -> bool:
    """True si la excepcion proviene de operar sobre un run no-activo.

    Pasa cuando el run fue marcado como `deleted` externamente (UI de MLflow,
    cliente paralelo, soft-delete via API) entre `start_run()` y la llamada
    de logging. El mensaje de MLflow incluye 'active' lifecycle_stage.
    """
    msg = str(exc).lower()
    return "lifecycle_stage" in msg or "must be in 'active'" in msg


@contextmanager
def safe_start_run(run_name: str) -> Iterator[mlflow.ActiveRun]:
    """Context manager equivalente a `mlflow.start_run` pero no aborta si el
    run queda inactivo durante el training.

    El `with mlflow.start_run()` nativo llama `set_terminated` en `__exit__`,
    y si el run fue marcado como `deleted` externamente (UI de MLflow,
    cliente paralelo) ese set_terminated lanza MlflowException y derriba
    el train_model entero — perdiendo el modelo ya entrenado.

    Aqui hacemos start manual, dejamos que el cuerpo corra, y al salir
    intentamos `end_run()`; si falla por lifecycle_stage, lo absorbemos
    (warning) en vez de propagar.
    """
    run = mlflow.start_run(run_name=run_name)
    try:
        yield run
    finally:
        try:
            mlflow.end_run()
        except MlflowException as exc:
            if _is_inactive_run_error(exc):
                _logger.warning(
                    "MLflow end_run ignorado: run inactivo (lifecycle_stage). "
                    "El modelo se persistio localmente; sigue el pipeline."
                )
            else:
                raise


def _safe_mlflow_call(fn: Callable, op_name: str) -> Any:
    """Invoca `fn()` y absorbe MlflowException por run inactivo con un warning.

    Devuelve lo que devuelva `fn()` (o `None` si el run quedo inactivo).
    Cualquier otro MlflowException PROPAGA (auth/red/schema son errores reales).
    El objetivo es que un run borrado externamente no aborte un training de
    1+ hora. El pipeline ya esta entrenado para cuando se loguea, asi que
    perderlo solo porque MLflow no acepta un log_metric es desproporcionado.
    """
    try:
        return fn()
    except MlflowException as exc:
        if _is_inactive_run_error(exc):
            _logger.warning(
                "MLflow %s ignorado: el run quedo inactivo (lifecycle_stage). "
                "El modelo ya esta entrenado y persistido localmente.",
                op_name,
            )
            return None
        raise


def init_mlflow(
    experiment_name: str | None = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> None:
    """Configura backend (local o remoto). Setea experimento si se provee.

    En multi-variedad NO se pasa experiment_name aqui: cada `train_one_variety`
    setea su propio experimento dinamicamente con `mlflow.set_experiment(...)`.
    """
    mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)


def set_experiment(experiment_name: str) -> None:
    """Wrapper sobre `mlflow.set_experiment`.

    El `artifact_location` lo decide el server (modo proxy: --serve-artifacts
    + --artifacts-destination s3://<bucket>/artifacts; bucket real en prod).
    El cliente NO debe pasar artifact_location: si lo hace, MLflow usaria un
    path del cliente que el server no puede leer.
    """
    mlflow.set_experiment(experiment_name)


def next_run_version(experiment_name: str, model_type: str) -> int:
    """Devuelve la siguiente version (1-indexed) para el (experimento, modelo).

    Cuenta los runs existentes con tag `model_type=<model_type>` en el
    experimento dado y devuelve `n + 1`. Permite que cada entrenamiento
    de la misma variedad+modelo aparezca con un run_name versionado
    (e.g. `xgb_v1`, `xgb_v2`, ...) en lugar de timestamps.

    Si el experimento no existe o el backend no soporta search, devuelve 1.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return 1
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.model_type = '{model_type}'",
            max_results=1000,
        )
        return len(runs) + 1
    except Exception:
        return 1


def log_metrics(metrics: dict[str, float]) -> None:
    safe = {k: float(v) for k, v in metrics.items() if v is not None}
    if safe:
        _safe_mlflow_call(lambda: mlflow.log_metrics(safe), "log_metrics")


def log_business_metrics(business_validation) -> None:
    """Loguea las metricas de negocio (KG/JR) en MLflow.

    Mete tags resumen 'business_oof_r2' y 'business_oof_mae' para que sean
    filtrables en la UI de MLflow sin abrir el run. Usa las metricas OOF
    (no las in-sample) porque son las que reflejan el rendimiento real.
    """
    if business_validation.is_empty():
        return
    metrics = business_validation.to_mlflow_metrics()
    log_metrics(metrics)
    # Tags resumen filtrables
    summary_tags: dict[str, str] = {}
    if "r2" in business_validation.metrics_oof:
        summary_tags["business_oof_r2"] = f"{business_validation.metrics_oof['r2']:.4f}"
    if "mae" in business_validation.metrics_oof:
        summary_tags["business_oof_mae"] = f"{business_validation.metrics_oof['mae']:.4f}"
    if "mape" in business_validation.metrics_oof:
        summary_tags["business_oof_mape"] = f"{business_validation.metrics_oof['mape']:.2f}"
    if summary_tags:
        set_tags(summary_tags)


def log_params(params: dict[str, object]) -> None:
    safe = {k: v for k, v in params.items() if v is not None}
    if safe:
        _safe_mlflow_call(lambda: mlflow.log_params(safe), "log_params")


def _safe_artifact_call(fn: Callable, op_name: str) -> Any:
    """Como `_safe_mlflow_call`, pero absorbe CUALQUIER MlflowException.

    Solo para operaciones de ARTIFACTS (log_artifact / log_pipeline). Un
    upload de artifact que falla (S3 caido, credenciales vencidas, 500 del
    proxy mlflow-artifacts) nunca justifica abortar un training de 1+ hora:
    el pipeline, el OOF y los summaries ya estan persistidos en disco local
    ANTES de tocar MLflow (ver single_run._persist_pipeline_and_oof_locally).
    Las operaciones de metrics/params/tags siguen usando `_safe_mlflow_call`
    (ahi un fallo de auth/schema si es un error real que debe propagar).
    """
    try:
        return fn()
    except MlflowException as exc:
        _logger.warning(
            "MLflow %s fallo (artifact no subido; el archivo sigue en disco "
            "local): %s",
            op_name,
            exc,
        )
        return None


def log_pipeline(
    pipeline,
    name: str = "model_pipeline",
    X_sample=None,
    y_sample=None,
) -> ModelInfo | None:
    """Loguea el pipeline como artifact con signature inferida (si se da X/y).

    Inferir signature + input_example hace que el modelo sea autodescribible
    al deployearlo (mlflow models serve, sagemaker, etc.) y silencia el
    warning "Model logged without a signature and input example".

    Devuelve el `ModelInfo` que retorna `mlflow.sklearn.log_model` (incluye
    `model_uri` con el formato nuevo `models:/m-<id>` que MLflow 3.x usa para
    Logged Models). El caller lo necesita para `register_model` y evitar el
    fallback "no artifacts at artifact path 'model_pipeline'" del runtime.
    Devuelve `None` si el run quedo inactivo (manejado por _safe_mlflow_call).

    `name=` (kwarg, no posicional) suprime la deprecation de MLflow 3.x del
    alias `artifact_path` cuando se pasa como segundo posicional.
    """
    kwargs: dict[str, object] = {}
    if X_sample is not None and y_sample is not None:
        try:
            kwargs["signature"] = infer_signature(X_sample, y_sample)
            kwargs["input_example"] = (
                X_sample.head(3) if hasattr(X_sample, "head") else X_sample[:3]
            )
        except Exception:
            # signature no es critico; preferimos loguear que abortar
            pass
    return _safe_artifact_call(
        lambda: mlflow.sklearn.log_model(pipeline, name=name, **kwargs),
        "log_pipeline",
    )


def set_tags(tags: dict[str, object]) -> None:
    """MLflow exige tags como str; se castea defensivamente."""
    safe = {k: ("" if v is None else str(v)) for k, v in tags.items()}
    if safe:
        _safe_mlflow_call(lambda: mlflow.set_tags(safe), "set_tags")


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    _safe_artifact_call(
        lambda: mlflow.log_artifact(str(path), artifact_path=artifact_path),
        "log_artifact",
    )


def register_model(
    run_id: str,
    artifact_name: str = "model_pipeline",
    variety: str = "default",
    stage: str | None = None,
    metrics: dict[str, float] | None = None,
    report_artifact_uri: str | None = None,
    extra_tags: dict[str, object] | None = None,
    model_uri: str | None = None,
) -> str | None:
    """Registra el modelo del run en MLflow Model Registry con metadata rica.

    Devuelve `nombre@version` o None si el backend es file:// (caso de
    emergencia: ADR-001 lo prohibe en operacion normal pero el guard se
    mantiene como red de seguridad).

    `model_uri` (opcional): URI del Logged Model devuelto por
    `mlflow.sklearn.log_model(...).model_uri` (formato `models:/m-<id>` en
    MLflow 3.x). Si se provee, se usa directamente — evita el warning
    "Run has no artifacts at artifact path '<artifact_name>'" que MLflow 3.x
    emite cuando intenta resolver el viejo URI `runs:/{run_id}/{artifact_name}`
    (los Logged Models ya no viven debajo del run, son entidades aparte).

    Si `model_uri` es None, cae al URI legacy por compatibilidad con runs
    historicos pre-3.x.

    Adicionalmente:
    - Setea description del Model Version con un resumen humano
      (R2, MAE_test, gap, link al reporte HTML).
    - Setea tags clave en la version (filtrables desde la UI).
    - Si `stage` esta en {Staging, Production}, transiciona y archiva las
      versiones anteriores en Production.

    Cualquier MlflowException propaga al caller (auth/red/schema son fallos
    reales que deben verse, no enmascararse).
    """
    model_name = f"{MODEL_REGISTRY_PREFIX}{variety}".strip("_")
    if model_uri is None:
        model_uri = f"runs:/{run_id}/{artifact_name}"
    is_file_backend = mlflow.get_tracking_uri().startswith("file:")
    try:
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        client = mlflow.tracking.MlflowClient()

        # ---- Description humana ----
        m = metrics or {}
        description = (
            f"Modelo de productividad para variedad '{variety}'.\n"
            f"R2 (Nested CV): {m.get('nested_cv_r2_mean', float('nan')):.4f}\n"
            f"MAE test: {m.get('nested_cv_mae_mean', float('nan')):.4f}  |  "
            f"MAE train: {m.get('nested_cv_mae_train_mean', float('nan')):.4f}  |  "
            f"gap: {m.get('nested_cv_gap_mean', float('nan')):+.4f}\n"
        )
        if report_artifact_uri:
            description += f"Reporte gerencial: {report_artifact_uri}"
        client.update_model_version(
            name=model_name,
            version=mv.version,
            description=description,
        )

        # ---- Tags en la version (filtrables) ----
        tags: dict[str, str] = {
            "variety": variety,
            "r2_mean": f"{m.get('nested_cv_r2_mean', float('nan')):.4f}",
            "mae_test_mean": f"{m.get('nested_cv_mae_mean', float('nan')):.4f}",
            "mae_train_mean": f"{m.get('nested_cv_mae_train_mean', float('nan')):.4f}",
            "overfit_gap": f"{m.get('nested_cv_gap_mean', float('nan')):+.4f}",
            "run_id": run_id,
        }
        if report_artifact_uri:
            tags["report_html"] = report_artifact_uri
        if extra_tags:
            tags.update({k: str(v) for k, v in extra_tags.items()})
        for k, v in tags.items():
            client.set_model_version_tag(model_name, mv.version, k, v)

        # ---- Promocion via alias (MLflow 3.x) ----
        # `transition_model_version_stage` fue deprecado en 2.9 y eliminado
        # de la API publica en 3.x. `set_registered_model_alias` reasigna el
        # alias atomicamente: la version anterior pierde el alias automatico
        # (mismo efecto que `archive_existing_versions` del flujo legacy).
        # "Archived" en el modelo aliases = no setear alias activo.
        if stage in ("Staging", "Production"):
            client.set_registered_model_alias(
                name=model_name,
                alias=stage.lower(),
                version=mv.version,
            )

        return f"{model_name} v{mv.version}"
    except mlflow.exceptions.MlflowException:
        # file://: no soporta Registry, retorno silencioso esperado (caso
        # local-only default). Backend SQL: error real (auth/red/schema)
        # -> propagar para que variety_runner lo capture y logue.
        if is_file_backend:
            return None
        raise
