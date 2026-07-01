"""Warm-start de Optuna desde el campeon ya registrado (2026-06-25).

Lee los `best_params` del modelo `rnd-forest-<variety>` que ya vive en el
MLflow Model Registry y los devuelve listos para `study.enqueue_trial(...)`.
Asi el tuning ARRANCA desde la configuracion que ya funciona y el resto de
los trials solo busca MEJORARLA, en vez de re-explorar a ciegas. Es el
mecanismo concreto de "usar lo ya entrenado para mejorar lo que tenemos".

Si no hay con que sembrar (variedad nueva, registry vacio, campeon de otro
backend, o cualquier fallo) devuelve None y el caller arranca en frio.

Decisiones de ROBUSTEZ y SEGURIDAD (lo que lo hace seguro):
  - Lee SOLO `run.data.params` (strings planos). NUNCA descarga ni
    deserializa el `.joblib` del modelo -> cero superficie de ejecucion de
    codigo arbitrario por pickle (el propio MLflow advierte de ese riesgo
    al guardar sklearn). El warm-start no toca artefactos binarios.
  - TODO va envuelto en try/except -> ante red caida, auth vencida, registry
    file://, parse roto, etc. se absorbe con warning y se arranca en frio.
    El warm-start jamas rompe un training (mismo contrato que
    _temporal_honesty_check).
  - Solo siembra si el campeon previo es del MISMO backend (signature-check
    por una clave exclusiva del espacio). Evita meter params de lgb en un
    estudio xgb (y viceversa).
  - Clipa max_depth / num_leaves|max_leaves a la grilla VIGENTE
    (TREE_MAX_DEPTH / TREE_MAX_LEAVES): si los topes cambiaron, la siembra
    nunca queda fuera de rango (VENTURA d8/l54 -> d7/l40, etc.).
  - Filtra a claves del search space (descarta metadata: git commit, sha,
    n_rows, perfil de tuning...).
  - Toggle por env `WARM_START_FROM_REGISTRY` (default ON) para apagarlo sin
    rebuild si hiciera falta.
"""

from __future__ import annotations

from src import config

# Clave EXCLUSIVA de cada backend en el search space: si no esta presente en
# los params del campeon registrado, ese campeon es de OTRO backend -> no se
# siembra (evita mezclar espacios).
_SIGNATURE_KEY: dict[str, str] = {
    "lgb": "regressor__regressor__num_leaves",
    "xgb": "regressor__regressor__grow_policy",
}
# Clave del nº de hojas/ramas por backend (se clipa contra TREE_MAX_LEAVES).
_LEAVES_KEY: dict[str, str] = {
    "lgb": "regressor__regressor__num_leaves",
    "xgb": "regressor__regressor__max_leaves",
}
_DEPTH_KEY = "regressor__regressor__max_depth"
_SPACE_PREFIXES = ("regressor__", "preprocessor__")


def _coerce(value: str) -> object:
    """Convierte un param de MLflow (siempre string) a su tipo nativo.

    Orden importa: bool antes que int/float (para no mapear 'True'->error),
    int antes que float (max_depth='5' debe ser int, no 5.0), y string como
    fallback (categoricas: 'percentile', 'depthwise'). Ninguna categorica del
    espacio es numerica, asi que esta heuristica es segura aqui.
    """
    if value == "True":
        return True
    if value == "False":
        return False
    if value.lstrip("-").isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _clip_to_grid(params: dict[str, object], model_type: str) -> dict[str, object]:
    """Recorta depth y hojas a la grilla VIGENTE (lazy import para leer el
    valor actual de los env de search_spaces, no uno horneado)."""
    from src.step_04_train.search_spaces import TREE_MAX_DEPTH, TREE_MAX_LEAVES

    p = dict(params)
    depth = TREE_MAX_DEPTH
    if _DEPTH_KEY in p:
        depth = max(3, min(int(p[_DEPTH_KEY]), TREE_MAX_DEPTH))
        p[_DEPTH_KEY] = depth

    leaves_key = _LEAVES_KEY.get(model_type)
    if leaves_key and leaves_key in p:
        if model_type == "lgb":
            lo, hi = 7, max(7, min(2**depth - 1, TREE_MAX_LEAVES))
        else:  # xgb
            lo, hi = 8, max(8, min(2**depth, TREE_MAX_LEAVES))
        p[leaves_key] = max(lo, min(int(p[leaves_key]), hi))
    return p


def _is_same_backend(space_params: dict, model_type: str) -> bool:
    """True si los params del campeon corresponden al backend que se va a tunear
    (la clave-firma del espacio esta presente)."""
    sig = _SIGNATURE_KEY.get(model_type)
    return bool(sig and sig in space_params)


def _fetch_champion_space(client, model_name: str) -> tuple[dict, str, str, dict] | None:
    """(params del search space, version, run_id, metrics) del campeon registrado
    mas reciente, o None si no hay versiones. Filtra metadata (git/sha/n_rows...):
    deja solo las claves del espacio (regressor__/preprocessor__). `metrics` trae
    las metricas del run (para el guard anti-overfit del caller)."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    latest = max(versions, key=lambda mv: int(mv.version))
    run_data = client.get_run(latest.run_id).data
    raw = run_data.params or {}
    space = {k: v for k, v in raw.items() if k.startswith(_SPACE_PREFIXES)}
    return space, latest.version, latest.run_id, dict(run_data.metrics or {})


def _prev_champion_overfit(metrics: dict) -> bool:
    """True si el campeon registrado quedo con gap_rel sobre el gate vigente.

    Guard 2026-07-01: un campeon puede registrarse CON warning de overfitting
    (el gap no bloquea, solo el MAPE). Sembrar la ronda final con esa config
    empuja al proximo reentreno hacia la MISMA zona memorizada — perpetua el
    overfit entre reentrenos. Si el gap_rel del run registrado supera
    CHAMPION_MAX_GAP_REL, mejor arranque frio (el TPE + los caps n-aware
    vigentes buscan una zona sana). Sin metricas suficientes -> False
    (no bloquea; fail-open como todo el warm-start).
    """
    gap = metrics.get("nested_cv_gap_mean")
    mae_test = metrics.get("nested_cv_mae_mean")
    if gap is None or mae_test is None or not mae_test > 0:
        return False
    return (abs(float(gap)) / float(mae_test)) > config.CHAMPION_MAX_GAP_REL


def build_warm_start_params(variety: str | None, model_type: str, logger) -> dict | None:
    """Params del campeon registrado para sembrar Optuna, o None (arranque frio).

    Orquesta: guard -> fetch (MLflow) -> mismo backend -> coerce + clip. Nunca
    lanza: cualquier problema -> warning + None (contrato fail-open).
    """
    if not config.WARM_START_FROM_REGISTRY or not variety:
        return None

    tag = f"[{variety}/{model_type}] warm-start"
    model_name = f"{config.MODEL_REGISTRY_PREFIX}{variety}".strip("_")
    try:
        import mlflow

        fetched = _fetch_champion_space(mlflow.tracking.MlflowClient(), model_name)
        if fetched is None:
            logger.info(f"{tag}: sin modelo previo en Registry ({model_name}); arranque en frio.")
            return None

        space, version, run_id, prev_metrics = fetched
        if not _is_same_backend(space, model_type):
            logger.info(f"{tag}: el campeon {model_name} v{version} es de otro backend; frio.")
            return None
        if _prev_champion_overfit(prev_metrics):
            logger.info(
                f"{tag}: campeon previo {model_name} v{version} registro con gap_rel sobre "
                f"el gate ({config.CHAMPION_MAX_GAP_REL}); NO se siembra su config "
                f"(evita perpetuar el overfit). Arranque en frio."
            )
            return None

        seed = _clip_to_grid({k: _coerce(v) for k, v in space.items()}, model_type)
        logger.info(
            f"{tag} ON | siembra desde {model_name} v{version} (run={run_id[:8]}) | "
            f"{len(seed)} params (depth={seed.get(_DEPTH_KEY)}, "
            f"leaves={seed.get(_LEAVES_KEY.get(model_type))})"
        )
        return seed
    except Exception as exc:
        logger.warning(f"{tag} fallo (arranque en frio): {exc}")
        return None
