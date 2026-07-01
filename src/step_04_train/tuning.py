"""Tuning bayesiano (Optuna) con Nested Cross-Validation.

Diseno:
- Outer CV    : estima el error de generalizacion del PROCEDIMIENTO completo
                (preprocesamiento + tuning + entrenamiento), no de un modelo
                concreto.
- Inner CV    : selecciona los mejores hiperparametros DENTRO de cada outer
                fold con un sampler TPE multivariado de Optuna.
- Final tune  : ronda extra de optimizacion sobre TODO el dataset (suele ser
                mas corta que las del nested CV via `final_trials`) y refit
                del pipeline que se promueve a produccion.

El espacio de busqueda incluye TANTO el modelo como el preprocesador
(`imputer__n_neighbors`, `outliers__factor`, `outliers__method`), de modo
que Optuna tunea el pipeline completo.

Acumulamos predicciones out-of-fold (OOF) durante el outer CV: son
predicciones honestas (cada fila predicha por un modelo que NO la vio en
entrenamiento) y se usan para construir los graficos del reporte gerencial.
"""

from __future__ import annotations

import contextlib
import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from optuna.trial import TrialState
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Silenciar warning experimental de optuna ANTES de importar los modulos
# del proyecto que a su vez disparan llamadas a optuna en tiempo de import.
warnings.filterwarnings("ignore", category=ExperimentalWarning)

from src.config import (  # noqa: E402  (filterwarnings debe ir antes)
    ADAPT_FOLDS_ROWS_PER_INNER,
    ADAPT_FOLDS_ROWS_PER_OUTER,
    ADAPT_FOLDS_TO_N,
    INNER_CV_FOLDS,
    OOF_ENSEMBLE_K,
    OUTER_CV_FOLDS,
    RANDOM_STATE,
    SAMPLE_WEIGHT_BINS,
    SAMPLE_WEIGHT_CAP,
)
from src.step_04_train.cv_strategy import build_cv_splitters  # noqa: E402
from src.step_04_train.oof_ensemble import OOFEnsembleRegressor  # noqa: E402
from src.step_04_train.registry import get_backend  # noqa: E402
from src.step_04_train.sample_weights import (  # noqa: E402
    compute_inv_target_weights,
    compute_sample_weights,
)
from src.step_04_train.search_spaces import suggest_full_params  # noqa: E402
from src.utils.sklearn_helpers import (  # noqa: E402
    fit_with_optional_sample_weight,
    index_or_none,
)

# Logger inerte hasta que el caller configure handlers (idem data_loader).
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factory: delega al BACKEND_REGISTRY (single source of truth)
# ---------------------------------------------------------------------------


def _build_model(model_type: str):
    """Construye el regressor envuelto (TTR + base) para `model_type`."""
    return get_backend(model_type).factory()


# ---------------------------------------------------------------------------
# Optuna study factory + objective
# ---------------------------------------------------------------------------


def _make_study(
    seed: int,
    warm_start_params: dict | None = None,
    study_name: str | None = None,
    logger=logger,
) -> optuna.Study:
    """TPE multivariado + MedianPruner opcional (ENABLE_PRUNER).

    El pruner poda trials cuyo MAE parcial (reportado por `_objective` tras cada
    inner fold) queda peor que la mediana -> ahorra el resto de folds de los
    trials malos. Con ENABLE_PRUNER=0 usa NopPruner (comportamiento historico).

    `study_name` (opcional): si se pasa Y `OPTUNA_STORAGE_URL` esta seteado, el
    estudio PERSISTE en Postgres con ese nombre y RESUME (`load_if_exists`). Solo
    la ronda final lo usa (los outer folds pasan study_name=None -> memoria).
    Si la conexion falla, cae a memoria con warning (nunca rompe el training).

    `warm_start_params` (opcional): config del campeon registrado a evaluar
    como PRIMER trial (study.enqueue_trial). El resto de los trials parte de
    ahi via TPE. `skip_if_exists=True` lo hace idempotente ante resume. La
    siembra nunca rompe el estudio: si enqueue falla, se sigue en frio.
    """
    from src.config import (
        ENABLE_PRUNER,
        OPTUNA_STORAGE_URL,
        PRUNER_STARTUP_TRIALS,
        PRUNER_WARMUP_STEPS,
    )

    sampler = optuna.samplers.TPESampler(
        seed=seed, multivariate=True, warn_independent_sampling=False
    )
    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=PRUNER_STARTUP_TRIALS,
            n_warmup_steps=PRUNER_WARMUP_STEPS,
        )
        if ENABLE_PRUNER
        else optuna.pruners.NopPruner()
    )
    common = {"direction": "minimize", "sampler": sampler, "pruner": pruner}
    study = None
    if study_name and OPTUNA_STORAGE_URL:
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=OPTUNA_STORAGE_URL,
                load_if_exists=True,  # RESUME si ya existe
                **common,
            )
            done = sum(
                t.state.is_finished() for t in study.get_trials(deepcopy=False)
            )
            if done:
                logger.info(f"Optuna RESUME | study={study_name} | trials previos={done}")
        except Exception as exc:
            logger.warning(
                f"Optuna storage no disponible ({exc}); estudio en memoria (sin resume)"
            )
            study = None
    if study is None:
        study = optuna.create_study(**common)  # en memoria (default / fallback)
    if warm_start_params:
        with contextlib.suppress(Exception):  # siembra best-effort, nunca rompe
            study.enqueue_trial(warm_start_params, skip_if_exists=True)
    return study


def _build_pipeline(preprocessor: Pipeline, model_type: str) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", _build_model(model_type)),
        ]
    )


def _objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Pipeline,
    inner_cv,
    model_type: str,
    sample_weights_train: np.ndarray | None = None,
    strat_label_train: pd.Series | None = None,
    n_rows: int | None = None,
) -> float:
    """Optuna objective: MAE promedio del inner CV con sample_weight por fold.

    Inner CV manual (no `cross_val_score`) porque sklearn no splitea
    sample_weight por fold y lo trataria como un kwarg estatico. Cuando
    `sample_weights_train` es None se pasa sample_weight=None al fit
    (degeneracion natural -> equivalente a `cross_val_score` sin pesos,
    pero sin la rama paralela que duplicaria codigo).

    `inner_cv` puede ser `KFold` o `StratifiedKFold`. Si es Stratified, hay
    que pasarle el `strat_label_train` (alineado con X_train por posicion).
    KFold ignora el segundo argumento, asi que llamamos `.split(X, y_label)`
    siempre y dejamos que sklearn decida.

    Nota: en produccion siempre va con weights (use_sample_weights=True
    es el default y compensa el sesgo 'regresion a la media' de los
    arboles).
    """
    from src.config import (
        OPTUNA_OBJECTIVE_GAP_PENALTY,
        OPTUNA_OBJECTIVE_STD_PENALTY,
    )

    track_gap = OPTUNA_OBJECTIVE_GAP_PENALTY > 0.0

    params = suggest_full_params(trial, model_type, n_rows=n_rows)
    scores: list[float] = []
    gaps: list[float] = []
    for step, (tr_i, te_i) in enumerate(inner_cv.split(X_train, strat_label_train)):
        Xt = X_train.iloc[tr_i]
        Xv = X_train.iloc[te_i]
        yt = y_train.iloc[tr_i]
        yv = y_train.iloc[te_i]
        pipe_local = _build_pipeline(preprocessor, model_type)
        pipe_local.set_params(**params)
        sw_fold = index_or_none(sample_weights_train, tr_i)
        fit_with_optional_sample_weight(pipe_local, Xt, yt, sample_weight=sw_fold)
        val_mae = float(mean_absolute_error(yv, pipe_local.predict(Xv)))
        scores.append(val_mae)
        # Pruning: reporta el MAE parcial (media de folds evaluados) y deja que
        # MedianPruner mate el trial si va peor que la mediana. Con NopPruner
        # (ENABLE_PRUNER=0) should_prune() es siempre False -> sin efecto.
        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if track_gap:
            # Costo extra (un predict del train) SOLO si la penalizacion esta
            # activa: gap = cuanto peor generaliza vs lo que memorizo del train.
            train_mae = float(mean_absolute_error(yt, pipe_local.predict(Xt)))
            gaps.append(max(0.0, val_mae - train_mae))
    # Penalizacion opcional por VARIANZA entre inner folds (robustez del
    # tuning, 2026-06-13): con lambda>0 TPE prefiere configs ESTABLES sobre
    # configs con buen promedio pero alta dispersion (que generalizan peor).
    # Default 0.0 = bit-identico al comportamiento historico (solo media).
    penalty = OPTUNA_OBJECTIVE_STD_PENALTY * float(np.std(scores))
    # Penalizacion opcional por GAP train->val (anti-overfit; ver config). Con
    # lambda>0 TPE evita configs que memorizan el train aunque tengan buen
    # MAE_val — las mismas que luego falla el gate del campeon. Default 0.0
    # (track_gap=False) -> sin costo ni cambio de comportamiento.
    if track_gap and gaps:
        penalty += OPTUNA_OBJECTIVE_GAP_PENALTY * float(np.mean(gaps))
    return float(np.mean(scores)) + penalty


# ---------------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------------


def _data_fingerprint(X: pd.DataFrame) -> str:
    """Hash corto y determinista de los datos de la variedad.

    Sirve para nombrar el estudio persistido: data nueva -> fingerprint nuevo
    -> estudio nuevo (no se mezclan valores calculados sobre otro dataset).
    """
    import hashlib

    payload = pd.util.hash_pandas_object(X, index=False).values.tobytes()
    return hashlib.sha1(payload).hexdigest()[:12]


def _adapt_folds_to_n(
    n: int, outer_folds: int, inner_folds: int
) -> tuple[int, int]:
    """Recorta outer/inner folds si n es chico (nunca sube sobre el perfil).

    Cada outer fold de TEST apunta a ~ADAPT_FOLDS_ROWS_PER_OUTER filas y cada
    inner fold de VAL a ~ADAPT_FOLDS_ROWS_PER_INNER, con piso 2. n grande ->
    devuelve los folds del perfil intactos (POP identico). Ver ADAPT_FOLDS_TO_N.
    """
    if not ADAPT_FOLDS_TO_N or n <= 0:
        return outer_folds, inner_folds
    o = max(2, min(outer_folds, n // ADAPT_FOLDS_ROWS_PER_OUTER))
    outer_train = int(n * (o - 1) / o)
    i = max(2, min(inner_folds, outer_train // ADAPT_FOLDS_ROWS_PER_INNER))
    return o, i


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------------
# Helpers privados de Nested CV (extraidos de perform_nested_cv para que el
# orquestador quede como lectura lineal de ~50 lineas).
# ---------------------------------------------------------------------------


@dataclass
class _OuterFoldResults:
    """Acumulado del outer CV loop. Mutable por construccion incremental
    (append por fold). El orquestador lo agrega a `nested_metrics` al final.
    """

    mae_test: list[float]
    mae_train: list[float]
    gap: list[float]
    r2: list[float]
    best_params: list[dict[str, object]]
    oof_pred: np.ndarray
    oof_fold: np.ndarray


def _maybe_sample_weights(
    y: pd.Series,
    use_sample_weights: bool,
    logger,
    X: pd.DataFrame | None = None,
    high_season_months: tuple | None = None,
    high_season_toggle: bool | None = None,
) -> np.ndarray | None:
    """Computa sample_weights por decil del target o devuelve None.

    Capas opcionales (config-driven, se multiplican y renormalizan a media=1):
      - `SAMPLE_WEIGHT_INV_Y` (Fase B.4): pesos ∝ 1/y (alineacion MAE->MAPE).
      - boost de temporada alta (autopsia OOF 2026-06-11: peor 5% de errores
        2-3x sobre-representado en ago-oct). Requiere `X` con la columna fecha.

    `high_season_toggle` (VarietyConfig.sample_weight_high_season): activa/apaga
    el boost POR VARIEDAD; None = env global SAMPLE_WEIGHT_HIGH_SEASON.
    `high_season_months`: meses pico POR VARIEDAD. Si es None y el boost esta
    activo, se DERIVAN de los datos (misma logica que las dummies TEMPORADA,
    2026-07-01) en vez de caer al POP 8,9,10 — evita boostear los meses
    equivocados en variedades con otro calendario.
    """
    if not use_sample_weights:
        return None
    from src.config import (
        DATE_COLUMN,
        SAMPLE_WEIGHT_HIGH_SEASON,
        SAMPLE_WEIGHT_HIGH_SEASON_BOOST,
        SAMPLE_WEIGHT_HIGH_SEASON_MONTHS,
        SAMPLE_WEIGHT_INV_Y,
        SAMPLE_WEIGHT_INV_Y_CAP,
    )

    high_season_on = (
        high_season_toggle if high_season_toggle is not None else SAMPLE_WEIGHT_HIGH_SEASON
    )

    # n_bins/weight_cap leidos de src.config para evitar override silencioso
    # del default de compute_sample_weights (antes hardcoded n_bins=10 aqui).
    sw = compute_sample_weights(
        y,
        n_bins=SAMPLE_WEIGHT_BINS,
        weight_cap=SAMPLE_WEIGHT_CAP,
    )
    extra_tags = ""
    if SAMPLE_WEIGHT_INV_Y:
        sw = sw * compute_inv_target_weights(y, weight_cap=SAMPLE_WEIGHT_INV_Y_CAP)
        sw = sw * (len(sw) / sw.sum())
        extra_tags += f" | inv_y ON (cap={SAMPLE_WEIGHT_INV_Y_CAP})"
    if high_season_on and X is not None and DATE_COLUMN in X.columns:
        meses_pico = high_season_months
        if meses_pico is None:
            # Data-driven (no POP 8,9,10) cuando la variedad no fija meses.
            from src.step_03_features.feature_engineering import FeatureGenerator

            alta_d, _ = FeatureGenerator._derive_season_months(X[DATE_COLUMN], y)
            meses_pico = alta_d if alta_d is not None else SAMPLE_WEIGHT_HIGH_SEASON_MONTHS
        months = pd.to_datetime(X[DATE_COLUMN], errors="coerce").dt.month
        boost = np.where(
            months.isin(meses_pico).to_numpy(),
            SAMPLE_WEIGHT_HIGH_SEASON_BOOST,
            1.0,
        )
        sw = sw * boost
        sw = sw * (len(sw) / sw.sum())
        extra_tags += (
            f" | high_season ON (meses={list(meses_pico)} x{SAMPLE_WEIGHT_HIGH_SEASON_BOOST})"
        )
    logger.info(
        f"Sample weights ON | n_bins={SAMPLE_WEIGHT_BINS} | "
        f"cap={SAMPLE_WEIGHT_CAP}{extra_tags} | "
        f"min={sw.min():.3f} max={sw.max():.3f} mean={sw.mean():.3f}"
    )
    return sw


def _run_outer_cv_loop(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Pipeline,
    model_type: str,
    outer_cv,
    inner_cv,
    strat_label: pd.Series | None,
    sample_weights: np.ndarray | None,
    n_trials: int,
    final_trials: int,
    skip_final_tuning: bool,
    outer_folds: int,
    random_state: int,
    t0: float,
    logger,
) -> _OuterFoldResults:
    """Itera outer folds: tune Optuna inner + refit + eval test/train.

    Acumula metricas por fold y predicciones OOF. El refit por fold es
    necesario para evaluar gap (MAE_test - MAE_train) honestamente.

    NO se hace warm-start aqui a proposito: el campeon registrado se entreno
    sobre datos que solapan con el test de cada fold, asi que sembrar sus
    params SESGARIA optimistamente la estimacion honesta de gap/MAPE_oof. El
    warm-start vive solo en la ronda final (_pick_final_params).
    """
    n = len(y)
    res = _OuterFoldResults(
        mae_test=[],
        mae_train=[],
        gap=[],
        r2=[],
        best_params=[],
        oof_pred=np.full(n, np.nan, dtype=float),
        oof_fold=np.full(n, -1, dtype=int),
    )
    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, strat_label),
        start=1,
    ):
        fold_t0 = time.perf_counter()
        logger.info(f"Outer fold {fold_idx}/{outer_folds} | tuning + eval")

        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        sw_tr = index_or_none(sample_weights, train_idx)
        # strat_label es pd.Series: requiere .iloc, no encaja en index_or_none.
        strat_tr = strat_label.iloc[train_idx] if strat_label is not None else None

        study = _make_study(random_state + fold_idx)
        study.optimize(
            # Binding por defaults: la lambda se consume dentro de esta
            # iteracion, pero el binding explicito blinda contra B023.
            lambda trial, X_tr=X_tr, y_tr=y_tr, sw_tr=sw_tr, strat_tr=strat_tr: _objective(
                trial,
                X_tr,
                y_tr,
                preprocessor,
                inner_cv,
                model_type,
                sample_weights_train=sw_tr,
                strat_label_train=strat_tr,
                # n_rows = n de la VARIEDAD (no del fold): la capacidad se acota
                # por el tamano del dataset completo, estable entre folds.
                n_rows=n,
            ),
            n_trials=n_trials,
            show_progress_bar=False,
            gc_after_trial=True,
            # Robustez (2026-06-13): un trial que crashea (inestabilidad
            # numerica del backend, OOM puntual) se marca FAILED y el study
            # sigue — no tira el nested CV completo. Si TODOS los trials
            # fallan, `study.best_params` levanta igual (correcto).
            catch=(Exception,),
        )

        best_pipeline = _build_pipeline(preprocessor, model_type)
        best_pipeline.set_params(**study.best_params)
        fit_with_optional_sample_weight(best_pipeline, X_tr, y_tr, sample_weight=sw_tr)

        y_pred_te = best_pipeline.predict(X_te)
        mae_test = float(mean_absolute_error(y_te, y_pred_te))
        r2_test = float(r2_score(y_te, y_pred_te))
        y_pred_tr = best_pipeline.predict(X_tr)
        mae_train = float(mean_absolute_error(y_tr, y_pred_tr))

        res.mae_test.append(mae_test)
        res.mae_train.append(mae_train)
        res.gap.append(mae_test - mae_train)
        res.r2.append(r2_test)
        res.best_params.append(dict(study.best_params))
        res.oof_pred[test_idx] = y_pred_te
        res.oof_fold[test_idx] = fold_idx

        fold_dt = time.perf_counter() - fold_t0
        elapsed = time.perf_counter() - t0
        eta = (elapsed / fold_idx) * (outer_folds - fold_idx) + (
            0 if skip_final_tuning else (final_trials / n_trials) * (elapsed / fold_idx)
        )
        logger.info(
            f"Fold {fold_idx} | MAE_test={mae_test:.4f} | MAE_train={mae_train:.4f} | "
            f"gap={mae_test - mae_train:+.4f} | R2={r2_test:.4f} | "
            f"dt={_format_eta(fold_dt)} | eta_resto={_format_eta(eta)}"
        )
    return res


def _aggregate_nested_metrics(res: _OuterFoldResults) -> dict[str, float]:
    """Agrega listas por-fold en el dict que consume el HTML / business audit."""
    return {
        # backward-compatible (lo que ya leia el HTML)
        "nested_cv_mae_mean": float(np.mean(res.mae_test)),
        "nested_cv_mae_std": float(np.std(res.mae_test)),
        "nested_cv_r2_mean": float(np.mean(res.r2)),
        "nested_cv_r2_std": float(np.std(res.r2)),
        # detector de overfitting
        "nested_cv_mae_train_mean": float(np.mean(res.mae_train)),
        "nested_cv_mae_train_std": float(np.std(res.mae_train)),
        "nested_cv_gap_mean": float(np.mean(res.gap)),
        "nested_cv_gap_std": float(np.std(res.gap)),
    }


def _temporal_honesty_check(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Pipeline,
    model_type: str,
    best_params: dict[str, object],
    sample_weights: np.ndarray | None,
    logger,
) -> dict[str, float]:
    """Chequeo honesto temporal post-tuning (Fase A.2, reporte dual).

    Cuando el outer CV fue stratified (interpolacion dentro de anios
    vistos), este chequeo mide ADEMAS el error de forecast real: fit con
    los `best_params` ya elegidos sobre folds expanding-window por ANIO
    y evalua sobre el anio siguiente. NO tunea (cero trials extra): solo
    DUAL_CV_FOLDS fits con params fijos.

    El MAPE OOF en KG/JR_H equivale fila a fila al MAPE de negocio en
    KG/JR (el factor H-EF multiplica y_true e y_pred por igual).

    Nunca rompe el training: cualquier excepcion se absorbe con warning
    y devuelve {} (las metricas duales simplemente no se loggean).
    """
    from src.config import (
        DUAL_CV_FOLDS,
        TEMPORAL_CV_MIN_TRAIN_YEARS,
        TEMPORAL_MAPE_REL_FLOOR,
    )
    from src.step_04_train.temporal_cv import TemporalYearSplit
    from src.step_05_evaluate.metrics import _mape_valid_mask, mape_safe

    try:
        splitter = TemporalYearSplit(
            year_col="ANIO",
            n_splits=DUAL_CV_FOLDS,
            min_train_years=TEMPORAL_CV_MIN_TRAIN_YEARS,
        )
        # Guard por años de historia (2026-07-01): con 2 años el splitter da 0
        # folds (antes: {} SILENCIOSO); con 3 años da 1 fold (una sola ventana,
        # metricas de alta varianza). Aca solo avisamos el porque; el consumidor
        # (quality_gate) usa temporal_n_folds para NO warnear drift con <2 folds.
        k_folds = splitter.get_n_splits(X)
        if k_folds <= 0:
            logger.info(
                "Chequeo temporal OMITIDO: historia insuficiente "
                f"(folds={k_folds} con min_train_years={TEMPORAL_CV_MIN_TRAIN_YEARS}); "
                "se necesitan al menos min_train_years+1 anios de datos."
            )
            return {}
        if k_folds == 1:
            logger.info(
                "Chequeo temporal con UN solo fold (3 anios de historia): "
                "las metricas temporales son una sola ventana (alta varianza), "
                "tomarlas como indicativas, no como senal de drift."
            )
        n = len(y)
        oof_pred = np.full(n, np.nan, dtype=float)
        mae_folds = []
        for train_idx, test_idx in splitter.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            sw_tr = index_or_none(sample_weights, train_idx)
            pipe = _build_pipeline(preprocessor, model_type)
            pipe.set_params(**best_params)
            fit_with_optional_sample_weight(pipe, X_tr, y_tr, sample_weight=sw_tr)
            y_pred = pipe.predict(X_te)
            oof_pred[test_idx] = y_pred
            mae_folds.append(float(mean_absolute_error(y_te, y_pred)))

        y_arr = np.asarray(y, dtype=float)
        base_mask = np.isfinite(oof_pred) & np.isfinite(y_arr)
        # Piso RELATIVO para el MAPE (fix 2026-07-01): el umbral viejo 1e-9
        # dejaba pasar targets casi-cero (artefactos de carga) que explotaban
        # el APE — ATLAS reporto temporal_MAPE 720% por 8 filas de ~0.0002.
        # R2/MAE siguen usando base_mask (robustos a escala); solo el MAPE
        # excluye denominadores implausibles. Ver TEMPORAL_MAPE_REL_FLOOR.
        med = float(np.nanmedian(np.abs(y_arr[base_mask]))) if base_mask.any() else 0.0
        denom_floor = max(1e-9, TEMPORAL_MAPE_REL_FLOOR * med)
        mask = base_mask & _mape_valid_mask(y_arr, denom_floor)
        n_mape_excl = int((base_mask & ~mask).sum())
        if not base_mask.any():
            logger.warning("Chequeo temporal sin filas OOF validas; se omite.")
            return {}
        if n_mape_excl:
            logger.info(
                f"Chequeo temporal: {n_mape_excl} fila(s) excluidas del MAPE por "
                f"denominador < {denom_floor:.4f} (artefactos casi-cero); "
                "R2/MAE las conservan."
            )
        metrics = {
            "temporal_mape_oof": mape_safe(
                y_arr[base_mask], oof_pred[base_mask], min_denom=denom_floor
            ),
            "temporal_r2_oof": float(r2_score(y_arr[base_mask], oof_pred[base_mask])),
            "temporal_mae_test_mean": float(np.mean(mae_folds)),
            "temporal_n_oof": float(base_mask.sum()),
            "temporal_n_folds": float(k_folds),
            "temporal_mape_n_excluded": float(n_mape_excl),
        }
        logger.info(
            f"Chequeo honesto temporal | MAPE_oof={metrics['temporal_mape_oof']:.2f}% | "
            f"R2_oof={metrics['temporal_r2_oof']:.4f} | "
            f"MAE_test={metrics['temporal_mae_test_mean']:.4f} | "
            f"n_oof={int(metrics['temporal_n_oof'])} "
            f"(forecast de anio no visto; el stratified mide interpolacion)"
        )
        return metrics
    except Exception as exc:
        logger.warning(f"Chequeo temporal fallo (se omite del reporte): {exc}")
        return {}


def _pick_final_params(
    *,
    fold_results: _OuterFoldResults,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Pipeline,
    inner_cv,
    model_type: str,
    sample_weights: np.ndarray | None,
    strat_label: pd.Series | None,
    final_trials: int,
    skip_final_tuning: bool,
    random_state: int,
    logger,
    warm_start_params: dict | None = None,
    study_name: str | None = None,
) -> dict[str, object]:
    """Devuelve los params para el refit final.

    Dos modos:
      - `skip_final_tuning=True`: fold MEDIANO por MAE_test (rapido).
      - `False` (default): ronda extra de Optuna sobre TODO el dataset.

    `warm_start_params`: siembra la ronda final con el campeon registrado.
    `study_name`: si se pasa y hay OPTUNA_STORAGE_URL, la ronda persiste y
    RESUME (solo corre los trials que faltan).
    """
    if skip_final_tuning:
        # Fold MEDIANO por MAE_test, no argmin (2026-06-13): el argmin
        # premiaba al fold con mas suerte (sesgo de seleccion); el mediano
        # es el representante mas honesto del procedimiento.
        order = np.argsort(fold_results.mae_test)
        best_idx = int(order[len(order) // 2])
        logger.info(
            f"Saltando ronda final | usando best_params del fold MEDIANO "
            f"#{best_idx + 1} (MAE_test={fold_results.mae_test[best_idx]:.4f})"
        )
        return fold_results.best_params[best_idx]

    final_study = _make_study(
        random_state, warm_start_params=warm_start_params, study_name=study_name, logger=logger
    )
    # RESUME: descontar los trials ya terminados (persistidos). Si ya se
    # completaron todos y hay al menos uno COMPLETE, se salta el optimize.
    trials = final_study.get_trials(deepcopy=False)
    finished = sum(t.state.is_finished() for t in trials)
    completed = sum(t.state == TrialState.COMPLETE for t in trials)
    n_remaining = max(0, final_trials - finished)
    if n_remaining == 0 and completed == 0:
        n_remaining = final_trials  # estudio sin trials utiles: recomputar
    logger.info(
        f"Ronda final | trials={final_trials} (ya={finished}, faltan={n_remaining}) "
        f"sobre dataset completo..."
    )
    if n_remaining == 0:
        logger.info("Ronda final ya completa (resume): usando best_params persistidos.")
        return final_study.best_params
    final_study.optimize(
        lambda trial: _objective(
            trial,
            X,
            y,
            preprocessor,
            inner_cv,
            model_type,
            sample_weights_train=sample_weights,
            strat_label_train=strat_label,
            n_rows=len(y),
        ),
        n_trials=n_remaining,
        show_progress_bar=False,
        gc_after_trial=True,
        catch=(Exception,),  # idem outer loop: trial fallido != run fallido
    )
    return final_study.best_params


def _fit_final_ensemble(
    *,
    preprocessor: Pipeline,
    model_type: str,
    best_params: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: np.ndarray | None,
    random_state: int,
    t0: float,
    logger,
) -> OOFEnsembleRegressor:
    """Wrap del pipeline tuneado en OOFEnsembleRegressor + fit sobre todo X."""
    base_pipeline = _build_pipeline(preprocessor, model_type)
    base_pipeline.set_params(**best_params)
    ensemble = OOFEnsembleRegressor(
        base_pipeline=base_pipeline,
        n_models=OOF_ENSEMBLE_K,
        random_state=random_state,
    )
    ensemble.fit(X, y, sample_weight=sample_weights)
    logger.info(
        f"Pipeline final entrenado | K={OOF_ENSEMBLE_K} pipelines promediados | "
        f"tiempo_total={_format_eta(time.perf_counter() - t0)} | "
        f"best_params={best_params}"
    )
    return ensemble


def perform_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Pipeline,
    n_trials: int = 30,
    final_trials: int | None = None,
    model_type: str = "xgb",
    outer_folds: int | None = None,
    inner_folds: int | None = None,
    random_state: int = RANDOM_STATE,
    skip_final_tuning: bool = False,
    inner_cv_n_jobs: int = -1,
    use_sample_weights: bool = True,
    logger=logger,
    variety_cfg=None,
) -> tuple[Pipeline, dict[str, object], dict[str, float], dict[str, np.ndarray]]:
    """Orquestador thin de Nested CV. La logica vive en helpers privados.

    Parametros
    ----------
    n_trials : trials de Optuna POR outer fold.
    final_trials : trials de la ronda extra sobre el dataset completo.
                   Si es None se usa el mismo `n_trials`.
    skip_final_tuning : si True, omite la ronda final y refitea con los
                        params del outer fold MEDIANO por MAE_test (el argmin
                        premiaba al fold con mas suerte; ver _pick_final_params).
                        Ahorra ~1/(outer_folds+1) del tiempo total.
    inner_cv_n_jobs : VESTIGIAL. Ya no aplica: el inner CV se hace manual
                      (fold a fold) para soportar sample_weight. Aceptamos
                      el flag para no romper la CLI/settings. La paralelizacion
                      real ahora es por variedad (`--parallel-varieties`).
    variety_cfg : VarietyConfig | None (P0.2). Hoy solo aporta los meses
                  pico por variedad al boost de sample weights; el resto
                  de overrides viaja dentro del `preprocessor` ya construido.

    Returns
    -------
    final_pipeline   : `OOFEnsembleRegressor` con K pipelines refiteados
                        sobre folds del KFold (K = `config.OOF_ENSEMBLE_K`).
    best_params      : dict con los hiperparametros del modelo de produccion.
    nested_metrics   : dict con MAE/R2 mean y std (test, train, gap).
    oof              : dict con `y_true`, `y_pred` y `fold_id`.
    """
    outer_folds = outer_folds or OUTER_CV_FOLDS
    inner_folds = inner_folds or INNER_CV_FOLDS
    final_trials = final_trials if final_trials is not None else n_trials

    # Recorte n-adaptativo de folds (variedades chicas): POP queda igual.
    _o0, _i0 = outer_folds, inner_folds
    outer_folds, inner_folds = _adapt_folds_to_n(len(X), outer_folds, inner_folds)
    if (outer_folds, inner_folds) != (_o0, _i0):
        logger.info(
            f"Folds n-adaptativos (n={len(X)}): outer {_o0}->{outer_folds}, "
            f"inner {_i0}->{inner_folds}"
        )

    outer_cv, inner_cv, strat_label, strat_strategy = build_cv_splitters(
        X,
        outer_folds,
        inner_folds,
        random_state,
    )

    total_trials = outer_folds * n_trials + (0 if skip_final_tuning else final_trials)
    logger.info(
        f"Nested CV inicio | model={model_type} | outer={outer_folds} | "
        f"inner={inner_folds} | trials/fold={n_trials} | "
        f"final_trials={0 if skip_final_tuning else final_trials} | "
        f"trials_total={total_trials}"
    )
    if strat_label is not None:
        logger.info(
            f"CV stratified by {strat_strategy} | "
            f"n_estratos={strat_label.nunique()} | "
            f"min_n_per_strato={int(strat_label.value_counts().min())}"
        )
    else:
        logger.info(
            "CV NO estratificado (variedad sin variabilidad util en FUNDO/FORMATO; KFold normal)"
        )

    sample_weights = _maybe_sample_weights(
        y,
        use_sample_weights,
        logger,
        X=X,
        high_season_months=getattr(variety_cfg, "sample_weight_high_season_months", None),
        high_season_toggle=getattr(variety_cfg, "sample_weight_high_season", None),
    )

    # Warm-start (2026-06-25): sembrar la RONDA FINAL con el campeon ya
    # registrado de esta variedad+backend para que los params de produccion
    # arranquen desde la zona buena (no a ciegas) y solo mejoren. Los outer
    # folds NO se siembran (preserva la honestidad del gap/MAPE_oof; ver
    # _run_outer_cv_loop). None si no hay modelo previo o el flag esta apagado.
    from src.step_04_train.warm_start import build_warm_start_params

    warm_start_params = build_warm_start_params(
        getattr(variety_cfg, "variety", None), model_type, logger
    )

    # Nombre del estudio para PERSISTIR + RESUME la ronda final (solo si hay
    # OPTUNA_STORAGE_URL; si no, None -> estudio en memoria). Incluye fingerprint
    # de los datos: data nueva -> estudio nuevo (no mezcla valores stale).
    from src.config import OPTUNA_STORAGE_URL

    final_study_name = None
    if OPTUNA_STORAGE_URL:
        variety = getattr(variety_cfg, "variety", None) or "novar"
        final_study_name = f"final_{variety}_{model_type}_{_data_fingerprint(X)}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.perf_counter()

    fold_results = _run_outer_cv_loop(
        X=X,
        y=y,
        preprocessor=preprocessor,
        model_type=model_type,
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        strat_label=strat_label,
        sample_weights=sample_weights,
        n_trials=n_trials,
        final_trials=final_trials,
        skip_final_tuning=skip_final_tuning,
        outer_folds=outer_folds,
        random_state=random_state,
        t0=t0,
        logger=logger,
    )
    nested_metrics = _aggregate_nested_metrics(fold_results)
    logger.info(
        f"Nested CV resultado | MAE_test={nested_metrics['nested_cv_mae_mean']:.4f} "
        f"+/- {nested_metrics['nested_cv_mae_std']:.4f} | "
        f"MAE_train={nested_metrics['nested_cv_mae_train_mean']:.4f} | "
        f"gap={nested_metrics['nested_cv_gap_mean']:+.4f} | "
        f"R2={nested_metrics['nested_cv_r2_mean']:.4f}"
    )

    best_params = _pick_final_params(
        fold_results=fold_results,
        X=X,
        y=y,
        preprocessor=preprocessor,
        inner_cv=inner_cv,
        model_type=model_type,
        sample_weights=sample_weights,
        strat_label=strat_label,
        final_trials=final_trials,
        skip_final_tuning=skip_final_tuning,
        random_state=random_state,
        logger=logger,
        warm_start_params=warm_start_params,
        study_name=final_study_name,
    )

    # Reporte dual (Fase A.2): si el outer fue stratified, anadir el chequeo
    # temporal honesto con los params finales. Si el outer YA fue temporal,
    # seria redundante.
    from src.config import CV_OUTER_STRATEGY, DUAL_CV_REPORT

    if DUAL_CV_REPORT and CV_OUTER_STRATEGY != "temporal_year":
        nested_metrics.update(
            _temporal_honesty_check(
                X=X,
                y=y,
                preprocessor=preprocessor,
                model_type=model_type,
                best_params=best_params,
                sample_weights=sample_weights,
                logger=logger,
            )
        )

    final_pipeline = _fit_final_ensemble(
        preprocessor=preprocessor,
        model_type=model_type,
        best_params=best_params,
        X=X,
        y=y,
        sample_weights=sample_weights,
        random_state=random_state,
        t0=t0,
        logger=logger,
    )
    oof = {
        "y_true": np.asarray(y, dtype=float),
        "y_pred": fold_results.oof_pred,
        "fold_id": fold_results.oof_fold,
    }
    return final_pipeline, best_params, nested_metrics, oof
