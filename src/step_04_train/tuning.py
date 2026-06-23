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

import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import ExperimentalWarning
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

# Silenciar warning experimental de optuna ANTES de importar los modulos
# del proyecto que a su vez disparan llamadas a optuna en tiempo de import.
warnings.filterwarnings("ignore", category=ExperimentalWarning)

from src.config import (  # noqa: E402  (filterwarnings debe ir antes)
    INNER_CV_FOLDS,
    OOF_ENSEMBLE_K,
    OUTER_CV_FOLDS,
    RANDOM_STATE,
    SAMPLE_WEIGHT_BINS,
    SAMPLE_WEIGHT_CAP,
)
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


def _make_study(seed: int) -> optuna.Study:
    """TPE multivariado. Sin pruner: cada trial devuelve UN solo score
    (CV ya hecho), no hay valores intermedios que prunir."""
    sampler = optuna.samplers.TPESampler(
        seed=seed, multivariate=True, warn_independent_sampling=False
    )
    return optuna.create_study(direction="minimize", sampler=sampler)


def _build_pipeline(preprocessor: Pipeline, model_type: str) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", _build_model(model_type)),
        ]
    )


def _build_strat_label(
    X: pd.DataFrame,
    min_count: int,
) -> tuple[pd.Series | None, str]:
    """Etiqueta de estratificacion ADAPTATIVA por variedad.

    Cascada de estrategias (mas especifica primero):
        1. `FUNDO_FORMATO` compuesto, con clases n<min_count -> 'RARE'.
        2. `FUNDO` solo (idem).
        3. `FORMATO` solo (idem).
        4. None -> caller cae a `KFold` sin estratificar.

    En cada nivel se valida que tras colapsar:
        - hay >=2 clases distintas (sin variabilidad no se puede stratify),
        - cada clase final tiene n>=min_count (requisito de StratifiedKFold).

    Asi una variedad con 4x4 categoricas y desbalance moderado entra por la
    estrategia compuesta; una variedad con 1 solo FUNDO entra por FORMATO; y
    una con 1 FUNDO y 1 FORMATO degenera a KFold sin tropezar.

    Devuelve (label, strategy_name) para que el caller logue la decision.
    """
    candidates: list[tuple[str, pd.Series]] = []
    if "FUNDO" in X.columns and "FORMATO" in X.columns:
        candidates.append(
            ("FUNDO_FORMATO", X["FUNDO"].astype(str) + "_" + X["FORMATO"].astype(str))
        )
    if "FUNDO" in X.columns:
        candidates.append(("FUNDO", X["FUNDO"].astype(str)))
    if "FORMATO" in X.columns:
        candidates.append(("FORMATO", X["FORMATO"].astype(str)))

    for name, label in candidates:
        counts = label.value_counts()
        rare = counts[counts < min_count].index
        if len(rare) > 0:
            label = label.where(~label.isin(rare), other="RARE")
        final_counts = label.value_counts()
        if len(final_counts) >= 2 and (final_counts >= min_count).all():
            return label, name
    return None, "none"


def _objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Pipeline,
    inner_cv,
    model_type: str,
    sample_weights_train: np.ndarray | None = None,
    strat_label_train: pd.Series | None = None,
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

    params = suggest_full_params(trial, model_type)
    scores: list[float] = []
    gaps: list[float] = []
    for tr_i, te_i in inner_cv.split(X_train, strat_label_train):
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


def _build_cv_splitters(
    X: pd.DataFrame,
    outer_folds: int,
    inner_folds: int,
    random_state: int,
):
    """Construye outer/inner CV. Outer puede ser stratified o temporal.

    Devuelve `(outer_cv, inner_cv, strat_label, strat_strategy)`.

    Outer strategy controlada por `CV_OUTER_STRATEGY` (env / config):
        - "stratified" (default): StratifiedKFold por FUNDO_FORMATO con
          fallback adaptativo a FUNDO -> FORMATO -> KFold.
        - "temporal_year": TemporalYearSplit (expanding window por ANIO).
          Resuelve drift severo: el modelo NO ve futuro en train. Necesita
          columna ANIO o DATE_COLUMN en X.

    Inner siempre stratified (dentro del outer fold el riesgo temporal ya
    se mitigo; el inner Optuna se beneficia del balance por estrato).
    """
    import math

    from src.config import CV_OUTER_STRATEGY, TEMPORAL_CV_MIN_TRAIN_YEARS

    strat_min_count = max(
        outer_folds,
        math.ceil(inner_folds * outer_folds / max(outer_folds - 1, 1)),
    )
    strat_label, strat_strategy = _build_strat_label(X, min_count=strat_min_count)

    # Outer
    if CV_OUTER_STRATEGY == "temporal_year":
        from src.step_04_train.temporal_cv import TemporalYearSplit

        outer_cv = TemporalYearSplit(
            year_col="ANIO",
            n_splits=outer_folds,
            min_train_years=TEMPORAL_CV_MIN_TRAIN_YEARS,
        )
    else:
        outer_splitter_cls = StratifiedKFold if strat_label is not None else KFold
        outer_cv = outer_splitter_cls(
            n_splits=outer_folds,
            shuffle=True,
            random_state=random_state,
        )

    # Inner: siempre stratified (cuando hay strat_label) — el outer fold
    # contiene multiples anios mezclados, el balance por FUNDO_FORMATO
    # estabiliza la inner CV de Optuna.
    inner_splitter_cls = StratifiedKFold if strat_label is not None else KFold
    inner_cv = inner_splitter_cls(
        n_splits=inner_folds,
        shuffle=True,
        random_state=random_state,
    )
    return outer_cv, inner_cv, strat_label, strat_strategy


def _maybe_sample_weights(
    y: pd.Series,
    use_sample_weights: bool,
    logger,
    X: pd.DataFrame | None = None,
    high_season_months: tuple | None = None,
) -> np.ndarray | None:
    """Computa sample_weights por decil del target o devuelve None.

    Capas opcionales (config-driven, se multiplican y renormalizan a media=1):
      - `SAMPLE_WEIGHT_INV_Y` (Fase B.4): pesos ∝ 1/y (alineacion MAE->MAPE).
      - `SAMPLE_WEIGHT_HIGH_SEASON`: boost a meses pico (autopsia OOF
        2026-06-11: peor 5% de errores 2-3x sobre-representado en ago-oct).
        Requiere `X` con la columna de fecha.

    `high_season_months` (P0.2, VarietyConfig): meses pico POR VARIEDAD.
    None = env global SAMPLE_WEIGHT_HIGH_SEASON_MONTHS (default POP 8,9,10).
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
    if SAMPLE_WEIGHT_HIGH_SEASON and X is not None and DATE_COLUMN in X.columns:
        meses_pico = (
            high_season_months
            if high_season_months is not None
            else SAMPLE_WEIGHT_HIGH_SEASON_MONTHS
        )
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
    from src.config import DUAL_CV_FOLDS, TEMPORAL_CV_MIN_TRAIN_YEARS
    from src.step_04_train.temporal_cv import TemporalYearSplit

    try:
        splitter = TemporalYearSplit(
            year_col="ANIO",
            n_splits=DUAL_CV_FOLDS,
            min_train_years=TEMPORAL_CV_MIN_TRAIN_YEARS,
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
        mask = np.isfinite(oof_pred) & np.isfinite(y_arr) & (np.abs(y_arr) > 1e-9)
        if not mask.any():
            logger.warning("Chequeo temporal sin filas OOF validas; se omite.")
            return {}
        ape = np.abs(oof_pred[mask] - y_arr[mask]) / np.abs(y_arr[mask])
        metrics = {
            "temporal_mape_oof": float(ape.mean() * 100.0),
            "temporal_r2_oof": float(r2_score(y_arr[mask], oof_pred[mask])),
            "temporal_mae_test_mean": float(np.mean(mae_folds)),
            "temporal_n_oof": float(mask.sum()),
        }
        logger.info(
            f"Chequeo honesto temporal | MAPE_oof={metrics['temporal_mape_oof']:.2f}% | "
            f"R2_oof={metrics['temporal_r2_oof']:.4f} | "
            f"MAE_test={metrics['temporal_mae_test_mean']:.4f} | "
            f"n_oof={int(metrics['temporal_n_oof'])} "
            f"(forecast de anio no visto; el stratified mide interpolacion)"
        )
        return metrics
    except Exception as exc:  # noqa: BLE001 — el chequeo jamas rompe el training
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
) -> dict[str, object]:
    """Devuelve los params para el refit final.

    Dos modos:
      - `skip_final_tuning=True`: argmin sobre los outer folds (rapido).
      - `False` (default): ronda extra de Optuna sobre TODO el dataset.
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

    logger.info(f"Ronda final | trials={final_trials} sobre dataset completo...")
    final_study = _make_study(random_state)
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
        ),
        n_trials=final_trials,
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
                        mejores parametros del MEJOR outer fold (argmin MAE
                        test). Ahorra ~1/(outer_folds+1) del tiempo total.
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

    outer_cv, inner_cv, strat_label, strat_strategy = _build_cv_splitters(
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
    )
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
