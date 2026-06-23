"""Espacios de busqueda Optuna por backend.

Aislar los espacios aqui (en vez de mezclarlos con el bucle Nested CV)
permite:
    - Anadir un nuevo backend (ngboost, tabnet, ...) editando UN solo
      archivo, sin tocar `tuning.py`.
    - Versionar cambios de search space sin churn en la logica de CV.
    - Reusar los espacios desde notebooks de exploracion.

Cada `_suggest_*` recibe el `optuna.Trial` y devuelve un dict de
`{nombre_param_pipeline: valor}` listo para `Pipeline.set_params(**dict)`.
Las claves usan el prefijo `regressor__` o `preprocessor__<step>__`
porque el pipeline final tiene la forma:

    Pipeline(steps=[("preprocessor", <pp>), ("regressor", <model>)])
"""

from __future__ import annotations

import optuna

# ---------------------------------------------------------------------------
# Preprocesador (compartido para todos los backends)
# ---------------------------------------------------------------------------


def suggest_preprocessor_params(trial: optuna.Trial) -> dict[str, object]:
    """Hiperparametros del preprocesador que tambien tuneamos.

    EDA POP 2026-05-09 incorporo dos transformers nuevos al preprocesador:
        - `LOFOutlierScorer` (step_02_clean.outlier_score) — agrega
          `lof_score` feature. Tuneamos `n_neighbors` (5-50) para que
          Optuna decida cuanto contexto local usar.
        - skew-mitigated features (log1p / sqrt) en FeatureGenerator no
          requieren tuning — son determinísticas dado el dataset.

    Los rangos de los hiperparametros del MODELO (XGB/LGB) NO se cambian
    en esta revision: estan bien justificados con runs historicos. La
    incorporacion de features nuevas (LOF + t_index + log1p/sqrt versions)
    se valida via A/B en Phase 8 antes de retunear el modelo.
    """
    return {
        "preprocessor__imputer__n_neighbors": trial.suggest_int(
            "preprocessor__imputer__n_neighbors", 3, 40
        ),
        "preprocessor__outliers__method": trial.suggest_categorical(
            "preprocessor__outliers__method", ["iqr", "percentile"]
        ),
        "preprocessor__outliers__factor": trial.suggest_float(
            "preprocessor__outliers__factor", 1.5, 5.0
        ),
        "preprocessor__outlier_score__n_neighbors": trial.suggest_int(
            "preprocessor__outlier_score__n_neighbors", 5, 50
        ),
    }


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def suggest_xgb_params(trial: optuna.Trial) -> dict[str, object]:
    """Search space rev. 7 (2026-06-10): capacidad reabierta + early stopping.

    Las revisiones 6.x recortaron la grilla (max_depth<=5, min_child_weight
    >=10, gamma>=1.5, reg_lambda>=3) "para pasar el quality gate" de gap.
    Eso convirtio el control de overfitting en SUBAJUSTE estructural: el
    modelo no podia aprender estructura fina aunque existiera, y el MAE_test
    se estanco. Rev. 7 invierte la estrategia:

    - El control de overfitting ya NO vive en la grilla. Vive en:
        (a) early stopping interno por fit (EarlyStoppingXGBRegressor),
        (b) CV outer temporal (drift-honesto) que castiga memorizar,
        (c) gate de gap en select_champion (restriccion, no objetivo).
    - n_estimators REMOVIDO del search: fijo en N_ESTIMATORS_MAX y el
      early stopping decide el corte real (la justificacion del cap viejo
      era exactamente "sin early stopping no se cortan").
    - Limites inferiores de regularizacion bajados a ~0 (log): Optuna decide
      cuanta regularizacion necesita, en vez de forzarla por construccion.
    - max_depth 3-8: deja explorar interacciones de mayor orden.

    Se mantiene de rev. 6.x lo que si estaba justificado: colsample_bylevel/
    bynode fuera (multiplicaban agresividad), max_delta_step fuera (marginal
    en MAE).

    Rev. 9 (2026-06-22): max_leaves pasa a tunearse SIEMPRE (antes solo bajo
    grow_policy='lossguide'). Con depthwise sin tope, ~50% de los trials
    crecian arboles a 2^depth hojas (hasta ~256 en depth 8) vs el cap 64 de
    num_leaves en LGB — asimetria estructural que empujaba a XGB al
    sobreajuste y lo descalificaba en el gate de gap. En XGBoost >=2.0
    max_leaves acota el ancho tambien con depthwise (verificado en 3.2.0),
    asi el control de ancho de XGB queda a la par del de LGB.
    """
    grow_policy = trial.suggest_categorical(
        "regressor__regressor__grow_policy", ["depthwise", "lossguide"]
    )
    max_depth = trial.suggest_int("regressor__regressor__max_depth", 3, 8)
    # max_leaves acoplado a depth y SIEMPRE tuneado: 8 .. min(2^depth, 64),
    # espejando el num_leaves de suggest_lgb_params (7 .. min(2^depth-1, 64)).
    # En depth bajos la formula acota sola; el cap absoluto 64 evita arboles
    # degenerados en depth 7-8.
    max_leaves_max = max(8, min(2**max_depth, 64))
    params = {
        "regressor__regressor__max_depth": max_depth,
        "regressor__regressor__learning_rate": trial.suggest_float(
            # Piso subido 3e-3 -> 1e-2 (2026-06-23): un LR < 1e-2 con
            # N_ESTIMATORS_MAX + early stopping crece cientos/miles de arboles
            # por fit (tiempo) y ajusta ruido sin ganar MAPE. El XGB elegido en
            # prod_xl 2026-06-22 ya uso lr=0.0179 (> 1e-2): el piso viejo solo
            # gastaba trials del TPE en una region lenta y sobreajustada.
            "regressor__regressor__learning_rate",
            1e-2,
            0.3,
            log=True,
        ),
        "regressor__regressor__subsample": trial.suggest_float(
            "regressor__regressor__subsample", 0.5, 1.0
        ),
        "regressor__regressor__colsample_bytree": trial.suggest_float(
            "regressor__regressor__colsample_bytree", 0.5, 1.0
        ),
        "regressor__regressor__min_child_weight": trial.suggest_float(
            "regressor__regressor__min_child_weight", 1.0, 50.0, log=True
        ),
        "regressor__regressor__gamma": trial.suggest_float(
            "regressor__regressor__gamma", 1e-3, 15.0, log=True
        ),
        "regressor__regressor__reg_alpha": trial.suggest_float(
            "regressor__regressor__reg_alpha", 1e-3, 20.0, log=True
        ),
        "regressor__regressor__reg_lambda": trial.suggest_float(
            "regressor__regressor__reg_lambda", 1e-3, 25.0, log=True
        ),
        "regressor__regressor__grow_policy": grow_policy,
        "regressor__regressor__max_leaves": trial.suggest_int(
            "regressor__regressor__max_leaves", 8, max_leaves_max
        ),
    }
    return params


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


def suggest_lgb_params(trial: optuna.Trial) -> dict[str, object]:
    """Search space rev. 8 (2026-06-10): capacidad reabierta + early stopping.

    Las revisiones 7.x recortaron la grilla "anti-gap" hasta el subajuste
    estructural: num_leaves cap 18, min_child_samples>=60 (cada hoja obligada
    a >=60 de ~10k filas), reg_lambda>=5, subsample<=0.8. El MAE_test se
    estanco porque el modelo no PODIA aprender mas, no porque no hubiera
    senal. Rev. 8 invierte la estrategia (igual que XGB rev. 7):

    - Control de overfitting fuera de la grilla: early stopping interno
      (EarlyStoppingLGBMRegressor), CV outer temporal, y gate de gap en
      select_champion como restriccion.
    - n_estimators REMOVIDO del search: fijo en N_ESTIMATORS_MAX, el early
      stopping decide el corte real.
    - num_leaves acoplado a depth: 7 .. min(2^depth - 1, 64). En depth bajos
      la formula limita sola; el cap absoluto 64 evita arboles degenerados
      en depth 7-8.
    - min_child_samples 5-100 (log), regularizacion con piso ~0 (log):
      Optuna decide cuanta necesita.

    Se mantiene de rev. 7.x lo justificado: feature_fraction_bynode fuera,
    subsample_freq fijo=1 en model_lgb.py, y las palancas `extra_trees` /
    `path_smooth` (rev. 7.3) que son ortogonales y Optuna puede apagar.
    Notas: objective='regression_l1' (ver model_lgb.py) -> MAE nativo.
    """
    max_depth = trial.suggest_int("regressor__regressor__max_depth", 3, 8)
    num_leaves_max = max(7, min(2**max_depth - 1, 64))
    return {
        "regressor__regressor__max_depth": max_depth,
        "regressor__regressor__num_leaves": trial.suggest_int(
            "regressor__regressor__num_leaves", 7, num_leaves_max
        ),
        "regressor__regressor__learning_rate": trial.suggest_float(
            # Piso subido 3e-3 -> 1e-2 (2026-06-23): ver nota en suggest_xgb_params.
            # El LGB campeon de prod_xl 2026-06-22 uso lr=0.0128 (> 1e-2), asi que
            # el piso no excluye la zona buena, solo la lenta/sobreajustada.
            "regressor__regressor__learning_rate",
            1e-2,
            0.3,
            log=True,
        ),
        "regressor__regressor__subsample": trial.suggest_float(
            "regressor__regressor__subsample", 0.5, 1.0
        ),
        "regressor__regressor__colsample_bytree": trial.suggest_float(
            "regressor__regressor__colsample_bytree", 0.5, 1.0
        ),
        "regressor__regressor__min_child_samples": trial.suggest_int(
            "regressor__regressor__min_child_samples", 5, 100, log=True
        ),
        "regressor__regressor__min_split_gain": trial.suggest_float(
            "regressor__regressor__min_split_gain", 1e-3, 5.0, log=True
        ),
        "regressor__regressor__reg_alpha": trial.suggest_float(
            "regressor__regressor__reg_alpha", 1e-3, 20.0, log=True
        ),
        "regressor__regressor__reg_lambda": trial.suggest_float(
            "regressor__regressor__reg_lambda", 1e-3, 25.0, log=True
        ),
        "regressor__regressor__extra_trees": trial.suggest_categorical(
            "regressor__regressor__extra_trees", [False, True]
        ),
        "regressor__regressor__path_smooth": trial.suggest_float(
            "regressor__regressor__path_smooth", 0.0, 2.0
        ),
    }


# ---------------------------------------------------------------------------
# `suggest_full_params`: combina preprocesador + backend.
#
# El registry de backends vive en `step_04_train/registry.py` (single source
# of truth para factory + search_space). Aqui lo importamos LAZY para evitar
# import circular: registry.py importa `suggest_xgb_params` y
# `suggest_lgb_params` de este modulo.
# ---------------------------------------------------------------------------


def suggest_full_params(trial: optuna.Trial, model_type: str) -> dict[str, object]:
    """Concatena search space del preprocesador + del modelo elegido."""
    from src.step_04_train.registry import get_backend  # lazy: rompe ciclo

    backend = get_backend(model_type)
    return {
        **suggest_preprocessor_params(trial),
        **backend.search_space(trial),
    }
