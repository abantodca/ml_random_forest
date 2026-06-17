"""Construccion del pipeline de preprocesamiento."""
from __future__ import annotations

from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from src.config import (
    ENABLE_FUNDO_FORMATO_INTERACTION,
    ENABLE_LOF_BEFORE_CAPPER,
    ENABLE_OUTLIER_CASCADE_FF,
)
from src.step_02_clean.imputers import CustomKNNImputer
from src.step_02_clean.missing_flags import MissingFlagger
from src.step_02_clean.outlier_score import LOFOutlierScorer
from src.step_02_clean.outliers import OutlierCapper
from src.step_03_features.exante import ConcurrentFeatureDropper
from src.step_03_features.feature_engineering import FeatureGenerator
from src.step_03_features.lag_features import LagFeatureTransformer
from src.variety_config import VarietyConfig


def create_preprocessing_pipeline(
    variety_cfg: VarietyConfig | None = None,
) -> Pipeline:
    """Encadena: lags -> missing flags -> imputacion KNN -> capping -> LOF score -> ciclicas -> filtro varianza.

    Lag features (step 0): `LagFeatureTransformer` calcula rolling windows
    POR fold durante CV (sin leakage) y memoriza el historial para
    inferencia. En entrenamiento ve TODO el train fold; en cada predict()
    de test reusa solo el historial del fit, sin contaminar entre folds.

    `OutlierCapper`: bounds aprenden del grupo. Default `group_col="FUNDO"`
    (legacy del LGB v3 baseline, MAPE 13.39% gap 0.138). Si la flag
    `ENABLE_OUTLIER_CASCADE_FF` esta activa, cambia a cascade:
        1. bounds por (FUNDO, FORMATO)   — mas especifico
        2. fallback bounds por FUNDO solo
        3. fallback bounds globales

    Justificacion del cascade: EDA POP detecto que el 86% del data es
    FORMATO=GRANEL y 72% FUNDO=A9; los bounds por-FUNDO solo reflejan A9
    (donde GRANEL domina) y no tocan outliers de CLAMSHELL pequenos.
    Pero es OPT-IN: hay que demostrar via ablation que mejora baseline.

    El `variance_filter` final descarta dummies constantes que aparecen
    cuando una variedad no observa todos los niveles de FUNDO/FORMATO
    (la dummy queda en 0 para todas las filas). `set_output('pandas')`
    preserva el DataFrame para mantener nombres de columna hacia XGB/LGB.

    Importante: el step `lag_features` requiere `y` en fit(); sklearn
    Pipeline lo propaga automaticamente cuando el caller hace `pipeline.fit(X, y)`.

    `variety_cfg` (P0.2): overrides POR VARIEDAD (meses de temporada,
    umbral KNN). None o campos None = defaults globales de hoy — POP queda
    bit-identico. Los valores quedan guardados en los __init__ de los
    componentes (clone-safe) y serializados con el pipeline.
    """
    cfg = variety_cfg or VarietyConfig(variety="")
    # kwargs condicionales: no pasar None pisaria el default del componente
    # (p.ej. fallback_threshold lee env IMPUTER_KNN_THRESHOLD).
    imputer_kwargs = (
        {"fallback_threshold": cfg.imputer_knn_threshold}
        if cfg.imputer_knn_threshold is not None else {}
    )
    capper_step = (
        "outliers",
        OutlierCapper(
            group_col=(
                ["FUNDO", "FORMATO"]
                if ENABLE_OUTLIER_CASCADE_FF
                else "FUNDO"
            ),
        ),
    )
    # LOF como FEATURE (additive). EDA POP 2026-05-09 detecto kurt=158
    # en DPC y 9.1% outliers IQR en KG/HA. LOF informa al modelo cuando
    # una fila es atipica multivariadamente — los arboles deciden si lo
    # usan o no. Va DESPUES del imputer (LOF no acepta NaN) y ANTES de
    # FeatureGenerator (asi el score se conserva en el output final).
    #
    # Orden LOF vs capper (flag ENABLE_LOF_BEFORE_CAPPER, Fase B.5):
    #   - OFF (legacy): capper -> LOF. El LOF puntua data ya recortada;
    #     los extremos que el capper clipeo desaparecen del score.
    #   - ON: LOF -> capper. El score captura los extremos REALES y el
    #     capper sigue protegiendo al modelo despues. lof_score no esta
    #     en NUMERIC_FEATURES, el capper no lo toca.
    lof_step = ("outlier_score", LOFOutlierScorer())
    middle_steps = (
        [lof_step, capper_step]
        if ENABLE_LOF_BEFORE_CAPPER
        else [capper_step, lof_step]
    )
    return Pipeline(
        steps=[
            ("lag_features", LagFeatureTransformer()),
            ("missing_flags", MissingFlagger()),
            ("imputer", CustomKNNImputer(**imputer_kwargs)),
            *middle_steps,
            (
                "feature_engineering",
                FeatureGenerator(
                    add_fundo_formato_interaction=ENABLE_FUNDO_FORMATO_INTERACTION,
                    high_season_months=cfg.high_season_months,
                    low_season_months=cfg.low_season_months,
                ),
            ),
            # Experimento EX-ANTE (#11): passthrough con EXANTE_MODE=0
            # (default); con el flag activo elimina las features del dia
            # del evento (KG/HA, %INDUS y derivadas) — ver exante.py.
            ("drop_concurrent", ConcurrentFeatureDropper()),
            (
                "variance_filter",
                VarianceThreshold(threshold=0.0).set_output(transform="pandas"),
            ),
        ]
    )
