"""P0.2 — VarietyConfig: overrides por variedad sin tocar a POP.

Contrato: None = default global de hoy -> una variedad sin overrides se
comporta EXACTAMENTE como antes del refactor. Los overrides viajan
explicitos (no env) y quedan serializados dentro del pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.variety_config as vc
from src.pipeline.build_pipeline import create_preprocessing_pipeline
from src.step_03_features.feature_engineering import FeatureGenerator
from src.variety_config import VarietyConfig, for_variety


def test_pop_no_tiene_overrides():
    cfg = for_variety("POP")
    assert cfg.variety == "POP"
    assert cfg.high_season_months is None
    assert cfg.imputer_knn_threshold is None
    assert cfg.rare_min_count is None


def test_variedad_desconocida_usa_defaults_globales():
    cfg = for_variety("VARIEDAD_NUEVA_SIN_EVIDENCIA")
    assert cfg.high_season_months is None
    assert cfg.rare_min_count is None


def test_override_declarado_se_aplica(monkeypatch):
    monkeypatch.setitem(
        vc.VARIETY_OVERRIDES,
        "ARA",
        {"high_season_months": (11, 12, 1), "imputer_knn_threshold": 0.45},
    )
    cfg = for_variety("ARA")
    assert cfg.high_season_months == (11, 12, 1)
    assert cfg.imputer_knn_threshold == 0.45
    assert cfg.rare_min_count is None  # lo no declarado sigue en default


def test_override_con_campo_invalido_lanza(monkeypatch):
    monkeypatch.setitem(vc.VARIETY_OVERRIDES, "MALA", {"campo_inexistente": 1})
    with pytest.raises(ValueError, match="campo_inexistente"):
        for_variety("MALA")


def test_feature_generator_respeta_meses_de_temporada():
    fechas = pd.Series(pd.to_datetime(["2024-07-15", "2024-12-15"]))
    # Default POP: julio es ALTA, diciembre es BAJA.
    out_pop = FeatureGenerator._date_features(fechas, add_year=False)
    assert out_pop["TEMPORADA_ALTA"].tolist() == [1, 0]
    assert out_pop["TEMPORADA_BAJA"].tolist() == [0, 1]
    # Variedad de pico veraniego invertido: diciembre es ALTA.
    out_ara = FeatureGenerator._date_features(
        fechas,
        add_year=False,
        high_season_months=(11, 12, 1),
        low_season_months=(5, 6, 7),
    )
    assert out_ara["TEMPORADA_ALTA"].tolist() == [0, 1]
    assert out_ara["TEMPORADA_BAJA"].tolist() == [1, 0]


def test_pipeline_factory_hornea_overrides_en_componentes():
    cfg = VarietyConfig(
        variety="ARA",
        high_season_months=(11, 12, 1),
        imputer_knn_threshold=0.45,
    )
    pipe = create_preprocessing_pipeline(cfg)
    fg = pipe.named_steps["feature_engineering"]
    assert fg.high_season_months == (11, 12, 1)
    assert pipe.named_steps["imputer"].fallback_threshold == 0.45


def test_pipeline_factory_sin_cfg_es_legacy():
    pipe = create_preprocessing_pipeline()
    fg = pipe.named_steps["feature_engineering"]
    assert fg.high_season_months is None  # None -> jun-oct en _date_features
    # El imputer conserva su default propio (env IMPUTER_KNN_THRESHOLD/0.30).
    assert pipe.named_steps["imputer"].fallback_threshold > 0
