"""P0 — Backend GPBoost: derivacion de grupos, pickle y contrato sklearn.

Cubre los riesgos especificos del backend gpb (los demas backends no los
tienen porque LightGBM/XGB picklan nativo y no derivan grupos):
  1. Derivacion de grupos desde dummies: categoria vista -> su etiqueta;
     dummy-block todo en 0 (categoria nueva en inferencia) -> __UNSEEN__.
  2. Pickle roundtrip del estimador FITEADO (el Booster de gpboost no es
     picklable directo; el wrapper serializa via save_model JSON) con
     predicciones bit-identicas.
  3. Las dummies de grupo NO entran a la matriz de arboles
     (drop_group_dummies=True default, brazo C2 del piloto).
  4. sample_weight se acepta (contrato TTR) pero no rompe el fit.
  5. El factory respeta el contrato del registry (TTR + espacio original).

Se salta solo si gpboost no esta instalado (contenedor api, p.ej.).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

gpb = pytest.importorskip("gpboost")

from src.step_04_train.model_gpb import (  # noqa: E402  (tras importorskip)
    _UNSEEN_LABEL,
    GPBoostMixedEffectsRegressor,
    get_gpb_model,
)


def _df_preprocesado(n: int = 240, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """DataFrame estilo output del preprocesador: numericas + dummies."""
    rng = np.random.RandomState(seed)
    fundo = rng.choice(["A9", "C6", "LN"], n)
    formato = rng.choice(["GRANEL", "CLAMSHELL"], n)
    df = pd.DataFrame({
        "KG/HA": rng.uniform(1000, 9000, n),
        "DPC": rng.uniform(50, 200, n),
        "lof_score": rng.normal(0, 1, n),
    })
    for cat in ["A9", "C6", "LN"]:
        df[f"FUNDO__{cat}"] = (fundo == cat).astype(int)
    for cat in ["GRANEL", "CLAMSHELL"]:
        df[f"FORMATO__{cat}"] = (formato == cat).astype(int)
    y = (
        df["KG/HA"] * 0.0004
        + (fundo == "A9") * 0.8
        + (formato == "GRANEL") * 0.3
        + rng.normal(0, 0.1, n)
    )
    return df, pd.Series(y)


def test_deriva_grupos_y_unseen():
    X, y = _df_preprocesado()
    model = GPBoostMixedEffectsRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)

    groups = model._derive_groups(X)
    assert groups.shape == (len(X), 2)
    # fila 0: la etiqueta derivada coincide con la dummy encendida
    row0_fundo = [c for c in X.columns if c.startswith("FUNDO__") and X[c].iloc[0] == 1]
    assert groups[0, 0] == row0_fundo[0].removeprefix("FUNDO__")

    # categoria NUEVA en inferencia: dummy-block todo en 0 -> __UNSEEN__
    X_new = X.head(1).copy()
    for c in X_new.columns:
        if c.startswith("FUNDO__"):
            X_new[c] = 0
    assert model._derive_groups(X_new)[0, 0] == _UNSEEN_LABEL
    # y predict no explota con el grupo no visto (cae al prior del GP)
    assert np.isfinite(model.predict(X_new)).all()


def test_dummies_de_grupo_fuera_de_la_matriz_de_arboles():
    X, y = _df_preprocesado()
    model = GPBoostMixedEffectsRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    assert not any(
        c.startswith(("FUNDO__", "FORMATO__", "FUNDO_FORMATO__"))
        for c in model.feature_cols_
    )
    assert "KG/HA" in model.feature_cols_


def test_pickle_roundtrip_predicciones_identicas():
    X, y = _df_preprocesado()
    model = GPBoostMixedEffectsRegressor(n_estimators=30, random_state=42)
    model.fit(X, y)
    pred_antes = model.predict(X)

    model2 = pickle.loads(pickle.dumps(model))
    pred_despues = model2.predict(X)

    np.testing.assert_allclose(pred_antes, pred_despues, rtol=0, atol=1e-12)


def test_sample_weight_se_ignora_sin_romper():
    X, y = _df_preprocesado()
    model = GPBoostMixedEffectsRegressor(n_estimators=10, random_state=42)
    sw = np.ones(len(X))
    model.fit(X, y, sample_weight=sw)  # gaussiana no soporta pesos: warn+ignore
    assert np.isfinite(model.predict(X)).all()


def test_factory_cumple_contrato_registry():
    from sklearn.base import clone
    from sklearn.compose import TransformedTargetRegressor

    ttr = get_gpb_model()
    assert isinstance(ttr, TransformedTargetRegressor)
    assert isinstance(ttr.regressor, GPBoostMixedEffectsRegressor)
    # clone-safe (requisito de OOFEnsembleRegressor y del inner CV)
    clone(ttr)

    # el TTR aplica log1p+cap en fit y expm1 en predict: el espacio original
    # del target se preserva end-to-end
    X, y = _df_preprocesado()
    y_kg = pd.Series(np.expm1(y))  # target positivo estilo KG/JR_H
    ttr = get_gpb_model(n_estimators=20)
    ttr.fit(X, y_kg)
    pred = ttr.predict(X)
    assert np.isfinite(pred).all()
    # error razonable en espacio original (senal sintetica fuerte)
    assert float(np.mean(np.abs(pred - y_kg))) < float(y_kg.std())
