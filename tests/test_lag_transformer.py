"""P0.1a — LagFeatureTransformer: serializacion self-contained y same-day.

Cubre los dos bugs por los que "ya nos quemamos":
  1. Flags horneados en fit (`flags_`): el pickle debe producir las MISMAS
     columnas aunque el proceso que deserializa tenga otros env flags
     (el bug de serializacion de 2026-06-11).
  2. Compat hacia atras: pickles previos a 2026-06-13 no tienen la clave
     "exante" en flags_ (rnd-forest-POP v1) y deben seguir funcionando.
  3. Modo ex-ante: el lag same-day-safe NO debe ver filas hermanas del
     mismo dia (leakage del evento en forecast).
"""
from __future__ import annotations

import pickle

import src.step_03_features.lag_features as lf
from src.step_03_features.lag_features import LagFeatureTransformer


def test_pickle_roundtrip_produce_mismas_columnas(df_raw_minimo):
    X, y = df_raw_minimo
    t = LagFeatureTransformer().fit(X, y)
    out_antes = t.transform(X)

    t2 = pickle.loads(pickle.dumps(t))
    out_despues = t2.transform(X)

    assert list(out_antes.columns) == list(out_despues.columns)
    assert out_antes.shape == out_despues.shape


def test_flags_horneados_ignoran_env_post_fit(df_raw_minimo, monkeypatch):
    X, y = df_raw_minimo
    # Fit con simple_lags ON -> el snapshot debe quedar horneado.
    monkeypatch.setattr(lf, "ENABLE_SIMPLE_LAGS", True)
    t = LagFeatureTransformer().fit(X, y)
    assert t.flags_["simple_lags"] is True

    # El "proceso de inferencia" tiene el flag OFF: transform debe seguir
    # produciendo las columnas del fit (self-contained), no las del env.
    monkeypatch.setattr(lf, "ENABLE_SIMPLE_LAGS", False)
    out = t.transform(X)
    assert "KG_JR_H_lag_FF_simple_1" in out.columns


def test_compat_pickle_viejo_sin_clave_exante(df_raw_minimo):
    X, y = df_raw_minimo
    t = LagFeatureTransformer().fit(X, y)
    # Simula un pickle serializado antes de 2026-06-13 (rnd-forest-POP v1).
    t.flags_.pop("exante", None)
    out = t.transform(X)  # no debe lanzar KeyError
    assert "KG_JR_H_lag_FF_7" in out.columns


def test_exante_lag_excluye_hermanas_same_day(df_raw_minimo, monkeypatch):
    X, y = df_raw_minimo

    # Nowcast (default): el shift(1) posicional mete a la hermana del mismo
    # dia en el lag -> las dos filas del 6to dia ven lags DISTINTOS.
    t_now = LagFeatureTransformer().fit(X, y)
    out_now = t_now.fit_transform(X, y)
    lags_now = out_now["KG_HA_lag_FF_7"].iloc[-2:].to_numpy()
    assert lags_now[0] != lags_now[1]

    # Ex-ante: serie diaria -> ambas hermanas ven SOLO los 5 dias previos
    # (mediana de 1..5 = 3), nunca el KG/HA=100 de su hermana.
    monkeypatch.setattr(lf, "EXANTE_MODE", True)
    t_ex = LagFeatureTransformer()
    out_ex = t_ex.fit_transform(X, y)
    lags_ex = out_ex["KG_HA_lag_FF_7"].iloc[-2:].to_numpy()
    assert lags_ex[0] == lags_ex[1] == 3.0
