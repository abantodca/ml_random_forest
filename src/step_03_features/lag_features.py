"""Features lag/agregadas por FUNDO+FORMATO, FUNDO y FORMATO usando historial.

Calcula medianas rolling de KG/JR_H y KG/HA por (FUNDO, FORMATO) y
tambien por FUNDO solo y FORMATO solo (mayor densidad para grupos
chicos), en ventanas de N OBSERVACIONES anteriores ordenadas por
FECHA. `shift(1)` excluye la fila actual.

Tambien agrega ratios "este dia vs su lag":
    KG_HA_ratio_30 = KG/HA actual / KG_HA_lag_FF_30

que capturan "la fila esta produciendo mejor/peor que su historico".

API
---
La forma canonica de usar este modulo es a traves de
`LagFeatureTransformer` (sklearn-compat) como PRIMER paso del Pipeline
de preprocesamiento. Asi:

    Pipeline([
        ("lag_features", LagFeatureTransformer()),
        ("missing_flags", MissingFlagger()),
        ...
    ])

El transformer:
  - En `fit(X, y)` memoriza el historial necesario (FUNDO, FORMATO,
    FECHA, KG/HA, target). Ese historial viaja serializado dentro del
    pipeline cuando MLflow guarda el modelo.
  - En `fit_transform(X, y)` ademas devuelve X con los 35 lag features
    calculados sobre el train fold (sin leakage cross-fold).
  - En `transform(X_new)` calcula lags para filas nuevas usando solo el
    historial memorizado. Permite que el backend serve solo necesite
    enviar las 9 columnas raw.

La funcion publica `add_lag_features(df)` se mantiene para compatibilidad
y como helper interno del transformer.

Mejora vs implementacion anterior (en data_loader, pre-CV)
----------------------------------------------------------
Antes los lags se calculaban sobre TODO el dataset antes del CV split, lo
que mezclaba info de test folds con train folds (leakage moderado). Con
el transformer adentro del pipeline, cada fold solo ve su propio train
para construir el historial.

Cold start
----------
Filas sin >=3 observaciones previas en su grupo reciben sentinel
`COLD_START_FILL_VALUE` (-1). Los modelos de arbol manejan -1 como una
hoja distinta sin necesidad de preprocessamiento adicional. Las flags
LAG_FF_COLD/LAG_FF_SEASONAL_COLD existian aqui pero se eliminaron tras
permutation_importance (mayo 2026): el sentinel -1 ya comunica el
cold-start a los arboles, la flag binaria era redundante.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    DATE_COLUMN,
    ENABLE_FEATURE_LAGS,
    ENABLE_SEASONAL_2Y,
    ENABLE_SIMPLE_LAGS,
    ENABLE_TARGET_VOLATILITY,
    EXANTE_MODE,
    LAG_LOG_DERIVED,
    TARGET,
)
from src.step_03_features._helpers import safe_ratio

logger = logging.getLogger(__name__)

WINDOWS: tuple[int, ...] = (7, 14, 30, 90)
MIN_PERIODS = 3
COLD_START_FILL_VALUE = -1.0
# Sentinel para features con rango real que cruza -1 (hoy: slope). Debe
# quedar fuera de todo valor fisico posible (min observado POP: -1189).
SLOPE_COLD_FILL_VALUE = -99999.0
KG_HA_COL = "KG/HA"

# Estabilizadores adicionales por FUNDO+FORMATO sobre KG/HA:
# - std rolling 30: VOLATILIDAD del grupo. Un FUNDO+FORMATO con std alta
#   es mas dificil de predecir; el arbol puede tratarlo distinto. Tambien
#   alimenta predict_with_std de OOFEnsembleRegressor.
# - slope rolling 30: regresion lineal de KG/HA contra t en ultimas 30 obs.
#   Captura momentum (alza/caida) que la mediana no ve.
# - days_since_last_FF: dias desde la cosecha previa en mismo FF. Senal de
#   cadencia agronomica.
# - REL_FORMATO_30: KG_HA_lag_F_30 / KG_HA_lag_FMT_30. Posicionamiento
#   relativo del fundo dentro de su cohorte de formato.
#
# EWMA halflife=15 + std_FF_90 fueron evaluados pero descartados: corr
# +0.98 con KG_HA_lag_FF_30 y +0.95 entre std_FF_30 y std_FF_90. Una sola
# ventana cubre la senal de volatilidad sin duplicar.
STD_WINDOW: int = 30
SLOPE_WINDOW: int = 30

# Lag estacional: ventana centrada en (fecha - 365d) con tolerancia +/-15d.
# Captura ciclo agronomico anual (mismo periodo del ano anterior por FUNDO+FORMATO).
# Ventana ±30d (wide) fue evaluada (2026-05-05) y descartada: corr +0.969
# con la ±15d sobre POP (cadencia diaria regular). La ventana mas amplia
# captura casi exactamente las mismas obs -> redundancia.
SEASONAL_PERIOD_DAYS: int = 365
SEASONAL_TOLERANCE_DAYS: int = 15

# Grupos a calcular: nombre_corto -> columnas de groupby
GROUP_DEFS: list[tuple[str, list[str]]] = [
    ("FF", ["FUNDO", "FORMATO"]),  # combinacion (mas especifica, menos densidad)
    ("F", ["FUNDO"]),  # solo fundo (mas densidad)
    ("FMT", ["FORMATO"]),  # solo formato
]


def _rolling_lag(
    df_sorted: pd.DataFrame, value_col: str, group_cols: list[str], window: int
) -> pd.Series:
    """Mediana rolling EXCLUYENDO la fila actual (shift(1) + rolling)."""
    return df_sorted.groupby(group_cols, sort=False)[value_col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=MIN_PERIODS).median()
    )


def _daily_series(df_sorted: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    """Serie DIARIA por grupo: 1 punto (mediana) por (grupo, dia).

    Base del modo ex-ante: el shift(1) posicional de `_rolling_lag` incluye
    filas hermanas del MISMO dia (mismo grupo y FECHA) — en nowcasting es
    valido (la cosecha del dia es conocida), en forecast ex-ante es leakage
    del propio evento. Colapsar a un punto por dia garantiza que el shift
    excluya TODO el dia actual.
    """
    return (
        df_sorted.groupby(group_cols + [DATE_COLUMN], sort=False)[value_col]
        .median()
        .reset_index()
        .sort_values(group_cols + [DATE_COLUMN])
    )


def _rolling_lag_exante(
    df_sorted: pd.DataFrame, value_col: str, group_cols: list[str], window: int
) -> pd.Series:
    """Mediana rolling sobre DIAS previos (excluye todo el dia actual).

    La ventana pasa a contar dias-con-observacion en vez de filas; misma
    semantica de MIN_PERIODS. Devuelve Series alineada con df_sorted.index.
    """
    daily = _daily_series(df_sorted, value_col, group_cols)
    daily["__lag"] = daily.groupby(group_cols, sort=False)[value_col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=MIN_PERIODS).median()
    )
    merged = df_sorted[group_cols + [DATE_COLUMN]].merge(
        daily.drop(columns=[value_col]), on=group_cols + [DATE_COLUMN], how="left"
    )
    return pd.Series(merged["__lag"].to_numpy(), index=df_sorted.index)


def _seasonal_lag_for_group(
    dates_d: np.ndarray,
    values: np.ndarray,
    period_days: int = SEASONAL_PERIOD_DAYS,
) -> np.ndarray:
    """Mediana estacional para UN grupo (FUNDO+FORMATO).

    Para cada fila i, mediana de `values` en filas cuya fecha cae en
    [date_i - (period+15)d, date_i - (period-15)d]. Asume `dates_d` ordenado
    ascendente. Devuelve NaN cuando hay <MIN_PERIODS observaciones en la
    ventana (típicamente filas del primer año del dataset).
    `period_days` permite el lag bienal (730) ademas del anual (365).
    """
    n = len(dates_d)
    out = np.full(n, np.nan, dtype=float)
    delta_lo = np.timedelta64(period_days + SEASONAL_TOLERANCE_DAYS, "D")
    delta_hi = np.timedelta64(period_days - SEASONAL_TOLERANCE_DAYS, "D")
    target_lo = dates_d - delta_lo
    target_hi = dates_d - delta_hi
    for i in range(n):
        lo_idx = np.searchsorted(dates_d, target_lo[i], side="left")
        hi_idx = np.searchsorted(dates_d, target_hi[i], side="right")
        if hi_idx - lo_idx >= MIN_PERIODS:
            out[i] = np.median(values[lo_idx:hi_idx])
    return out


# ---------------------------------------------------------------------------
# Helpers privados de add_lag_features. Cada uno muta `df_work` in-place
# (escribe columnas nuevas) y devuelve la lista de nombres agregados. La
# mutacion es deliberada para evitar copiar ~10k filas x 30 columnas en
# cada paso intermedio (cada call de pipeline.fit lo invoca).
# ---------------------------------------------------------------------------


def _compute_rolling_lags(df_work: pd.DataFrame, exante: bool = False) -> list[str]:
    """Lags rolling por grupo (FF, F, FMT) x valor (KG_JR_H, KG_HA) x ventana.

    Para cada (alias, group_cols) en GROUP_DEFS y cada ventana en WINDOWS,
    calcula la mediana rolling EXCLUYENDO la fila actual (shift(1) +
    rolling). Resultado: ~24 columnas (3 grupos x 2 valores x 4 ventanas).

    `exante`: usa la variante same-day-safe (`_rolling_lag_exante`) que
    excluye TODO el dia actual, no solo la fila (ver EXANTE_MODE en config).
    """
    lag_fn = _rolling_lag_exante if exante else _rolling_lag
    new_cols: list[str] = []
    for alias, group_cols in GROUP_DEFS:
        df_sorted = df_work.sort_values(group_cols + [DATE_COLUMN])
        for value_col, vname in [(TARGET, "KG_JR_H"), (KG_HA_COL, "KG_HA")]:
            for w in WINDOWS:
                name = f"{vname}_lag_{alias}_{w}"
                df_work.loc[df_sorted.index, name] = lag_fn(df_sorted, value_col, group_cols, w)
                new_cols.append(name)
    # NOTA: las flags LAG_FF_COLD y LAG_FF_SEASONAL_COLD existian aqui para
    # marcar filas sin historia. Se eliminaron tras permutation_importance
    # (mayo 2026) que mostro importance ~0 / negativa: el sentinel -1 ya
    # comunica el cold-start a los arboles, la flag binaria era redundante.
    return new_cols


def _compute_volatility_and_momentum_lags(df_work: pd.DataFrame, exante: bool = False) -> list[str]:
    """Volatilidad + momentum + cadencia por FUNDO+FORMATO (no usa target).

    Anade 3 features con senal unica verificada (corr <0.85 con cualquier
    rolling lag existente):

    - KG_HA_std_FF_30: desviacion estandar rolling con shift(1). Senal de
      volatilidad del grupo. Aporta capacidad de modular prediccion segun
      cuan estable es el FF y alimenta predict_with_std.
    - KG_HA_slope_FF_30: pendiente de regresion lineal KG/HA vs t en
      ultimas 30 obs (shift(1)). Captura momentum (alza/caida) que la
      mediana rolling no ve. Calculado sobre indices 0..n-1 de cada
      ventana, normalizado para que la unidad sea kg/HA por observacion.
    - days_since_last_FF: gap en dias hasta la observacion previa en el
      mismo FF. Cadencia agronomica: gap largo -> fruta mas madura.

    Operan solo sobre KG/HA y FECHA (no target) -> CV-safe sin logica
    adicional. KG/HA y FECHA estan disponibles en filas nuevas.
    """
    new_cols: list[str] = []
    df_sorted = df_work.sort_values(["FUNDO", "FORMATO", DATE_COLUMN])
    grouped_kgha = df_sorted.groupby(["FUNDO", "FORMATO"], sort=False)[KG_HA_COL]

    # Helper slope: recibe Series, devuelve Series de slopes (shift(1)
    # excluye self). Usa apply para mantener legibilidad; el costo es
    # aceptable porque la ventana es chica (30) y solo corre en
    # LagFeatureTransformer.fit_transform (no en cada predict).
    def _rolling_slope(s: pd.Series) -> pd.Series:
        s_shift = s.shift(1)
        return s_shift.rolling(SLOPE_WINDOW, min_periods=MIN_PERIODS).apply(
            _slope_of_window, raw=True
        )

    std_name = f"KG_HA_std_FF_{STD_WINDOW}"
    slope_name = f"KG_HA_slope_FF_{SLOPE_WINDOW}"

    if exante:
        # Modo ex-ante: std y slope sobre la serie DIARIA por FF (mismo
        # razonamiento que _rolling_lag_exante — el shift(1) por fila
        # incluia hermanas del dia actual). std diaria pierde la varianza
        # intra-dia, pero esa varianza es justamente la senal concurrente
        # que el ex-ante no puede ver.
        daily = _daily_series(df_sorted, KG_HA_COL, ["FUNDO", "FORMATO"])
        g_daily = daily.groupby(["FUNDO", "FORMATO"], sort=False)[KG_HA_COL]
        daily["__std"] = g_daily.transform(
            lambda s: s.shift(1).rolling(STD_WINDOW, min_periods=MIN_PERIODS).std()
        )
        daily["__slope"] = g_daily.transform(_rolling_slope)
        merged = df_sorted[["FUNDO", "FORMATO", DATE_COLUMN]].merge(
            daily.drop(columns=[KG_HA_COL]),
            on=["FUNDO", "FORMATO", DATE_COLUMN],
            how="left",
        )
        df_work.loc[df_sorted.index, std_name] = merged["__std"].to_numpy()
        df_work.loc[df_sorted.index, slope_name] = merged["__slope"].to_numpy()
    else:
        # 1) std rolling 30 sobre KG/HA por FF
        df_work.loc[df_sorted.index, std_name] = grouped_kgha.transform(
            lambda s: s.shift(1).rolling(STD_WINDOW, min_periods=MIN_PERIODS).std()
        )
        # 2) slope rolling 30: pendiente OLS sobre KG/HA(t) en ventana de 30
        df_work.loc[df_sorted.index, slope_name] = grouped_kgha.transform(_rolling_slope)

    new_cols.append(std_name)
    new_cols.append(slope_name)

    # 3) days_since_last_FF: diferencia en dias hasta la fila previa en
    #    mismo FF. Primera fila del grupo => NaN (cold-start, captado por
    #    sentinel -1 al final de add_lag_features).
    days_name = "days_since_last_FF"
    fechas_sorted = pd.to_datetime(df_sorted[DATE_COLUMN])
    diffs = (
        fechas_sorted.groupby([df_sorted["FUNDO"], df_sorted["FORMATO"]], sort=False)
        .diff()
        .dt.days.astype(float)
    )
    df_work.loc[df_sorted.index, days_name] = diffs.values
    new_cols.append(days_name)

    # 4) tenure_FUNDO: dias desde la PRIMERA observacion del FUNDO en el
    #    dataset. Captura "antiguedad" del fundo dentro del registro. Un
    #    fundo nuevo (tenure chico) puede tener manejo agronomico distinto
    #    a uno con anos de historia. Independiente de days_since_last_FF
    #    (que es gap entre cosechas consecutivas dentro de un FF).
    #
    #    LEAK NOTE: el min(FECHA) por FUNDO se calcula aqui sobre `df_work`
    #    (combined = history + new en transform). Si llega una fila con
    #    fecha anterior al min visto en fit (backfill), el origen se redefine
    #    y tenure cambia retroactivamente. LagFeatureTransformer.transform
    #    sobrescribe esta columna usando `self.fundo_first_seen_` memoizado
    #    en fit; el calculo aqui es solo el fallback para `fit_transform`
    #    y `add_lag_features` standalone (donde leak no aplica porque solo
    #    se ve el train).
    tenure_name = "tenure_FUNDO_days"
    fechas_full = pd.to_datetime(df_work[DATE_COLUMN])
    first_per_fundo = fechas_full.groupby(df_work["FUNDO"]).transform("min")
    df_work[tenure_name] = (fechas_full - first_per_fundo).dt.days.astype(float)
    new_cols.append(tenure_name)

    return new_cols


def _slope_of_window(arr: np.ndarray) -> float:
    """Pendiente OLS de arr[i] vs i (i=0..n-1). NaN-aware via mean centering.

    Cerrada algebraica para evitar la sobrecarga de scipy/sklearn:
        slope = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
    """
    n = len(arr)
    if n < MIN_PERIODS:
        return np.nan
    # Saltar NaN en y (raro porque KG/HA no tiene NaN post-imputer, pero
    # defensivo cuando el helper se llama via rolling.apply en init de fold)
    mask = ~np.isnan(arr)
    if mask.sum() < MIN_PERIODS:
        return np.nan
    y = arr[mask]
    x = np.arange(n, dtype=float)[mask]
    x_centered = x - x.mean()
    denom = (x_centered**2).sum()
    if denom <= 0.0:
        return np.nan
    return float((x_centered * (y - y.mean())).sum() / denom)


def _compute_simple_lags_and_diff(df_work: pd.DataFrame) -> list[str]:
    """Lags simples shift(1), shift(2) + diff(1) del target por (FUNDO, FORMATO).

    Complementan los rolling medians (que suavizan): PACF de POP muestra
    lag 1=0.50, lag 2=0.33 — los rolling 7d aplastan esa senal puntual.
    Diff(1) = shift(1) - shift(2): captura direccion del cambio entre las
    dos ultimas obs del FF. CV-safe por construccion (todo via shift, no
    usa valor actual del target).

    Bajos costos: 3 columnas (vs 24+ del rolling). Cold-start: NaN se
    rellena al final con sentinel -1.
    """
    new_cols: list[str] = []
    df_sorted = df_work.sort_values(["FUNDO", "FORMATO", DATE_COLUMN])
    grouped_target = df_sorted.groupby(["FUNDO", "FORMATO"], sort=False)[TARGET]

    # shift(1) y shift(2) del target por FF (ordenado por FECHA).
    shift_1 = grouped_target.transform(lambda s: s.shift(1))
    shift_2 = grouped_target.transform(lambda s: s.shift(2))

    for k, series in [(1, shift_1), (2, shift_2)]:
        name = f"KG_JR_H_lag_FF_simple_{k}"
        df_work.loc[df_sorted.index, name] = series
        new_cols.append(name)

    # diff(1) en lo previo (no usa target actual).
    diff_name = "KG_JR_H_diff_1_FF"
    df_work.loc[df_sorted.index, diff_name] = (shift_1 - shift_2).values
    new_cols.append(diff_name)

    return new_cols


def _compute_seasonal_lags(df_work: pd.DataFrame, seasonal_2y: bool = False) -> list[str]:
    """Lag estacional (mismo periodo del ano anterior, ventana +/-15d).

    Solo para grupo FUNDO+FORMATO. Captura ciclo agronomico anual que
    el rolling 90d no ve. Devuelve 2 columnas (KG_JR_H, KG_HA).

    Con ENABLE_SEASONAL_2Y agrega ademas el lag bienal (730d +/-15d) del
    target: corr intra-FF medida 2026-06-11 = +0.32 (n=4671), senal de
    alternancia que el lag anual (+0.48) no captura. Solo target: KG_HA
    a 2 anios aporto corr mas baja (+0.24) y alta redundancia.
    """
    new_cols: list[str] = []
    df_sorted_ff = df_work.sort_values(["FUNDO", "FORMATO", DATE_COLUMN])
    dates_d_all = pd.to_datetime(df_sorted_ff[DATE_COLUMN]).values.astype("datetime64[D]")
    specs: list[tuple[str, str, int]] = [
        (TARGET, "KG_JR_H_lag_FF_seasonal", SEASONAL_PERIOD_DAYS),
        (KG_HA_COL, "KG_HA_lag_FF_seasonal", SEASONAL_PERIOD_DAYS),
    ]
    if seasonal_2y:
        specs.append((TARGET, "KG_JR_H_lag_FF_seasonal_2y", 2 * SEASONAL_PERIOD_DAYS))
    group_indices = df_sorted_ff.groupby(["FUNDO", "FORMATO"], sort=False).indices
    for value_col, name, period in specs:
        seasonal_arr = np.full(len(df_sorted_ff), np.nan, dtype=float)
        vals_all = df_sorted_ff[value_col].values.astype(float)
        for _, pos_arr in group_indices.items():
            seasonal_arr[pos_arr] = _seasonal_lag_for_group(
                dates_d_all[pos_arr], vals_all[pos_arr], period_days=period
            )
        df_work.loc[df_sorted_ff.index, name] = seasonal_arr
        new_cols.append(name)
    return new_cols


def _compute_feature_lags(df_work: pd.DataFrame) -> list[str]:
    """Medianas rolling por FF de P/BAYA y DPC (ENABLE_FEATURE_LAGS).

    Justificacion (ACF intra-FF 2026-06-11): P/BAYA lag1=+0.74, DPC
    lag1=+0.61 — son las features de mayor drift entre anios y su historia
    de grupo es senal fuerte. El lag de P/BAYA ademas funciona como
    imputacion implicita superior: la raw tiene 39.2% NaN (imputada con
    mediana), pero su mediana rolling del FF refleja el nivel del grupo.
    rolling.median() ignora NaN, asi que el 39% missing no rompe el lag.
    """
    new_cols: list[str] = []
    df_sorted = df_work.sort_values(["FUNDO", "FORMATO", DATE_COLUMN])
    for value_col, vname in [("P/BAYA", "PBAYA"), ("DPC", "DPC")]:
        if value_col not in df_work.columns:
            continue
        for w in (7, 30):
            name = f"{vname}_lag_FF_{w}"
            df_work.loc[df_sorted.index, name] = _rolling_lag(
                df_sorted, value_col, ["FUNDO", "FORMATO"], w
            )
            new_cols.append(name)
    return new_cols


def _compute_target_volatility(df_work: pd.DataFrame) -> list[str]:
    """std rolling 30 del TARGET por FF (ENABLE_TARGET_VOLATILITY).

    Complementa KG_HA_std_FF_30: mide cuan predecible es el grupo en la
    unidad que el modelo intenta predecir. CV-safe: shift(1) excluye la
    fila actual; en transform las filas nuevas tienen TARGET=NaN y el
    rolling lo ignora (solo usa historia).
    """
    df_sorted = df_work.sort_values(["FUNDO", "FORMATO", DATE_COLUMN])
    name = f"KG_JR_H_std_FF_{STD_WINDOW}"
    df_work.loc[df_sorted.index, name] = df_sorted.groupby(["FUNDO", "FORMATO"], sort=False)[
        TARGET
    ].transform(lambda s: s.shift(1).rolling(STD_WINDOW, min_periods=MIN_PERIODS).std())
    return [name]


# Derivadas con colas pesadas que LAG_LOG_DERIVED comprime. Auditoria
# 2026-06-11 sobre POP (el OutlierCapper/LOF/skew solo cubren las 6 raw):
#   KG_HA_ratio_FF_30 kurt=383 max=55x | slope kurt=393 | days_since kurt=836.
# log1p es monotona y estateless (misma fila -> mismo valor train/inference)
# y el sentinel -1 (cold-start) queda fuera del rango de log1p(x>=0).
_LOG_RATIO_COLS = [
    "KG_HA_ratio_FF_30",
    "KG_HA_ratio_FF_90",
    "KG_HA_REL_GLOBAL_30",
    "KG_HA_REL_FORMATO_30",
]
_LOG_POSITIVE_COLS = [
    "days_since_last_FF",
    f"KG_HA_std_FF_{STD_WINDOW}",
    f"KG_JR_H_std_FF_{STD_WINDOW}",
]
_SIGNED_LOG_COLS = [f"KG_HA_slope_FF_{SLOPE_WINDOW}"]


def _apply_log_derived(df_work: pd.DataFrame, cols: list[str]) -> None:
    """Comprime colas de las derivadas listadas (solo las presentes en cols).

    - ratios (>0 por construccion): log1p(x)
    - positivas (days_since, std): log1p(x)
    - slope (signo informativo): sign(x) * log1p(|x|)

    Corre ANTES del sentinel fill (-1): los NaN cold-start se preservan y
    el -1 posterior sigue siendo distinguible (log1p(x>=0) >= 0).
    """
    for c in _LOG_RATIO_COLS + _LOG_POSITIVE_COLS:
        if c in cols and c in df_work.columns:
            v = df_work[c].astype(float)
            df_work[c] = np.log1p(v.clip(lower=0.0))
    for c in _SIGNED_LOG_COLS:
        if c in cols and c in df_work.columns:
            v = df_work[c].astype(float)
            df_work[c] = np.sign(v) * np.log1p(np.abs(v))


def _compute_ratios(df_work: pd.DataFrame) -> list[str]:
    """Ratios "actual vs algo": locales (vs propio lag) y global (vs pool).

    Locales (KG_HA solo: NO usa target):
        KG_HA_ratio_FF_30 = KG_HA_actual / KG_HA_lag_FF_30
        KG_HA_ratio_FF_90 = KG_HA_actual / KG_HA_lag_FF_90

    Global pool (vectorizado con rolling 30 obs sobre dataset ordenado por
    fecha, shift(1) excluye self):
        KG_HA_REL_GLOBAL_30 = KG_HA_actual / median(KG_HA en ultimas 30 obs
                              globales). Capta "este fundo rinde mejor o
                              peor que el promedio del mercado". Sesgo de
                              incluir self-fundo es chico (~1/n_fundos).
                              CV-safe via shift(1).

    Delta short/long del target (ratio entre lags FF, NO usa el target
    actual -> sigue siendo CV-safe):
        delta_KG_JR_H_30_90 = KG_JR_H_lag_FF_30 / KG_JR_H_lag_FF_90

    Requiere que `_compute_rolling_lags` ya haya escrito los lags FF/30 y
    FF/90 (de los cuales este helper depende).
    """
    new_cols: list[str] = []

    # Locales (KG_HA actual vs su lag FF)
    df_work["KG_HA_ratio_FF_30"] = safe_ratio(df_work[KG_HA_COL], df_work["KG_HA_lag_FF_30"])
    df_work["KG_HA_ratio_FF_90"] = safe_ratio(df_work[KG_HA_COL], df_work["KG_HA_lag_FF_90"])
    new_cols += ["KG_HA_ratio_FF_30", "KG_HA_ratio_FF_90"]

    # Global pool (KG_HA actual vs mediana cross-fundos rolling 30 obs)
    df_sorted_date = df_work.sort_values(DATE_COLUMN)
    rolling_global_30 = (
        df_sorted_date[KG_HA_COL].shift(1).rolling(30, min_periods=MIN_PERIODS).median()
    )
    df_work.loc[df_sorted_date.index, "_KG_HA_lag_GLOBAL_30"] = rolling_global_30
    df_work["KG_HA_REL_GLOBAL_30"] = safe_ratio(df_work[KG_HA_COL], df_work["_KG_HA_lag_GLOBAL_30"])
    df_work.drop(columns=["_KG_HA_lag_GLOBAL_30"], inplace=True)
    new_cols.append("KG_HA_REL_GLOBAL_30")

    # Delta short/long del target (entre lags, no leakage)
    df_work["delta_KG_JR_H_30_90"] = safe_ratio(
        df_work["KG_JR_H_lag_FF_30"], df_work["KG_JR_H_lag_FF_90"]
    )
    new_cols.append("delta_KG_JR_H_30_90")

    # Posicionamiento del FUNDO dentro de su cohorte de FORMATO en el lag 30:
    # KG_HA_lag_F_30 / KG_HA_lag_FMT_30 -> >1 si el fundo rinde por encima
    # del promedio de su formato en ese horizonte, <1 si por debajo. Auditado
    # vs los lag base: corr <0.5 -> senal independiente.
    df_work["KG_HA_REL_FORMATO_30"] = safe_ratio(
        df_work["KG_HA_lag_F_30"], df_work["KG_HA_lag_FMT_30"]
    )
    new_cols.append("KG_HA_REL_FORMATO_30")

    return new_cols


def _current_flags() -> dict:
    """Snapshot de los feature flags de lag leidos del entorno ACTUAL.

    Serializacion (fix 2026-06-11, hallazgo de la revision experta): los
    flags se leian del env en CADA llamada, tambien durante `transform` en
    inferencia. Un pipeline entrenado con un flag ON, servido por un proceso
    sin ese env (la API), producia columnas distintas a las esperadas por
    `feature_names_out_` -> skew silencioso o KeyError. Ahora el transformer
    HORNEA este snapshot en `fit` (self.flags_) y `transform` lo reusa: el
    pickle es self-contained.
    """
    return {
        "simple_lags": ENABLE_SIMPLE_LAGS,
        "target_volatility": ENABLE_TARGET_VOLATILITY,
        "feature_lags": ENABLE_FEATURE_LAGS,
        "seasonal_2y": ENABLE_SEASONAL_2Y,
        "log_derived": LAG_LOG_DERIVED,
        # Ex-ante (experimento #11): lags same-day-safe. Los pickles previos
        # a 2026-06-13 no tienen esta clave -> leer SIEMPRE con .get(False).
        "exante": EXANTE_MODE,
    }


def add_lag_features(df: pd.DataFrame, flags: dict | None = None) -> pd.DataFrame:
    """Orquestador thin: rolling lags + seasonal + ratios + sentinel fill.

    Devuelve `df` con orden original preservado. Las columnas agregadas son
    las listadas en `lag_output_columns(flags)`.

    `flags`: snapshot de feature flags (ver `_current_flags`). None = leer
    el entorno actual (uso standalone/notebooks); el transformer SIEMPRE
    pasa su snapshot horneado en fit.

    Pipeline:
        1. _compute_rolling_lags                 : FF/F/FMT x KG_JR_H/KG_HA x 7/14/30/90
        2. _compute_seasonal_lags                : FF x ano-1 +/-15d (y 2y opcional)
        3. _compute_volatility_and_momentum_lags : std + slope + days_since + tenure
        4. _compute_ratios                       : ratios FF/30,FF/90 + global + delta + REL_FORMATO
        5. Sentinel fill (-1)                    : reemplaza NaN cold-start
    """
    flags = flags if flags is not None else _current_flags()
    needed = [c for grp in GROUP_DEFS for c in grp[1]] + [DATE_COLUMN, TARGET, KG_HA_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"add_lag_features: columnas faltantes: {missing}")

    df_work = df.copy()
    # .get con default False: pickles serializados antes de 2026-06-13
    # tienen flags_ sin la clave "exante" (compat con rnd-forest-POP v1).
    exante = bool(flags.get("exante", False))
    new_cols: list[str] = []
    new_cols.extend(_compute_rolling_lags(df_work, exante=exante))
    if flags["simple_lags"]:
        # NOTA: los simple lags shift(1)/shift(2) son posicionales (incluyen
        # hermanas same-day) — no combinarlos con exante sin adaptarlos.
        new_cols.extend(_compute_simple_lags_and_diff(df_work))
    seasonal_cols = _compute_seasonal_lags(df_work, seasonal_2y=flags["seasonal_2y"])
    new_cols.extend(seasonal_cols)
    new_cols.extend(_compute_volatility_and_momentum_lags(df_work, exante=exante))
    if flags["target_volatility"]:
        new_cols.extend(_compute_target_volatility(df_work))
    if flags["feature_lags"]:
        new_cols.extend(_compute_feature_lags(df_work))
    new_cols.extend(_compute_ratios(df_work))
    if flags["log_derived"]:
        _apply_log_derived(df_work, new_cols)

    # Conteo de filas con cold-start (solo informativo para el log).
    n_cold_pre = int(df_work[[c for c in new_cols if "_lag_FF_" in c]].isna().all(axis=1).sum())
    n_cold_seasonal_pre = int(df_work[seasonal_cols].isna().all(axis=1).sum())

    # Sentinel en todas las features nuevas (incluyendo ratios). El -1 ya
    # le comunica al arbol que la fila es cold-start sin necesidad de flag.
    #
    # EXCEPCION (fix 2026-06-11): el slope tiene valores REALES negativos
    # (48% de filas <= -0.5, min -1189 en POP): el sentinel -1 colisionaba
    # con "grupo cayendo ~1 kg/HA por obs" y el arbol no podia distinguir
    # cold-start de declive real. Esas features usan un sentinel fuera de
    # cualquier rango fisico. El resto (ratios, lags, std, days) son >= 0
    # por construccion y -1 sigue siendo seguro.
    for c in new_cols:
        fill = SLOPE_COLD_FILL_VALUE if c.startswith("KG_HA_slope") else COLD_START_FILL_VALUE
        df_work[c] = df_work[c].fillna(fill)

    # DEBUG porque se llama por cada pipeline.fit dentro de Optuna nested CV
    # (~4500 veces en TUNING=prod). Subir a INFO temporal solo para diagnostico.
    logger.debug(
        f"Lag features agregadas | grupos={[g[0] for g in GROUP_DEFS]} | "
        f"cold_start_FF={n_cold_pre} ({n_cold_pre / len(df_work) * 100:.1f}%) | "
        f"cold_seasonal={n_cold_seasonal_pre} ({n_cold_seasonal_pre / len(df_work) * 100:.1f}%) | "
        f"n_nuevas_cols={len(new_cols)}"
    )
    return df_work


# ---------------------------------------------------------------------------
# Sklearn transformer wrapper
# ---------------------------------------------------------------------------
def lag_output_columns(flags: dict | None = None) -> list[str]:
    """Columnas que produce add_lag_features dado un snapshot de flags."""
    f = flags if flags is not None else _current_flags()
    return (
        [
            f"{vname}_lag_{alias}_{w}"
            for alias, _ in GROUP_DEFS
            for vname in ("KG_JR_H", "KG_HA")
            for w in WINDOWS
        ]
        + (
            # Solo se exponen los simple lags si el flag esta activo: con OFF
            # el pipeline reproduce exactamente el output del LGB v3 baseline
            # (75 cols antes de FUNDO_FORMATO interaction).
            ["KG_JR_H_lag_FF_simple_1", "KG_JR_H_lag_FF_simple_2", "KG_JR_H_diff_1_FF"]
            if f["simple_lags"]
            else []
        )
        + [
            "KG_JR_H_lag_FF_seasonal",
            "KG_HA_lag_FF_seasonal",
        ]
        + (["KG_JR_H_lag_FF_seasonal_2y"] if f["seasonal_2y"] else [])
        + [
            f"KG_HA_std_FF_{STD_WINDOW}",
            f"KG_HA_slope_FF_{SLOPE_WINDOW}",
            "days_since_last_FF",
            "tenure_FUNDO_days",
        ]
        + ([f"KG_JR_H_std_FF_{STD_WINDOW}"] if f["target_volatility"] else [])
        + (
            ["PBAYA_lag_FF_7", "PBAYA_lag_FF_30", "DPC_lag_FF_7", "DPC_lag_FF_30"]
            if f["feature_lags"]
            else []
        )
        + [
            "KG_HA_ratio_FF_30",
            "KG_HA_ratio_FF_90",
            "delta_KG_JR_H_30_90",
            "KG_HA_REL_GLOBAL_30",
            "KG_HA_REL_FORMATO_30",
        ]
    )


def _history_cols(flags: dict | None = None) -> list[str]:
    """Columnas raw que el transformer memoriza en history_. Con
    feature_lags activo tambien P/BAYA y DPC (sus lags requieren historia
    en inferencia, igual que KG/HA)."""
    f = flags if flags is not None else _current_flags()
    return ["FUNDO", "FORMATO", DATE_COLUMN, KG_HA_COL] + (
        ["P/BAYA", "DPC"] if f["feature_lags"] else []
    )


class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Stateful transformer que calcula lags rolling/seasonal/ratios.

    Encapsula `add_lag_features` para que se ejecute DENTRO del Pipeline
    de sklearn. Memoriza el historial necesario en `fit`; en `transform`
    consulta ese historial para calcular lags de filas nuevas.

    Diseño
    ------
    - `fit(X, y)`        : guarda `self.history_` con (FUNDO, FORMATO,
      FECHA, KG/HA, target) extraido de los datos de entrenamiento.
    - `fit_transform(X, y)` : ademas devuelve X con las columnas de lag
      agregadas (segun `lag_output_columns(self.flags_)`), calculadas sobre
      el propio train (esto es lo que ven los siguientes pasos del pipeline
      durante fit).
    - `transform(X_new)` : combina history + filas nuevas (con TARGET=NaN
      para las nuevas), llama a `add_lag_features` y devuelve solo las
      filas nuevas con lags. Para inferencia el caller no necesita
      conocer el historial.

    Comportamiento ante NaN del target
    ----------------------------------
    `pandas.rolling(...).median()` ignora NaN, asi que las filas nuevas
    no contaminan los lags de filas historicas (aunque esos resultados
    los descartamos igualmente).
    """

    def __init__(self) -> None:
        # Sin hiperparametros tuneables: ventanas/sentinel son constantes
        # globales del modulo. Mantener __init__ vacio respeta el contrato
        # sklearn (no se debe hacer trabajo aqui).
        pass

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------
    def _flags(self) -> dict:
        """Snapshot horneado en fit; fallback al env actual para pickles
        legacy (pre-fix) que no tienen flags_ — mismo comportamiento que
        antes del fix."""
        return getattr(self, "flags_", None) or _current_flags()

    def _validate_input(self, X: pd.DataFrame) -> None:
        missing = [c for c in _history_cols(self._flags()) if c not in X.columns]
        if missing:
            raise ValueError(
                f"LagFeatureTransformer: columnas requeridas faltantes en X: {missing}"
            )

    def _build_history(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """Crea snapshot historico minimo (FUNDO, FORMATO, FECHA, KG/HA, TARGET)."""
        history = X[_history_cols(self._flags())].copy()
        history[TARGET] = y.values if isinstance(y, pd.Series) else np.asarray(y, dtype=float)
        # Normalizamos el index para que pd.concat en transform no produzca
        # duplicados confusos.
        return history.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> LagFeatureTransformer:
        if y is None:
            raise ValueError(
                "LagFeatureTransformer.fit requiere y (KG/JR_H) para construir history_."
            )
        # Hornear el snapshot de flags ANTES de cualquier uso: el pipeline
        # serializado debe producir las mismas columnas en cualquier proceso
        # (API sin env vars incluida). Ver _current_flags.
        self.flags_ = _current_flags()
        self._validate_input(X)
        self.history_ = self._build_history(X, y)
        # Memoiza la fecha mas temprana vista por FUNDO durante el fit. Se usa
        # en `transform` para evitar leak temporal en tenure_FUNDO_days: sin
        # esto, si una fila nueva llega con fecha anterior al min visto en fit
        # (backfill), `transform("min")` sobre el dataframe combinado redefine
        # el origen del fundo -> tenure de filas historicas y nuevas inconsistente.
        fechas_fit = pd.to_datetime(X[DATE_COLUMN], errors="coerce")
        self.fundo_first_seen_: dict = fechas_fit.groupby(X["FUNDO"]).min().dropna().to_dict()
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        # 1. Memoriza historial.
        self.fit(X, y)
        # 2. Calcula lags sobre el propio train llamando a la implementacion
        #    canonica con (X + target). Devuelve X con las nuevas columnas.
        df = X.copy()
        df[TARGET] = y.values if isinstance(y, pd.Series) else np.asarray(y, dtype=float)
        df_with_lags = add_lag_features(df, self._flags()).drop(columns=[TARGET])
        # Cache de transient: permite que `final_pipeline.predict(X_train)`
        # (caso 'Aplicacion Total' en single_run.py) reutilice los lags ya
        # computados en fit_transform en vez de pasar por el camino `transform`
        # que duplicaria filas y produciria lags con leakage o ventana
        # diluida. Se descarta al picklear (`__getstate__`) para no inflar
        # el artifact MLflow ni filtrarse a inferencia.
        self._fit_X_ref_ = X
        self._fit_output_ = df_with_lags
        return df_with_lags

    def __getstate__(self):
        state = self.__dict__.copy()
        # Caches transient: no deben viajar en el pickle del modelo. En
        # inferencia, el LagFeatureTransformer recibe data nueva y `transform`
        # entra por el camino normal (history_ + filas nuevas).
        state.pop("_fit_X_ref_", None)
        state.pop("_fit_output_", None)
        return state

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "history_"):
            raise RuntimeError(
                "LagFeatureTransformer no fue ajustado. Llama fit/fit_transform primero."
            )
        # Atajo in-sample: si la pipeline llama transform con el MISMO objeto
        # que se uso en fit_transform, devolvemos los lags ya calculados.
        # Object identity (`is`) es estricto a proposito: cualquier copia
        # cae al camino normal.
        cached_X = getattr(self, "_fit_X_ref_", None)
        if cached_X is not None and X is cached_X:
            return self._fit_output_

        self._validate_input(X)

        X_work = X.copy().reset_index(drop=True)
        # __row_id preserva el orden original para reordenar al final.
        X_work["__row_id"] = np.arange(len(X_work))
        X_work["__is_new"] = True
        # add_lag_features requiere TARGET; en inferencia no lo tenemos.
        # NaN propaga correctamente por rolling.median (skipna).
        if TARGET not in X_work.columns:
            X_work[TARGET] = np.nan

        history = self.history_.copy()
        history["__row_id"] = -1
        history["__is_new"] = False

        # Alinear columnas: history tiene las minimas (_HISTORY_COLS+TARGET);
        # X_work puede traer mas columnas raw (DPC, %INDUS, etc). Para el
        # calculo solo importan _HISTORY_COLS+TARGET, asi que rellenamos las
        # faltantes en history con NaN.
        for col in X_work.columns:
            if col not in history.columns:
                history[col] = np.nan
        # Y a la inversa: si history trajera columnas que X_work no tiene
        # (no deberia pero defensivo), las descartamos.
        history = history[X_work.columns]

        combined = pd.concat([history, X_work], axis=0, ignore_index=True)
        combined_with_lags = add_lag_features(combined, self._flags())

        new_only = (
            combined_with_lags[combined_with_lags["__is_new"]]
            .sort_values("__row_id")
            .reset_index(drop=True)
        )

        # Override tenure_FUNDO_days usando el origen memoizado en fit.
        # add_lag_features lo recalculo sobre el combined (history + new) lo
        # cual reintroduce leak si llegan filas con fechas anteriores al
        # min(FUNDO) del fit (backfill). Aqui reescribimos con el dict del fit.
        # Fundo no visto en fit -> tenure=0 (cold-start: primera vez que se ve).
        if hasattr(self, "fundo_first_seen_") and "tenure_FUNDO_days" in new_only.columns:
            fechas_new = pd.to_datetime(new_only[DATE_COLUMN], errors="coerce")
            first_seen_series = new_only["FUNDO"].map(self.fundo_first_seen_)
            tenure_override = (fechas_new - first_seen_series).dt.days.astype(float)
            # Fundos no vistos en fit (NaN first_seen) -> tenure = 0
            new_only["tenure_FUNDO_days"] = tenure_override.fillna(0.0)

        # Limpiar helpers y el placeholder de TARGET.
        drop_cols = ["__row_id", "__is_new"]
        if TARGET in new_only.columns and TARGET not in X.columns:
            drop_cols.append(TARGET)
        return new_only.drop(columns=drop_cols)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        base: list[str] = list(input_features) if input_features is not None else []
        # Filtra TARGET si vino en input_features (no es output).
        base = [c for c in base if c != TARGET]
        return base + lag_output_columns(self._flags())
