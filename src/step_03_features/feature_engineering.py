"""Ingenieria de variables (sklearn-compat).

`FeatureGenerator` hace TODO el trabajo de transformar el DataFrame raw
(numericas + categoricas + columna fecha) en una matriz numerica lista
para el modelo:

    1. Deriva features ciclicas y `ANIO` desde la columna de fecha.
    2. Calcula ratios estructurales intra-fila (sin tiempo, sin target):
       KG_TOTAL, INDUS_KG_HA, KG_PER_BAYA, KG_HA_PER_DPC.
    3. Aplica one-hot encoding a las categoricas, memorizando categorias
       en `fit` para que `transform` produzca SIEMPRE las mismas columnas.
    4. Descarta la columna de fecha original.
    5. Devuelve un DataFrame con orden de columnas estable.

Codificacion ciclica de tiempo:
    sin(2*pi * x / period), cos(2*pi * x / period)
para que el modelo perciba que diciembre y enero estan adyacentes.

Ratios estructurales: combinaciones intra-fila que los lag features (que
operan a nivel grupo+temporal) no capturan. Son determinísticas por fila
y NO usan target ni H-EF, asi que no hay riesgo de leakage. Las divisiones
usan np.where(den > 0, num/den, NaN) y dejan que XGB/LGB redirijan los NaN
a su rama default por loss (tratamiento nativo de NaN en gradient boosting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    CATEGORICAL_FEATURES,
    DATE_COLUMN,
    ENABLE_CALENDAR_EXTRA,
    NUMERIC_FEATURES,
    SKEW_AUTO_DETECT,
    SKEW_DROP_RAW,
    SKEW_KURT_THRESHOLD,
    SKEW_THRESHOLD,
)
from src.step_03_features._helpers import safe_ratio


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """One-hot + ciclicas de fecha.

    Parametros
    ----------
    categorical_cols : list[str] | None
        Categoricas a codificar. None = `config.CATEGORICAL_FEATURES`.
    date_col : str | None
        Nombre de la columna de fecha. None = `config.DATE_COLUMN`.
        Si la columna no existe, no se generan derivadas temporales.
    add_year : bool
        Agrega columna `ANIO` (entero) para capturar deriva temporal.
    add_fundo_formato_interaction : bool
        Si FUNDO y FORMATO estan ambos en categorical_cols, agrega dummies
        para la combinacion FUNDO__FORMATO. Default False (uso standalone);
        en el pipeline real (`build_pipeline`) se pasa
        `ENABLE_FUNDO_FORMATO_INTERACTION` que es True por defecto tras
        validar gap -8% en prod_xl POP (2026-05-09). EDA POP mostro
        Cramer's V vs target = 0.29 (FUNDO) y 0.23 (FORMATO),
        V(FUNDO,FORMATO)=0.26 (no redundancia).
    """

    def __init__(
        self,
        categorical_cols: list[str] | None = None,
        date_col: str | None = None,
        add_year: bool = True,
        add_fundo_formato_interaction: bool = False,
        high_season_months: tuple | None = None,
        low_season_months: tuple | None = None,
    ):
        self.categorical_cols = categorical_cols
        self.date_col = date_col
        self.add_year = add_year
        self.add_fundo_formato_interaction = add_fundo_formato_interaction
        # Meses de temporada por VARIEDAD (P0.2, VarietyConfig). None =
        # legacy POP (jun-oct / dic-abr). Guardados verbatim (clone-safe)
        # y serializados con el pipeline: la API no depende de env.
        self.high_season_months = high_season_months
        self.low_season_months = low_season_months

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_categorical(self, X: pd.DataFrame) -> list[str]:
        cols = (
            self.categorical_cols
            if self.categorical_cols is not None
            else CATEGORICAL_FEATURES
        )
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"FeatureGenerator: categoricas inexistentes: {missing}")
        return list(cols)

    def _resolve_date_col(self, X: pd.DataFrame) -> str | None:
        col = self.date_col if self.date_col is not None else DATE_COLUMN
        return col if col in X.columns else None

    @staticmethod
    def _resolve_structural(X: pd.DataFrame) -> list[str]:
        """Decide que ratios se pueden calcular dado X. Defensivo: si una
        columna raw falta, su ratio simplemente no se genera."""
        names: list[str] = []
        if "KG/HA" in X.columns and "HA" in X.columns:
            names.append("KG_TOTAL")
        if "%INDUS" in X.columns and "KG/HA" in X.columns:
            names.append("INDUS_KG_HA")
        if "KG/HA" in X.columns and "P/BAYA" in X.columns:
            names.append("KG_PER_BAYA")
        if "KG/HA" in X.columns and "DPC" in X.columns:
            names.append("KG_HA_PER_DPC")
        return names

    @staticmethod
    def _detect_skew_features(X: pd.DataFrame) -> tuple[list[str], list[str], dict]:
        """Auto-detecta features que se beneficiarian de log1p o sqrt.

        Reglas (corren SOLO en `fit`, no en `transform`):
          - LOG1P: |skew| > SKEW_THRESHOLD (default 1.5) Y kurtosis razonable
            (< SKEW_KURT_THRESHOLD * 10, default 50). Clasica skewness moderada.
          - SQRT : kurtosis EXTREMA (> SKEW_KURT_THRESHOLD, default 50).
            Para distribuciones con outliers brutales en colas (ej. DPC en
            POP tenia kurt=158).
          - Skip si distribucion ya es ~simetrica.

        Tambien computa el SHIFT por columna (memo per-fit) para que
        train/test apliquen la misma transformacion sin importar
        diferencias de min entre splits.

        Devuelve (log1p_names, sqrt_names, shifts_dict).
        Ej: (["KG/HA_LOG1P", "%INDUS_LOG1P"], ["DPC_SQRT"], {"KG/HA": 0.0, ...})
        """
        log1p_names: list[str] = []
        sqrt_names: list[str] = []
        shifts: dict = {}

        candidates = [c for c in NUMERIC_FEATURES if c in X.columns]
        for c in candidates:
            v = pd.to_numeric(X[c], errors="coerce").dropna()
            if len(v) < 30 or v.std() == 0:
                continue
            sk = float(v.skew())
            kt = float(v.kurtosis())
            min_v = float(v.min())
            # Shift por columna: memoizado para evitar inconsistencia
            # train/test cuando cambian los rangos.
            shift = -min_v if min_v < 0 else 0.0

            if abs(kt) > SKEW_KURT_THRESHOLD:
                # Kurtosis extrema -> sqrt comprime mas las colas
                sqrt_names.append(f"{c}_SQRT")
                shifts[c] = shift
            elif abs(sk) > SKEW_THRESHOLD:
                # Skew moderada-alta -> log1p estabiliza
                log1p_names.append(f"{c}_LOG1P")
                shifts[c] = shift
            # else: skip (distribucion sana)

        return log1p_names, sqrt_names, shifts

    @staticmethod
    def _skew_mitigated(X: pd.DataFrame, names: list[str], shifts: dict) -> pd.DataFrame:
        """Aplica log1p / sqrt a las columnas listadas usando los SHIFTS
        memoizados del fit (no recomputa min sobre la data de transform).

        Esto garantiza que la misma fila produzca el mismo valor en train
        y en inference, aunque los rangos de los datasets difieran.
        """
        out = pd.DataFrame(index=X.index)
        for name in names:
            if name.endswith("_LOG1P"):
                src_col = name[: -len("_LOG1P")]
                v = pd.to_numeric(X[src_col], errors="coerce").astype(float)
                v = v + float(shifts.get(src_col, 0.0))
                # Defensivo en transform: si una fila nueva quedo < 0 tras el
                # shift del fit (data drift), clip a 0 para evitar log1p(-x).
                v = v.clip(lower=0.0)
                out[name] = np.log1p(v)
            elif name.endswith("_SQRT"):
                src_col = name[: -len("_SQRT")]
                v = pd.to_numeric(X[src_col], errors="coerce").astype(float)
                v = v + float(shifts.get(src_col, 0.0))
                v = v.clip(lower=0.0)
                out[name] = np.sqrt(v)
        return out

    @staticmethod
    def _structural_ratios(X: pd.DataFrame, names: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        if "KG_TOTAL" in names:
            out["KG_TOTAL"] = (X["KG/HA"].astype(float) * X["HA"].astype(float))
        if "INDUS_KG_HA" in names:
            out["INDUS_KG_HA"] = (X["%INDUS"].astype(float) * X["KG/HA"].astype(float))
        if "KG_PER_BAYA" in names:
            num = X["KG/HA"].astype(float).to_numpy()
            den = X["P/BAYA"].astype(float).to_numpy()
            out["KG_PER_BAYA"] = safe_ratio(num, den)
        if "KG_HA_PER_DPC" in names:
            num = X["KG/HA"].astype(float).to_numpy()
            den = X["DPC"].astype(float).to_numpy()
            out["KG_HA_PER_DPC"] = safe_ratio(num, den)
        return out

    @staticmethod
    def _date_features(series: pd.Series, add_year: bool,
                       trend_anchor: pd.Timestamp | None = None,
                       calendar_extra: bool = False,
                       high_season_months: tuple | None = None,
                       low_season_months: tuple | None = None) -> pd.DataFrame:
        """Features derivadas de FECHA: ciclicas (Fourier) + tendencia.

        `trend_anchor` (memorized en fit): timestamp inicial del dataset
        de training. En transform usamos el MISMO anchor para que
        `t_index` calculado en inference sea consistente con el del
        training (no leakage de "cuanto tiempo paso entre training y
        inference"). EDA POP 2026-05-09 detecto DW=0.19 + KPSS rechazo
        de estacionariedad: el target tiene tendencia que no fue
        capturada por las ciclicas + lags. Exponer t_index como feature
        permite a los arboles modelar tendencia explicitamente.
        """
        s = pd.to_datetime(series, errors="coerce")
        if s.isna().any():
            if s.dropna().empty:
                raise ValueError(
                    f"columna {series.name} de fechas totalmente vacia; "
                    "no se puede extraer features de fecha"
                )
            anchor = s.dropna().median()
            s = s.fillna(anchor)

        out = pd.DataFrame(index=series.index)
        month = s.dt.month.astype(int)
        week = s.dt.isocalendar().week.astype(int)

        # MES: armonicos Fourier orden 1-3. El orden 1 modela la onda anual;
        # 2 y 3 capturan asimetria observada (caida ene-abr, pico sep-oct).
        out["MES_SIN"] = np.sin(2 * np.pi * month / 12.0)
        out["MES_COS"] = np.cos(2 * np.pi * month / 12.0)
        out["MES_SIN2"] = np.sin(4 * np.pi * month / 12.0)
        out["MES_COS2"] = np.cos(4 * np.pi * month / 12.0)
        out["MES_SIN3"] = np.sin(6 * np.pi * month / 12.0)
        out["MES_COS3"] = np.cos(6 * np.pi * month / 12.0)

        out["SEMANA_SIN"] = np.sin(2 * np.pi * week / 52.0)
        out["SEMANA_COS"] = np.cos(2 * np.pi * week / 52.0)
        if calendar_extra:
            # Armonico 2 de SEMANA (paridad con MES que tiene orden 1-3):
            # captura asimetria intra-estacional fina que el orden 1 no ve.
            # El flag llega HORNEADO desde fit (self.calendar_extra_), no del
            # env del proceso que llama transform: el pickle es self-contained.
            out["SEMANA_SIN2"] = np.sin(4 * np.pi * week / 52.0)
            out["SEMANA_COS2"] = np.cos(4 * np.pi * week / 52.0)

        # DIA_SEM_SIN/COS removidas tras auditoria empirica (2026-05-05):
        # corr con KG/JR_H = +0.031 / -0.001 sobre 10073 filas POP. El η²=0.0014
        # original era universal (toda la senal cabia en ruido); con boosted
        # trees max_depth=4 las tomaba como splits aleatorios en folds chicos
        # y inflaba varianza entre trials de Optuna.

        # Temporada agronomica (referencia=TRANSICION: meses fuera de ambas
        # listas quedan en 0). Defaults POP: ALTA=jun-oct (z>0), BAJA=dic-abr
        # (z<0 por condicion de fruta). Otras variedades tienen otros picos:
        # los meses llegan por VarietyConfig (P0.2) via __init__.
        alta = list(high_season_months) if high_season_months else [6, 7, 8, 9, 10]
        baja = list(low_season_months) if low_season_months else [12, 1, 2, 3, 4]
        out["TEMPORADA_ALTA"] = month.isin(alta).astype(int)
        out["TEMPORADA_BAJA"] = month.isin(baja).astype(int)

        # FEATURES DE TENDENCIA (EDA POP 2026-05-09: DW=0.19 + KPSS rechaza)
        # t_index: dias desde el anchor (primer dia del training fold).
        # En inference, anchor es el del fit -> permite extrapolar en forma
        # consistente. Boosted tree puede splittar t_index para capturar
        # tendencia (e.g. "antes de t=400 KG/JR_H promedio era X, despues Y").
        if trend_anchor is not None:
            t_days = (s - trend_anchor).dt.days.astype(float)
            out["t_index_days"] = t_days
            out["t_index_years"] = t_days / 365.25

        if add_year:
            out["ANIO"] = s.dt.year.astype(int)
        return out

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None) -> FeatureGenerator:
        # Snapshot del flag en fit (self-contained pickle; ver _date_features).
        self.calendar_extra_ = ENABLE_CALENDAR_EXTRA
        cat_cols = self._resolve_categorical(X)
        date_col = self._resolve_date_col(X)

        self.categorical_cols_ = cat_cols
        self.date_col_ = date_col

        # Categorias memorizadas
        self.categories_ = {
            c: sorted(map(str, X[c].dropna().unique().tolist())) for c in cat_cols
        }

        # Interaccion FUNDO_FORMATO: solo si ambos estan presentes y la flag
        # esta activa. Memorizamos las combinaciones VISTAS en train (no el
        # producto cartesiano) para evitar dummies constantes. Combinaciones
        # nuevas en transform caen a la dummy "OTROS" (todas en 0).
        self.ff_categories_: list[str] = []
        if (
            self.add_fundo_formato_interaction
            and "FUNDO" in cat_cols and "FORMATO" in cat_cols
        ):
            ff = (
                X["FUNDO"].astype(str) + "__" + X["FORMATO"].astype(str)
            ).dropna()
            self.ff_categories_ = sorted(ff.unique().tolist())

        # Columnas que pasan tal cual: numericas (todo lo demas)
        passthrough = [
            c for c in X.columns if c not in cat_cols and c != date_col
        ]
        self.passthrough_cols_ = passthrough

        # Ratios estructurales: cuales se pueden calcular dadas las columnas
        # disponibles. Memorizado para que transform() produzca el mismo
        # set sin re-evaluar.
        self.structural_feature_names_ = self._resolve_structural(X)

        # Skew-mitigated features auto-detectadas (sin hardcoded list por
        # variedad): para cada numeric feature de NUMERIC_FEATURES, decide en
        # fit si necesita log1p (|skew|>threshold) o sqrt (kurt>threshold).
        # Tambien memoiza el SHIFT por columna para garantizar consistencia
        # train/test (sin esto, _skew_mitigated calculaba min per-call y la
        # misma fila daba valores distintos en train vs inference).
        if SKEW_AUTO_DETECT:
            log1p_names, sqrt_names, shifts = self._detect_skew_features(X)
            self.skew_feature_names_ = log1p_names + sqrt_names
            self.skew_shifts_ = shifts
        else:
            self.skew_feature_names_ = []
            self.skew_shifts_ = {}

        # SKEW_DROP_RAW (Fase B.5): cuando una columna tiene version
        # skew-mitigada (log1p/sqrt), su RAW sale del passthrough. Los
        # arboles son invariantes a transformaciones monotonas, asi que
        # mantener ambas solo duplica la columna (ruido para Optuna y
        # splits repartidos entre versiones en los importance reports).
        # La raw SIGUE en X para ratios estructurales y lags (esos se
        # calculan aparte); solo deja de salir como feature directa.
        if SKEW_DROP_RAW and self.skew_shifts_:
            dropped = set(self.skew_shifts_.keys())
            self.passthrough_cols_ = [
                c for c in self.passthrough_cols_ if c not in dropped
            ]
            passthrough = self.passthrough_cols_

        # Trend anchor: primer dia visto durante el fit. Se usa en transform
        # para que t_index_days/years sean consistentes entre training y
        # inference (relativos al mismo origen). Si date_col no esta, anchor
        # queda None y t_index_* no se generan.
        self.trend_anchor_ = None
        if date_col is not None:
            dates = pd.to_datetime(X[date_col], errors="coerce").dropna()
            if not dates.empty:
                self.trend_anchor_ = dates.min()

        # Derivadas de fecha (DIA_SEM_SIN/COS removidas: corr con target ~0).
        if date_col is not None:
            self.date_feature_names_ = (
                [
                    "MES_SIN", "MES_COS",
                    "MES_SIN2", "MES_COS2",
                    "MES_SIN3", "MES_COS3",
                    "SEMANA_SIN", "SEMANA_COS",
                ]
                + (["SEMANA_SIN2", "SEMANA_COS2"] if self.calendar_extra_ else [])
                + [
                    "TEMPORADA_ALTA", "TEMPORADA_BAJA",
                ]
                + (["t_index_days", "t_index_years"] if self.trend_anchor_ is not None else [])
                + (["ANIO"] if self.add_year else [])
            )
        else:
            self.date_feature_names_ = []

        # Frequency encoding (ENABLE_CALENDAR_EXTRA): proporcion del grupo en
        # train. Complementa el one-hot: le dice al arbol "cuanta historia
        # respalda esta fila" en una sola columna ordinal (refuerza el
        # sentinel -1 de cold-start de los lags). Categorias no vistas -> 0.
        self.freq_maps_: dict = {}
        self.freq_feature_names_: list[str] = []
        if self.calendar_extra_:
            for c in ("FUNDO", "FORMATO"):
                if c in cat_cols:
                    vc = X[c].astype(str).value_counts(normalize=True)
                    self.freq_maps_[c] = vc.to_dict()
                    self.freq_feature_names_.append(f"{c}_FREQ")

        # Dummies finales
        dummy_cols = [
            f"{c}__{cat}" for c in cat_cols for cat in self.categories_[c]
        ]
        ff_dummy_cols = [
            f"FUNDO_FORMATO__{cat}" for cat in self.ff_categories_
        ]
        self.feature_names_out_ = (
            passthrough
            + self.structural_feature_names_
            + self.skew_feature_names_
            + self.date_feature_names_
            + self.freq_feature_names_
            + dummy_cols
            + ff_dummy_cols
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy().reset_index(drop=True)

        passthrough_part = X[self.passthrough_cols_].copy()
        structural_part = self._structural_ratios(X, self.structural_feature_names_)
        skew_part = self._skew_mitigated(
            X,
            getattr(self, "skew_feature_names_", []),
            getattr(self, "skew_shifts_", {}),
        )

        if self.date_col_ is not None:
            date_part = self._date_features(
                X[self.date_col_], self.add_year,
                trend_anchor=getattr(self, "trend_anchor_", None),
                # getattr: pickles legacy (pre-fix) no tienen el atributo ->
                # False/None, que era el comportamiento default de entonces.
                calendar_extra=getattr(self, "calendar_extra_", False),
                high_season_months=getattr(self, "high_season_months", None),
                low_season_months=getattr(self, "low_season_months", None),
            )
        else:
            date_part = pd.DataFrame(index=X.index)

        dummy_frames: list[pd.DataFrame] = []
        for c in self.categorical_cols_:
            series = X[c].astype(str)
            dummies = pd.get_dummies(series, prefix=c, prefix_sep="__")
            for cat in self.categories_[c]:
                col_name = f"{c}__{cat}"
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
            keep = [f"{c}__{cat}" for cat in self.categories_[c]]
            dummy_frames.append(dummies[keep].astype(int))

        # Interaccion FUNDO_FORMATO: dummies solo de combinaciones vistas en
        # fit. Combinaciones nuevas en inference caen al sentinel implicito
        # (todas las dummies en 0).
        ff_categories = getattr(self, "ff_categories_", [])
        if ff_categories and "FUNDO" in X.columns and "FORMATO" in X.columns:
            ff_series = X["FUNDO"].astype(str) + "__" + X["FORMATO"].astype(str)
            ff_dummies = pd.get_dummies(
                ff_series, prefix="FUNDO_FORMATO", prefix_sep="__",
            )
            for cat in ff_categories:
                col_name = f"FUNDO_FORMATO__{cat}"
                if col_name not in ff_dummies.columns:
                    ff_dummies[col_name] = 0
            keep_ff = [f"FUNDO_FORMATO__{cat}" for cat in ff_categories]
            dummy_frames.append(ff_dummies[keep_ff].astype(int))

        freq_part = pd.DataFrame(index=X.index)
        for c, fmap in getattr(self, "freq_maps_", {}).items():
            freq_part[f"{c}_FREQ"] = (
                X[c].astype(str).map(fmap).fillna(0.0).astype(float)
            )

        out = pd.concat(
            [passthrough_part, structural_part, skew_part, date_part,
             freq_part, *dummy_frames],
            axis=1,
        )
        return out.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_out_)
