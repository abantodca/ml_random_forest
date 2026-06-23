"""Dropper de features concurrentes para el modo EX-ANTE (experimento #11).

El modelo en produccion es NOWCASTING: KG/HA, %INDUS y sus derivadas
comparten el evento de cosecha con el target KG/JR_H — el MAPE OOF del
campeon esta condicionado a conocer la cosecha del dia. Este transformer,
ubicado al FINAL del pipeline de preprocesamiento (antes del filtro de
varianza), elimina esas columnas cuando EXANTE_MODE esta activo, dejando
solo informacion disponible ANTES del evento: lags historicos, calendario,
categoricas y derivadas lag-vs-lag.

Que se elimina (exactos + prefijos, cubre variantes _LOG1P/_SQRT):
    KG/HA, %INDUS              — valores raw del dia del evento
    KG_TOTAL, INDUS_KG_HA,
    KG_PER_BAYA, KG_HA_PER_DPC — ratios estructurales que usan KG/HA o %INDUS
    KG_HA_ratio_FF_30/_90      — actual / lag (usa el actual)
    KG_HA_REL_GLOBAL_30        — actual / pool global (usa el actual)
    %INDUS__MISS               — flag de missingness de una concurrente
                                 (en serving ex-ante seria constante)

Que se conserva: KG_HA_lag_* / KG_JR_H_lag_* (historicos), KG_HA_std/slope
(historicos; same-day-safe via flag exante del LagFeatureTransformer),
delta_KG_JR_H_30_90 y KG_HA_REL_FORMATO_30 (lag-vs-lag), days_since_last_FF
y tenure (conocidos del plan de cosecha), calendario y dummies.

NOTA: N_MISS_RAW sigue contando %INDUS en su suma — sesgo menor aceptado
para el experimento (la columna individual si se elimina).

El flag se HORNEA en fit (self.exante_) — mismo contrato self-contained
que LagFeatureTransformer.flags_: el pickle no relee env en transform.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import EXANTE_MODE

logger = logging.getLogger(__name__)

# Nombres exactos (no cubiertos limpiamente por prefijo).
_CONCURRENT_EXACT: frozenset[str] = frozenset({"KG/HA", "%INDUS", "%INDUS__MISS"})

# Prefijos: cubren la columna base y sus variantes skew (_LOG1P/_SQRT).
# OJO: "KG/HA" (con slash) NO matchea los lags "KG_HA_lag_*" (underscore).
_CONCURRENT_PREFIXES: tuple[str, ...] = (
    "KG/HA",
    "%INDUS",
    "KG_TOTAL",
    "INDUS_KG_HA",
    "KG_PER_BAYA",
    "KG_HA_PER_DPC",
    "KG_HA_ratio_FF_",
    "KG_HA_REL_GLOBAL_",
)


def _es_concurrente(col: str) -> bool:
    return col in _CONCURRENT_EXACT or col.startswith(_CONCURRENT_PREFIXES)


class ConcurrentFeatureDropper(BaseEstimator, TransformerMixin):
    """Elimina features del dia del evento cuando EXANTE_MODE esta activo.

    Con el flag OFF (default) es un passthrough puro — el pipeline de
    nowcasting no cambia en nada.
    """

    def fit(self, X: pd.DataFrame, y=None) -> ConcurrentFeatureDropper:
        self.exante_ = EXANTE_MODE
        self.cols_to_drop_ = [c for c in X.columns if _es_concurrente(c)] if self.exante_ else []
        if self.cols_to_drop_:
            # DEBUG: se dispara en cada pipeline.fit del nested CV (~4500
            # veces en prod) — mismo criterio que LagFeatureTransformer.
            logger.debug(
                f"EXANTE_MODE: eliminando {len(self.cols_to_drop_)} features "
                f"concurrentes: {self.cols_to_drop_}"
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "exante_", False):
            return X
        return X.drop(columns=[c for c in self.cols_to_drop_ if c in X.columns])

    def get_feature_names_out(self, input_features=None):
        feats = list(input_features) if input_features is not None else []
        if getattr(self, "exante_", False):
            return [c for c in feats if c not in set(self.cols_to_drop_)]
        return feats
