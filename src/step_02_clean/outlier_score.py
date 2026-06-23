"""LOF (Local Outlier Factor) como FEATURE adicional, no como filtro.

EDA POP 2026-05-09 detecto:
    - DPC kurt=158.98 (outliers extremos en colas)
    - KG/HA 9.1% outliers IQR (918 filas)

El OutlierCapper actual recorta valores extremos pero no informa al modelo
"esta fila es atipica multivariadamente". LOF mide cuanto se desvia una
fila respecto a la densidad local de sus vecinos: capta outliers que
viven en regiones del feature space alejadas, no solo univariados.

Inyectamos `lof_score` como feature mas. El boosted tree decide si
splittarlo o no — si efectivamente captura informacion, aparece en el
top de feature importance; si no, se ignora sin costo (max_depth limita).

Diferencia conceptual con OutlierCapper:
    - OutlierCapper: PROTEGE al modelo de outliers extremos (clip)
    - LOFScorer: INFORMA al modelo que la fila es atipica (additive)

Ambos coexisten sin conflicto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from src.config import NUMERIC_FEATURES
from src.step_02_clean._helpers import resolve_cols


class LOFOutlierScorer(BaseEstimator, TransformerMixin):
    """Agrega columna `lof_score` (negative_outlier_factor_) sobre numericas.

    Convencion sklearn: `negative_outlier_factor_` es **mas negativo** para
    outliers (closer to inlier => closer to -1). Convertimos a positivo
    mediante `score = -negative_outlier_factor` para que valores altos
    signifiquen "mas outlier" (intuitivo).

    Parametros
    ----------
    n_neighbors : int
        Vecinos del LOF. 20 estandar; 30-50 si dataset >10k filas.
    contamination : float | 'auto'
        Estimacion de fraccion de outliers. 'auto' = 0.1 default sklearn.
        Solo afecta el threshold de prediccion binaria, NO el score
        continuo que es lo que usamos.
    numeric_cols : list[str] | None
        Columnas a usar para distancias. None = NUMERIC_FEATURES.
    output_col : str
        Nombre de la columna de salida.

    Notas de implementacion
    -----------------------
    LOF requiere features sin NaN. El imputer corre ANTES en el pipeline
    cuando se inserta como step posterior. Si se invoca sobre data con
    NaN, falla con ValueError de sklearn.

    LOF requiere features escaladas comparablemente (igual que KNN). Usamos
    RobustScaler interno (mediana/IQR) para no estar dominados por la
    columna de mayor rango (KG/HA escala 100-1000 vs DPC 1-10).
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float | str = "auto",
        numeric_cols: list[str] | None = None,
        output_col: str = "lof_score",
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.numeric_cols = numeric_cols
        self.output_col = output_col

    def _resolve_cols(self, X: pd.DataFrame) -> list[str]:
        return resolve_cols(X, self.numeric_cols, NUMERIC_FEATURES, "LOFOutlierScorer")

    def fit(self, X: pd.DataFrame, y=None) -> LOFOutlierScorer:
        cols = self._resolve_cols(X)
        self.numeric_cols_ = cols

        X_num = X[cols].copy()
        # Si hay NaN al fit, imputamos con mediana inline (defensivo).
        # En el pipeline real este transformer va DESPUES del imputer, asi
        # que el path NaN no deberia activarse. Pero si alguien lo usa
        # standalone, no rompe.
        if X_num.isna().any().any():
            X_num = X_num.fillna(X_num.median(numeric_only=True))

        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X_num)

        # novelty=True permite usar `score_samples` en transform sobre data
        # nueva. novelty=False solo permite scoring durante fit (no util en
        # pipeline sklearn).
        n_samples = len(X_scaled)
        n_neigh = min(int(self.n_neighbors), max(2, n_samples - 1))
        self.lof_ = LocalOutlierFactor(
            n_neighbors=n_neigh,
            contamination=self.contamination,
            novelty=True,
        )
        self.lof_.fit(X_scaled)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = self.numeric_cols_

        X_num = X[cols].copy()
        if X_num.isna().any().any():
            X_num = X_num.fillna(X_num.median(numeric_only=True))
        X_scaled = self.scaler_.transform(X_num)

        # score_samples devuelve LOF scores: mas negativo => mas outlier.
        # Negamos para que mas alto => mas outlier.
        scores = self.lof_.score_samples(X_scaled)
        X[self.output_col] = -np.asarray(scores, dtype=float)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return base + [self.output_col]
