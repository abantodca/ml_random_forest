"""Cross-validator temporal por ANIO (expanding window).

Resuelve el problema diagnosticado por el EDA POP 2026-05-09: 5 de 6 features
numericas tienen drift severo entre anios consecutivos (PSI hasta 2.09 en
P/BAYA, 1.12 en DPC). El `StratifiedKFold` actual mezcla anios entre folds:
el modelo ve patrones de 2025-2026 en train y los reusa en test, lo que
infla artificialmente las metricas y oculta el riesgo en produccion.

Patron `expanding-window`:

    anios=[2022, 2023, 2024, 2025, 2026]
    fold 1: train={2022,2023}      test=2024
    fold 2: train={2022,2023,2024} test=2025
    fold 3: train={2022,...,2025}  test=2026

El primer fold requiere al menos UN ano en train (la primera ventana NO
puede ser solo 2022 porque entonces no existirian lags de cosechas previas
para construir features). Por eso el primer test es 2024 con n_splits=3:
deja 2022+2023 como warmup. n_splits se ajusta al numero de anios
disponibles si excede `n_years - 1`.

Uso (como outer CV en `tuning.py`):

    outer_cv = TemporalYearSplit(year_col="ANIO", n_splits=3, min_train_years=2)
    for train_idx, test_idx in outer_cv.split(X):
        ...
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection._split import BaseCrossValidator

from src.config import DATE_COLUMN


class TemporalYearSplit(BaseCrossValidator):
    """K folds tipo expanding-window por valor de columna ANIO.

    Parametros
    ----------
    year_col : str
        Columna en X que tiene el año (int). Default 'ANIO'.
    n_splits : int
        Numero deseado de folds. Se ajusta a `min(n_splits, n_years - min_train_years)`.
    min_train_years : int
        Minimo de anios que debe tener el train fold. Default 2 (dejar al
        modelo bootstrappear lags + skew detection con suficiente data).
    """

    def __init__(
        self,
        year_col: str = "ANIO",
        n_splits: int = 3,
        min_train_years: int = 2,
    ):
        self.year_col = year_col
        self.n_splits = n_splits
        self.min_train_years = min_train_years

    # -----------------------------------------------------------
    # API sklearn (BaseCrossValidator)
    # -----------------------------------------------------------
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            return self.n_splits
        years = self._extract_years(X)
        n_years = len(np.unique(years))
        return min(self.n_splits, max(0, n_years - self.min_train_years))

    def _iter_test_indices(self, X=None, y=None, groups=None):
        # BaseCrossValidator delega aqui desde split(); produce solo indices
        # de test, BaseCrossValidator construye el train por diferencia.
        # Pero queremos expanding window (no leave-one-out), asi que
        # OVERRIDE split() directamente y dejamos _iter_test_indices solo
        # para satisfacer el contrato.
        years = self._extract_years(X)
        unique_years = np.array(sorted(np.unique(years)))
        k = self.get_n_splits(X)
        if k <= 0:
            return
        # Test years son los ultimos k anios; cada uno es UN fold de test.
        for test_year in unique_years[-k:]:
            test_idx = np.where(years == test_year)[0]
            yield test_idx

    def split(self, X, y=None, groups=None) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        years = self._extract_years(X)
        unique_years = np.array(sorted(np.unique(years)))
        k = self.get_n_splits(X)
        if k <= 0:
            return
        # Test years son los ultimos k. Train fold = todos los años ANTES
        # del test year (expanding window).
        for test_year in unique_years[-k:]:
            train_mask = years < test_year
            test_mask = years == test_year
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    # -----------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------
    def _extract_years(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.year_col in X.columns:
                return X[self.year_col].astype(int).to_numpy()
            # Fallback: derivar de FECHA si la columna ANIO no esta presente
            if DATE_COLUMN in X.columns:
                return pd.to_datetime(X[DATE_COLUMN]).dt.year.to_numpy()
        raise ValueError(
            f"TemporalYearSplit requiere columna '{self.year_col}' en X "
            f"(o '{DATE_COLUMN}' como fallback)."
        )
