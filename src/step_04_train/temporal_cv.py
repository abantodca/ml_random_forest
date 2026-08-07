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


def recent_training_window(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    window_days: int | None,
    reference_date,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Recorta train a una ventana anterior a ``reference_date``.

    Devuelve también la máscara booleana, alineada posicionalmente con el
    train de entrada, para recortar labels auxiliares sin recalcular fechas.
    ``None`` conserva todas las filas y mantiene el comportamiento histórico.
    """
    if window_days is None:
        keep = pd.Series(True, index=X_train.index)
        return X_train, y_train, keep
    if window_days < 1:
        raise ValueError("training_window_days debe ser >= 1")
    if DATE_COLUMN not in X_train.columns:
        raise ValueError(f"Ventana temporal requiere columna '{DATE_COLUMN}'")

    dates = pd.to_datetime(X_train[DATE_COLUMN], errors="coerce")
    if dates.isna().any():
        raise ValueError("Ventana temporal no acepta fechas nulas o invalidas")
    reference = pd.Timestamp(reference_date)
    cutoff = reference - pd.Timedelta(days=window_days)
    keep = dates >= cutoff
    if not keep.any():
        raise ValueError(f"Ventana de {window_days} dias no deja filas antes de {reference.date()}")
    return X_train.loc[keep], y_train.loc[keep], keep


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

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            return self.n_splits
        years = self._extract_years(X)
        n_years = len(np.unique(years))
        return min(self.n_splits, max(0, n_years - self.min_train_years))

    def _iter_test_indices(self, X=None, y=None, groups=None):
        years = self._extract_years(X)
        unique_years = np.array(sorted(np.unique(years)))
        k = self.get_n_splits(X)
        if k <= 0:
            return
        for test_year in unique_years[-k:]:
            test_idx = np.where(years == test_year)[0]
            yield test_idx

    def split(self, X, y=None, groups=None) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        years = self._extract_years(X)
        unique_years = np.array(sorted(np.unique(years)))
        k = self.get_n_splits(X)
        if k <= 0:
            return
        for test_year in unique_years[-k:]:
            train_mask = years < test_year
            test_mask = years == test_year
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def _extract_years(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.year_col in X.columns:
                return X[self.year_col].astype(int).to_numpy()
            if DATE_COLUMN in X.columns:
                return pd.to_datetime(X[DATE_COLUMN]).dt.year.to_numpy()
        raise ValueError(
            f"TemporalYearSplit requiere columna '{self.year_col}' en X "
            f"(o '{DATE_COLUMN}' como fallback)."
        )


class PurgedDateSplit(BaseCrossValidator):
    """Expanding-window por fechas únicas, con separación temporal.

    A diferencia de ``TimeSeriesSplit`` aplicado a filas, todas las
    observaciones de una misma fecha quedan juntas. ``gap_periods`` purga las
    últimas fechas del train antes de cada validación.
    """

    def __init__(
        self,
        n_splits: int = 3,
        date_col: str = DATE_COLUMN,
        gap_periods: int = 1,
        min_train_periods: int = 30,
        min_test_periods: int = 10,
    ) -> None:
        self.n_splits = n_splits
        self.date_col = date_col
        self.gap_periods = gap_periods
        self.min_train_periods = min_train_periods
        self.min_test_periods = min_test_periods

    def _dates(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame) or self.date_col not in X.columns:
            raise ValueError(f"PurgedDateSplit requiere columna '{self.date_col}'")
        dates = pd.to_datetime(X[self.date_col], errors="coerce")
        if dates.isna().any():
            raise ValueError("PurgedDateSplit no acepta fechas nulas o inválidas")
        return dates.to_numpy(dtype="datetime64[ns]")

    def _boundaries(self, X: pd.DataFrame) -> list[tuple[int, int, int]]:
        dates = self._dates(X)
        unique_dates = np.unique(dates)
        available = len(unique_dates) - self.min_train_periods - self.gap_periods
        k = min(self.n_splits, max(0, available // self.min_test_periods))
        if k <= 0:
            return []

        test_size = max(self.min_test_periods, available // k)
        first_test = len(unique_dates) - k * test_size
        boundaries: list[tuple[int, int, int]] = []
        for fold in range(k):
            test_start = first_test + fold * test_size
            test_end = len(unique_dates) if fold == k - 1 else test_start + test_size
            train_end = test_start - self.gap_periods
            if train_end < self.min_train_periods:
                continue
            boundaries.append((train_end, test_start, test_end))
        return boundaries

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits if X is None else len(self._boundaries(X))

    def split(self, X, y=None, groups=None) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        dates = self._dates(X)
        unique_dates = np.unique(dates)
        for train_end, test_start, test_end in self._boundaries(X):
            train_dates = unique_dates[:train_end]
            test_dates = unique_dates[test_start:test_end]
            train_idx = np.flatnonzero(np.isin(dates, train_dates))
            test_idx = np.flatnonzero(np.isin(dates, test_dates))
            if train_idx.size and test_idx.size:
                yield train_idx, test_idx

    def _iter_test_indices(self, X=None, y=None, groups=None):
        for _, test_idx in self.split(X, y, groups):
            yield test_idx
