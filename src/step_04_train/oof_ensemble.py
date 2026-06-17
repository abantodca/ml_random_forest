"""OOF Ensemble: K pipelines refiteados sobre folds del KFold.

`predict(X)` promedia las K predicciones. Reduce varianza del modelo
final sin cambiar el tuning (cada pipeline ve ~(K-1)/K del dataset; al
promediar, el ruido de cada estimador parcial se cancela parcialmente).

K=1 degenera al comportamiento legacy (refit unico sobre todo X). Esto
da rollback trivial: cambiar `OOF_ENSEMBLE_K=1` en config y los numeros
vuelven a ser bit-for-bit identicos a la version anterior.

La clase es sklearn-compatible (BaseEstimator + RegressorMixin) para
que joblib la serialice y `mlflow.sklearn.log_model` la trate como un
modelo opaco mas, sin necesidad de un flavor custom.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.utils.sklearn_helpers import (
    fit_with_optional_sample_weight,
    index_or_none,
)


class OOFEnsembleRegressor(BaseEstimator, RegressorMixin):
    """K pipelines refiteados; predict() promedia.

    Parametros
    ----------
    base_pipeline : Pipeline ya configurado con `set_params(**best_params)`.
                    Se clona K veces (una por fold). NO debe estar fiteado:
                    `clone` requiere un estimador unfitted.
    n_models : K. Si 1, hace refit unico sobre todo X (legacy).
    random_state : seed del KFold.

    Notas
    -----
    - Usa KFold (NO StratifiedKFold) en el refit final: el objetivo aqui
      es reducir varianza del predictor, no validacion insesgada. KFold
      tambien evita el riesgo de que la etiqueta de stratificacion del
      outer CV (max FUNDO/FORMATO) no soporte K splits si la variedad
      tiene pocos estratos.
    - `sample_weight` se splitea por fold (igual que en `_objective` del
      nested CV): cada pipeline recibe los pesos de SUS filas de train.
    """

    def __init__(
        self,
        base_pipeline: Pipeline,
        n_models: int = 5,
        random_state: int = 42,
    ):
        self.base_pipeline = base_pipeline
        self.n_models = n_models
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> OOFEnsembleRegressor:
        if self.n_models < 1:
            raise ValueError(
                f"n_models debe ser >=1, recibido {self.n_models}"
            )

        if self.n_models == 1:
            pipe = clone(self.base_pipeline)
            fit_with_optional_sample_weight(pipe, X, y, sample_weight)
            self.models_ = [pipe]
            return self

        cv = KFold(
            n_splits=self.n_models,
            shuffle=True,
            random_state=self.random_state,
        )
        models: list[Pipeline] = []
        for tr_idx, _ in cv.split(X):
            pipe = clone(self.base_pipeline)
            X_tr = X.iloc[tr_idx]
            y_tr = y.iloc[tr_idx]
            sw_tr = index_or_none(sample_weight, tr_idx)
            fit_with_optional_sample_weight(pipe, X_tr, y_tr, sw_tr)
            models.append(pipe)
        self.models_ = models
        return self

    def _stacked_preds(self, X) -> np.ndarray:
        """Stack de las K predicciones (filas x K modelos). Helper compartido
        por `predict` y `predict_with_std` para no duplicar el list comp."""
        return np.column_stack([m.predict(X) for m in self.models_])

    def predict(self, X) -> np.ndarray:
        return self._stacked_preds(X).mean(axis=1)

    def predict_with_std(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (mean, std) de las K predicciones.

        `std` por fila es un proxy directo de incertidumbre del modelo:
        cuando los K pipelines (entrenados en train folds distintos pero
        misma configuracion) coinciden, el modelo es confiable; cuando
        divergen, hay incertidumbre alta sobre esa prediccion.

        Util para construir intervalos en negocio:
            estimacion = mean
            banda_inf  = mean - 1.96 * std
            banda_sup  = mean + 1.96 * std

        Cuando `n_models=1`, std es 0 para todas las filas (un solo modelo
        no tiene varianza interna). En ese caso usar `predict()` directo.
        """
        preds = self._stacked_preds(X)
        # ddof=1 (sample std). Con K=1 devuelve NaN -> reemplazo a 0 para no
        # propagar NaN al consumer (es honesto: 1 modelo => 0 incertidumbre
        # entre modelos, que es trivialmente cierto).
        if preds.shape[1] <= 1:
            return preds.mean(axis=1), np.zeros(preds.shape[0])
        return preds.mean(axis=1), preds.std(axis=1, ddof=1)
