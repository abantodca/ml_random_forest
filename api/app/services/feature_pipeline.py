"""Feature Pipeline (thin builder de DataFrame raw).

A partir de mayo 2026 el pipeline serializado en MLflow incluye
`LagFeatureTransformer` como primer paso, asi que el modelo recibe SOLO
las 9 columnas crudas (signature MLflow). El historial necesario para
calcular lags viaja DENTRO del pickle del modelo (memorizado en
`LagFeatureTransformer.history_` durante `fit`).

Por eso este servicio queda reducido a convertir los `ForecastCreate` del
cliente en un `pd.DataFrame` con el orden y nombres de columna que la
signature del modelo espera. El backend ya NO debe traer historial de
Postgres ni llamar a `add_lag_features`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from app.schemas import ForecastCreate

# Orden EXACTO de columnas de la signature MLflow del modelo entrenado
# (ver MLmodel.signature.inputs del run mas reciente). Cualquier cambio
# en el set de columnas raw del entrenamiento debe espejarse aqui.
MODEL_INPUT_COLUMNS: list[str] = [
    "KG/HA",
    "%INDUS",
    "DPC",
    "P/BAYA",
    "HA",
    "DIA_COSECHA",
    "FORMATO",
    "FUNDO",
    "FECHA",
]


class FeaturePipeline:
    """Builder stateless del DataFrame raw que el modelo MLflow espera."""

    def build_features(
        self,
        forecasts_data: Sequence[ForecastCreate],
    ) -> pd.DataFrame:
        """Convierte los forecasts del cliente al DataFrame de 9 columnas.

        El feature engineering completo (lags, ratios, ciclicas, one-hot)
        ya esta dentro del pipeline serializado en MLflow.
        """
        if not forecasts_data:
            return pd.DataFrame(columns=MODEL_INPUT_COLUMNS)

        df = pd.DataFrame(
            [
                {
                    "KG/HA": f.kg_ha,
                    "%INDUS": f.indus_pct,
                    "DPC": f.dpc,
                    "P/BAYA": f.p_baya,
                    "HA": f.ha,
                    # DIA_COSECHA en el schema es int pero la signature MLflow
                    # lo declara double; casteo explicito para que el enforce
                    # de schema no rechace el batch.
                    "DIA_COSECHA": float(f.dia_cosecha),
                    "FORMATO": f.formato,
                    "FUNDO": f.fundo,
                    "FECHA": pd.to_datetime(f.fecha),
                }
                for f in forecasts_data
            ]
        )
        return df[MODEL_INPUT_COLUMNS]
