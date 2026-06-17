"""Servicio de orquestación de pronósticos.

Responsabilidad única (SRP): coordinar el flujo de negocio de un
pronóstico —construir features, predecir con MLflow, calcular drift,
persistir y ensamblar la respuesta— para que el router quede fino (solo
HTTP). Recibe los colaboradores ya construidos (MLflow, FeaturePipeline,
DriftService) por inyección; la sesión de DB se pasa por método porque es
per-request.

El cálculo de drift es metadata opcional: si falla, se loguea como
warning y la respuesta sigue siendo válida (nunca tumba el pronóstico).
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.schemas import (
    BatchDriftReport,
    DriftReport,
    ForecastCreate,
    ForecastListResponse,
    ForecastResponse,
    PredictionResponse,
)
from app.services.drift_service import DriftService
from app.services.feature_pipeline import FeaturePipeline
from app.services.mlflow_service import MLflowService

logger = logging.getLogger(__name__)


class ForecastService:
    """Orquesta predicción + drift + persistencia de pronósticos."""

    def __init__(
        self,
        mlflow: MLflowService,
        features: FeaturePipeline,
        drift: DriftService,
    ) -> None:
        self._mlflow = mlflow
        self._features = features
        self._drift = drift

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    async def create_one(
        self,
        db: AsyncSession,
        variety: str,
        forecast_data: ForecastCreate,
    ) -> ForecastResponse:
        """Crea un pronóstico individual.

        Flujo:
        1. FeaturePipeline construye el DataFrame raw (9 columnas).
        2. MLflow predice KGHORA (el `LagFeatureTransformer` interno del
           pipeline se encarga de los lags usando el historial picklado).
        3. DriftService compara las features contra el baseline derivado
           del propio pipeline serializado. El reporte se adjunta a la
           respuesta sin persistirlo.
        4. Persiste el pronóstico (incluye KGJN_PRED si hay HORAS_EFECTIVAS).
        """
        preds, stds, drift_reports, _ = await self._predict(
            variety, [forecast_data]
        )
        kghora_pred = preds[0]

        forecast = await crud.forecast.create_forecast(
            db, variety, forecast_data, kghora_pred
        )

        logger.info(
            "Forecast created: id=%d variety=%s kghora=%.4f",
            forecast.id, variety, kghora_pred,
        )
        response = ForecastResponse.model_validate(forecast)
        response = self._attach_uncertainty(
            response, kghora_pred, stds[0] if stds else None
        )
        return self._attach_drift(
            response, drift_reports[0] if drift_reports else None
        )

    async def create_batch(
        self,
        db: AsyncSession,
        variety: str,
        forecasts_data: list[ForecastCreate],
    ) -> ForecastListResponse:
        """Predice batch + persiste + adjunta drift por fila + drift agregado."""
        kghora_preds, stds, drift_reports, batch_drift_dict = await self._predict(
            variety, forecasts_data,
        )
        forecasts = await crud.forecast.create_forecasts_batch(
            db, variety, forecasts_data, kghora_preds
        )

        items: list[ForecastResponse] = []
        for idx, forecast in enumerate(forecasts):
            response = ForecastResponse.model_validate(forecast)
            response = self._attach_uncertainty(
                response,
                kghora_preds[idx],
                stds[idx] if stds and idx < len(stds) else None,
            )
            drift_dict = drift_reports[idx] if idx < len(drift_reports) else None
            items.append(self._attach_drift(response, drift_dict))

        batch_drift_report: BatchDriftReport | None = None
        if batch_drift_dict is not None:
            try:
                batch_drift_report = BatchDriftReport.model_validate(
                    batch_drift_dict
                )
            except Exception as exc:
                logger.warning("No se pudo serializar batch drift: %s", exc)

        return ForecastListResponse(
            items=items,
            total=len(items),
            limit=len(items),
            offset=0,
            batch_drift=batch_drift_report,
        )

    async def predict_only(
        self,
        variety: str,
        forecast_data: ForecastCreate,
    ) -> PredictionResponse:
        """Predice KGHORA SIN persistir (dry-run).

        Mismo pipeline que `create_one` (features → modelo → drift) pero no
        toca la base de datos. Sirve para:
          - Predicción exploratoria en el UI sin ensuciar la tabla forecasts.
          - Re-predecir sobre inputs reales en la descomposición de error
            (`error_modelo = predict(real) − real`).
        """
        preds, stds, drift_reports, _ = await self._predict(
            variety, [forecast_data]
        )
        kghora = preds[0]
        # KGJN en memoria (no hay fila que persistir); misma fórmula que el CRUD.
        kgjn = crud.forecast.calc_kgjn(kghora, forecast_data.horas_efectivas)

        drift: DriftReport | None = None
        drift_dict = drift_reports[0] if drift_reports else None
        if drift_dict is not None:
            try:
                drift = DriftReport.model_validate(drift_dict)
            except Exception as exc:
                logger.warning("No se pudo serializar reporte de drift: %s", exc)

        response = PredictionResponse(
            variety=variety, kghora_pred=kghora, kgjn_pred=kgjn, drift=drift,
        )
        return self._attach_uncertainty(
            response, kghora, stds[0] if stds else None
        )

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    async def _predict(
        self,
        variety: str,
        forecasts_data: list[ForecastCreate],
    ) -> tuple[
        list[float],
        list[float] | None,
        list[dict[str, Any] | None],
        dict[str, Any] | None,
    ]:
        """Construye el DataFrame raw (9 columnas), predice y calcula drift.

        El feature engineering (lags, ratios, cíclicas, one-hot) vive DENTRO
        del pipeline serializado en MLflow, así que el backend ya no necesita
        traer historial de Postgres ni calcular lags aquí: el
        `LagFeatureTransformer` empaquetado en el pickle se encarga.

        Retorna:
          - preds: KGHORA por fila.
          - stds: incertidumbre por fila (dispersion del ensemble interno);
            None con modelos legacy sin `predict_with_std`.
          - row_drifts: drift por fila (z-score + lookup categórico).
          - batch_drift: drift agregado del lote (PSI + K-S + Chi²) si
            n_filas ≥ 30, sino None.

        Si cualquier cálculo de drift falla, devuelve `None` en su slot y
        deja el resto del flujo intacto.
        """
        features_df = self._features.build_features(forecasts_data)
        preds, stds = await self._mlflow.predict_with_std(variety, features_df)

        # `DriftService.compute` y `compute_batch` son síncronos. En el camino
        # caliente (baseline ya cacheado por variedad) solo hacen aritmética
        # vectorial (NumPy/Pandas) — intensivo en CPU pero rápido para N típico.
        # En el camino frío (primera predicción de la variedad o tras un reload)
        # `_get_baseline` hace dos llamadas de red a MLflow
        # (get_latest_version_info + sklearn.load_model), que son bloqueantes.
        # Para no congelar el event loop en ninguno de los dos casos,
        # delegamos ambos cálculos al threadpool.
        loop = asyncio.get_running_loop()
        row_drifts = await loop.run_in_executor(
            None, self._drift.compute, variety, features_df
        )
        batch_drift = await loop.run_in_executor(
            None,
            partial(
                self._drift.compute_batch,
                variety,
                features_df,
                per_row_reports=row_drifts,
            ),
        )
        return preds, stds, row_drifts, batch_drift

    # Umbral del SEMIANCHO relativo (halfwidth/pred) para clasificar
    # confianza. El semiancho viene de MLflowService.predict_with_std:
    # conformal q90 por fundo (con factor cold-start x2) cuando el modelo
    # trae `conformal_`, o ±1.96·std del ensemble en modelos legacy.
    # Referencias POP 2026-06-11: A9 ~0.31 rel (media), LN ~0.37 (media),
    # cold-start C6 ~0.87 (baja — revisar manualmente).
    _CONF_MEDIA = 0.25
    _CONF_BAJA = 0.50

    @classmethod
    def _attach_uncertainty(cls, response, pred: float, halfwidth: float | None):
        """Inyecta banda de confianza en la respuesta (mismo patron que drift:
        metadata opcional que nunca tumba el response).

        `kghora_std` conserva el nombre por compatibilidad de schema pero
        contiene el SEMIANCHO de la banda (kghora_hi - kghora_pred)."""
        if halfwidth is None or pred is None or pred <= 0:
            return response
        try:
            rel = halfwidth / pred
            response.kghora_std = round(float(halfwidth), 4)
            response.kghora_lo = round(max(0.0, pred - halfwidth), 4)
            response.kghora_hi = round(pred + halfwidth, 4)
            response.confidence = (
                "baja" if rel > cls._CONF_BAJA
                else "media" if rel > cls._CONF_MEDIA
                else "alta"
            )
        except Exception as exc:
            logger.warning("No se pudo adjuntar incertidumbre: %s", exc)
        return response

    @staticmethod
    def _attach_drift(
        response: ForecastResponse, drift_dict: dict[str, Any] | None,
    ) -> ForecastResponse:
        """Inyecta el reporte de drift en la respuesta sin romper si es None."""
        if drift_dict is None:
            return response
        try:
            response.drift = DriftReport.model_validate(drift_dict)
        except Exception as exc:
            # Drift es metadata opcional: nunca debe tumbar el response.
            logger.warning("No se pudo serializar reporte de drift: %s", exc)
        return response
