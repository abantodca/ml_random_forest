"""Casos de uso para pronósticos (CRUD, batch, predicción individual)."""

from __future__ import annotations

from app.client import endpoints
from app.client.api_client import ApiClient
from app.client.mappers import to_forecast, to_forecast_list
from app.core import ApiResponseError
from app.schemas import (
    DriftReport,
    ForecastListResult,
    ForecastRecord,
    PredictionResult,
)


class ForecastService:
    def __init__(self, client: ApiClient) -> None:
        self._client = client

    def list(
        self,
        *,
        variety: str | None = None,
        fecha: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ForecastListResult:
        params: dict = {"limit": limit, "offset": offset}
        if variety:
            params["variety"] = variety
        if fecha:
            params["fecha"] = fecha
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        data = self._client.get(
            endpoints.FORECASTS, timeout=self._client.timeout_read, params=params
        )
        return to_forecast_list(data)

    def create(self, variety: str, payload: dict) -> ForecastRecord:
        data = self._client.post(
            endpoints.forecast_create(variety),
            timeout=self._client.timeout_write,
            json=payload,
        )
        return to_forecast(data)

    def create_batch(self, variety: str, records: list[dict]) -> ForecastListResult:
        data = self._client.post(
            endpoints.forecast_batch(variety),
            timeout=self._client.timeout_batch,
            json={"forecasts": records},
        )
        return to_forecast_list(data)

    def update(self, forecast_id: int, payload: dict) -> ForecastRecord:
        data = self._client.patch(
            endpoints.forecast_by_id(forecast_id),
            timeout=self._client.timeout_write,
            json=payload,
        )
        return to_forecast(data)

    def delete(self, forecast_id: int) -> bool:
        data = self._client.delete(
            endpoints.forecast_by_id(forecast_id),
            timeout=self._client.timeout_write,
        )
        return data.get("deleted", 0) > 0

    def delete_by_fecha(self, fecha: str) -> int:
        data = self._client.delete(
            endpoints.forecasts_by_fecha(fecha),
            timeout=self._client.timeout_write,
        )
        return data.get("deleted", 0)

    def upload_excel(self, variety: str, file_bytes: bytes, filename: str) -> ForecastListResult:
        files = {
            "file": (
                filename,
                file_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        }
        data = self._client.post(
            endpoints.forecast_upload(variety),
            timeout=self._client.timeout_batch,
            files=files,
        )
        return to_forecast_list(data)

    def predict_dry(self, variety: str, payload: dict) -> PredictionResult:
        """Predicción dry-run: NO persiste (POST /forecasts/{variety}/predict).

        Para exploración sin ensuciar la tabla y para re-predecir sobre inputs
        reales en el seguimiento de precisión.
        """
        data = self._client.post(
            endpoints.forecast_predict(variety),
            timeout=self._client.timeout_write,
            json=payload,
        )
        return self._to_prediction_result(data, variety, payload)

    def predict_batch_dry(
        self,
        variety: str,
        payloads: list[dict],
    ) -> tuple[PredictionResult, ...]:
        """Predice hasta 1000 filas con una llamada y sin persistir."""
        if not payloads:
            return ()

        data = self._client.post(
            endpoints.forecast_predict_batch(variety),
            timeout=self._client.timeout_batch,
            json={"forecasts": payloads},
        )
        items = data.get("items")
        if not isinstance(items, list) or len(items) != len(payloads):
            raise ApiResponseError("Respuesta batch inválida: cantidad de predicciones inesperada.")
        return tuple(
            self._to_prediction_result(item, variety, payload)
            for item, payload in zip(items, payloads, strict=True)
        )

    @staticmethod
    def _to_prediction_result(
        data: dict,
        variety: str,
        payload: dict,
    ) -> PredictionResult:
        if not isinstance(data, dict):
            raise ApiResponseError("Respuesta de predicción inválida.")
        drift_raw = data.get("drift")
        return PredictionResult(
            variety=data.get("variety", variety),
            kghora=data.get("kghora_pred", 0.0),
            kgjn=data.get("kgjn_pred"),
            kghora_std=data.get("kghora_std"),
            kghora_lo=data.get("kghora_lo"),
            kghora_hi=data.get("kghora_hi"),
            confidence=data.get("confidence"),
            inputs=payload,
            drift=DriftReport.model_validate(drift_raw) if drift_raw else None,
        )
