"""Casos de uso del estado del servicio backend."""

from __future__ import annotations

from app.client import endpoints
from app.client.api_client import ApiClient
from app.client.mappers import to_health
from app.core import ApiConnectionError, ApiResponseError, logger
from app.schemas import ServiceHealth


class HealthService:
    def __init__(self, client: ApiClient) -> None:
        self._client = client

    def get(self) -> ServiceHealth | None:
        try:
            data = self._client.get(endpoints.HEALTH, timeout=self._client.timeout_health)
        except (ApiConnectionError, ApiResponseError) as exc:
            logger.warning("Health check fallido: %s", exc)
            return None
        return to_health(data)

    def reload_models(self) -> dict:
        try:
            return self._client.post(endpoints.RELOAD_MODELS, timeout=self._client.timeout_batch)
        except (ApiResponseError, ApiConnectionError) as exc:
            return {"error": str(exc)}
