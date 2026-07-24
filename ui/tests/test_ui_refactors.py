from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
from app.client.api_client import ApiClient
from app.core import API_BATCH_MAX_ROWS, ApiResponseError, Configuracion
from app.schemas import (
    ForecastListResult,
    ForecastRecord,
    HistoricalObservation,
    PredictionResult,
)
from app.services.batch_validation import BatchValidationError, validate_batch_upload
from app.services.forecast_service import ForecastService
from app.services.tracking_service import TrackingService


def _config() -> Configuracion:
    return Configuracion(
        api_url="http://api:8000",
        timeout_health=5,
        timeout_read=10,
        timeout_write=15,
        timeout_batch=60,
        cache_ttl_health=60,
        cache_ttl_varieties=120,
        cache_ttl_forecasts=30,
        log_level="INFO",
    )


def test_config_rejects_non_positive_timeout(monkeypatch) -> None:
    monkeypatch.setenv("TIMEOUT_READ", "0")
    with pytest.raises(ValueError, match="debe ser > 0"):
        Configuracion.desde_entorno()


def test_api_client_rejects_non_object_json(monkeypatch) -> None:
    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> list:
            return []

    monkeypatch.setattr("requests.request", lambda *args, **kwargs: Response())

    with pytest.raises(ApiResponseError, match="se esperaba un objeto"):
        ApiClient(_config()).get("/api/health", timeout=5)


def test_batch_larger_than_api_contract_is_rejected_before_requests() -> None:
    n_rows = API_BATCH_MAX_ROWS + 1
    df = pd.DataFrame(
        {
            "VARIEDAD": ["POP"] * n_rows,
            "FECHA": ["2026-01-10"] * n_rows,
            "KG/HA": [5000.0] * n_rows,
            "DPC": [120.0] * n_rows,
            "HA": [10.0] * n_rows,
            "DIA_COSECHA": [30] * n_rows,
            "FORMATO": ["GRANEL"] * n_rows,
            "FUNDO": ["C5"] * n_rows,
        }
    )

    with pytest.raises(BatchValidationError, match="errores de validación") as exc:
        validate_batch_upload(df, valid_varieties=["POP"])

    assert "exceden el máximo" in exc.value.issues[0].motivo


def test_forecast_batch_dry_maps_results_and_uses_one_request() -> None:
    client = MagicMock()
    client.timeout_batch = 60
    client.post.return_value = {
        "items": [
            {"variety": "POP", "kghora_pred": 7.5, "kgjn_pred": None},
            {"variety": "POP", "kghora_pred": 8.5, "kgjn_pred": None},
        ],
        "total": 2,
    }
    payloads = [{"FECHA": "2026-01-10"}, {"FECHA": "2026-01-11"}]

    result = ForecastService(client).predict_batch_dry("POP", payloads)

    assert [item.kghora for item in result] == [7.5, 8.5]
    client.post.assert_called_once()


def test_tracking_accuracy_uses_batch_prediction() -> None:
    service = TrackingService(MagicMock())
    forecast = ForecastRecord(
        id=1,
        variety="POP",
        fecha="2026-01-10",
        kg_ha=5000,
        dpc=120,
        ha=10,
        dia_cosecha=30,
        formato="GRANEL",
        fundo="C5",
        kghora_pred=7,
        created_at="2026-01-01T00:00:00Z",
    )
    history = HistoricalObservation(
        id=1,
        variety="POP",
        fecha="2026-01-10",
        kg_ha=5100,
        kg_jr_h=8,
        formato="GRANEL",
        fundo="C5",
    )
    service._forecasts.list = MagicMock(
        return_value=ForecastListResult(items=(forecast,), total=1, limit=1)
    )
    service.list_history = MagicMock(return_value=[history])
    service._forecasts.predict_batch_dry = MagicMock(
        return_value=(PredictionResult(variety="POP", kghora=7.8),)
    )

    points = service.build_accuracy("POP")

    assert len(points) == 1
    assert points[0].pred_on_real == 7.8
    service._forecasts.predict_batch_dry.assert_called_once()
