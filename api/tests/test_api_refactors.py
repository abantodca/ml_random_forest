from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from app import crud
from app.core.config import Settings
from app.core.excel_helpers import validate_excel_file
from app.schemas import (
    ForecastCreate,
    ForecastUpdate,
    HistoricalObservationCreate,
)
from app.services.forecast_service import ForecastService
from pydantic import ValidationError


def _forecast(**overrides):
    values = {
        "id": 10,
        "variety": "POP",
        "fecha": date(2026, 1, 10),
        "external_id": "A",
        "kg_ha": 5000.0,
        "indus_pct": 5.0,
        "dpc": 120.0,
        "p_baya": 2.5,
        "ha": 10.0,
        "dia_cosecha": 30,
        "formato": "CLAMSHELL 4.4 OZ",
        "fundo": "C5",
        "horas_efectivas": 8.0,
        "kghora_pred": 5.0,
        "kgjn_pred": 40.0,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_forecast_schema_rejects_values_outside_operational_contract() -> None:
    with pytest.raises(ValidationError):
        ForecastCreate.model_validate(
            {
                "FECHA": "2026-01-10",
                "KG/HA": 5000,
                "DPC": 401,
                "HA": 10,
                "DIA_COSECHA": 30,
                "FORMATO": "GRANEL",
                "FUNDO": "C5",
            }
        )


def test_patch_of_model_input_repredicts_and_updates_prediction(monkeypatch) -> None:
    existing = _forecast()
    monkeypatch.setattr(
        crud.forecast,
        "get_forecast_by_id",
        AsyncMock(return_value=existing),
    )

    async def update(_db, _forecast_id, update, *, kghora_pred=None):
        for key, value in update.model_dump(exclude_unset=True).items():
            setattr(existing, key, value)
        if kghora_pred is not None:
            existing.kghora_pred = kghora_pred
            existing.kgjn_pred = kghora_pred * existing.horas_efectivas
        return existing

    monkeypatch.setattr(crud.forecast, "update_forecast", update)
    service = ForecastService(mlflow=None, features=None, drift=None)  # type: ignore[arg-type]
    service._predict = AsyncMock(return_value=([7.5], [1.0], [None], None))

    response = asyncio.run(
        service.update_one(
            object(),  # type: ignore[arg-type]
            existing.id,
            ForecastUpdate.model_validate({"KG/HA": 6000.0}),
        )
    )

    assert response.kg_ha == 6000.0
    assert response.kghora_pred == 7.5
    assert response.kgjn_pred == 60.0
    service._predict.assert_awaited_once()


def test_patch_of_metadata_does_not_call_model(monkeypatch) -> None:
    existing = _forecast()
    monkeypatch.setattr(
        crud.forecast,
        "get_forecast_by_id",
        AsyncMock(return_value=existing),
    )

    async def update(_db, _forecast_id, update, *, kghora_pred=None):
        assert kghora_pred is None
        existing.external_id = update.external_id
        return existing

    monkeypatch.setattr(crud.forecast, "update_forecast", update)
    service = ForecastService(mlflow=None, features=None, drift=None)  # type: ignore[arg-type]
    service._predict = AsyncMock()

    response = asyncio.run(
        service.update_one(
            object(),  # type: ignore[arg-type]
            existing.id,
            ForecastUpdate.model_validate({"EXTERNAL_ID": "B"}),
        )
    )

    assert response.external_id == "B"
    service._predict.assert_not_awaited()


def test_dry_run_batch_predicts_once_and_preserves_order() -> None:
    rows = [
        ForecastCreate.model_validate(
            {
                "FECHA": f"2026-01-{day:02d}",
                "KG/HA": 5000 + day,
                "DPC": 120,
                "HA": 10,
                "DIA_COSECHA": 30,
                "FORMATO": "GRANEL",
                "FUNDO": "C5",
            }
        )
        for day in (10, 11)
    ]
    service = ForecastService(mlflow=None, features=None, drift=None)  # type: ignore[arg-type]
    service._predict = AsyncMock(
        return_value=([7.5, 8.5], [1.0, 1.2], [None, None], None)
    )

    result = asyncio.run(service.predict_batch_only("POP", rows))

    assert result.total == 2
    assert [item.kghora_pred for item in result.items] == [7.5, 8.5]
    service._predict.assert_awaited_once_with("POP", rows)


def test_history_empty_replace_is_rejected_before_delete() -> None:
    db = SimpleNamespace(execute=AsyncMock(), commit=AsyncMock(), rollback=AsyncMock())

    with pytest.raises(ValueError, match="se conserva"):
        asyncio.run(
            crud.historical_observation.import_rows(
                db,  # type: ignore[arg-type]
                "POP",
                [],
                replace=True,
            )
        )

    db.execute.assert_not_awaited()
    db.commit.assert_not_awaited()


def test_history_replace_commits_delete_and_insert_together() -> None:
    db = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(rowcount=3)),
        add_all=lambda rows: setattr(db, "added", rows),
        commit=AsyncMock(),
        rollback=AsyncMock(),
    )
    row = HistoricalObservationCreate.model_validate(
        {
            "FUNDO": "C5",
            "FORMATO": "GRANEL",
            "FECHA": "2026-01-10",
            "KG/HA": 5000,
            "KG/JR_H": 5,
        }
    )

    deleted, inserted = asyncio.run(
        crud.historical_observation.import_rows(
            db,  # type: ignore[arg-type]
            "POP",
            [row],
            replace=True,
        )
    )

    assert (deleted, inserted) == (3, 1)
    assert len(db.added) == 1
    db.commit.assert_awaited_once()
    db.rollback.assert_not_awaited()


def test_settings_reject_unsafe_cors_and_invalid_limits() -> None:
    base = {"database_url": "postgresql://user:pass@localhost/forecasts"}
    with pytest.raises(ValidationError):
        Settings(**base, cors_origins=["*"])
    with pytest.raises(ValidationError):
        Settings(**base, max_excel_file_size_mb=0)


def test_settings_ignore_variables_owned_by_other_services() -> None:
    settings = Settings(
        database_url="postgresql://user:pass@localhost/forecasts",
        s3_artifacts_bucket="shared-env-value",
    )
    assert settings.database_url.endswith("/forecasts")


def test_legacy_xls_is_rejected_when_reader_dependency_is_not_shipped() -> None:
    with pytest.raises(ValueError, match=r"\.xlsx"):
        validate_excel_file(b"legacy", "forecast.xls")
