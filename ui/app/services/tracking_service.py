"""Caso de uso: seguimiento de precisión de los pronósticos.

Empareja lo PROYECTADO (pronósticos almacenados) con lo REAL (observaciones
históricas) por `(fundo, formato, fecha)` y, opcionalmente, descompone el
error re-prediciendo sobre el KG/HA real (dry-run, sin persistir).

No depende de Streamlit: lógica pura y testeable. La vista solo grafica lo
que este servicio devuelve.
"""

from __future__ import annotations

import io
from collections import defaultdict
from datetime import date

import pandas as pd

from app.client import endpoints
from app.client.api_client import ApiClient
from app.core import ApiConnectionError, ApiResponseError, logger
from app.schemas import (
    AccuracyPoint,
    ForecastRecord,
    HistoricalObservation,
    WeekAggregate,
)
from app.services.forecast_service import ForecastService
from app.services.payload_builder import build_prediction_payload


def _iso_week_label(fecha: str) -> str:
    """`'2026-04-10'` → `'2026-W15'` (año-semana ISO). '' si no parsea."""
    try:
        iso = date.fromisoformat(fecha[:10]).isocalendar()
    except (ValueError, TypeError):
        return ""
    return f"{iso.year}-W{iso.week:02d}"


def forecast_verdict(mape: float, bias: float) -> tuple[str, str]:
    """Traduce la precisión a una DECISIÓN gerencial (para directorio).

    Devuelve `(status, mensaje)` donde status ∈ {ok, warning, alert} para que
    la vista lo pinte como banner verde/amarillo/rojo. Convierte estadística
    (MAPE/sesgo) en una acción concreta, sin pedirle al directivo leer métricas.
    """
    sesgo = "sobreestima" if bias > 0 else "subestima"
    if mape < 10:
        return ("ok",
                f"✅ Pronóstico CONFIABLE (error típico {mape:.1f}%). "
                "Apto para decidir con el modelo.")
    if mape < 20:
        return ("warning",
                f"🟡 Confianza MODERADA (error {mape:.1f}%, {sesgo} ~{abs(bias):.2f}). "
                "Validar con criterio agronómico antes de decisiones grandes.")
    return ("alert",
            f"🔴 Precisión BAJA (error {mape:.1f}%). Revisar datos / reentrenar "
            "antes de decidir solo con el pronóstico.")


class TrackingService:
    def __init__(self, client: ApiClient) -> None:
        self._client = client
        self._forecasts = ForecastService(client)

    # ---- lectura de reales ----------------------------------------------

    def list_history(
        self, variety: str, *, limit: int = 5000
    ) -> list[HistoricalObservation]:
        data = self._client.get(
            endpoints.history_list(variety),
            timeout=self._client.timeout_read,
            params={"limit": limit},
        )
        return [
            HistoricalObservation.model_validate(it)
            for it in data.get("items", [])
        ]

    def upload_real_excel(
        self,
        variety: str,
        file_bytes: bytes,
        filename: str,
        *,
        replace: bool = True,
    ) -> dict:
        """Sube el Excel de datos REALES (mismas columnas del pronóstico +
        KG/JR_H). `replace=True` reemplaza el historial de la variedad.

        Devuelve el resumen del backend: {variety, inserted,
        skipped_invalid_rows, message}.
        """
        files = {
            "file": (
                filename,
                file_bytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        }
        path = f"{endpoints.history_upload(variety)}?replace={str(replace).lower()}"
        return self._client.post(
            path, timeout=self._client.timeout_batch, files=files,
        )

    def delete_history(self, variety: str) -> int:
        """Borra TODOS los datos reales de la variedad. Devuelve cuántos borró."""
        data = self._client.delete(
            endpoints.history_list(variety), timeout=self._client.timeout_write,
        )
        return int(data.get("deleted", 0))

    def replace_real_from_rows(self, variety: str, df: pd.DataFrame) -> dict:
        """Reemplaza TODO el real de la variedad con las filas editadas.

        Reconstruye el Excel desde la grilla y lo sube con replace=True
        (no hay PATCH por fila en el backend; editar = reemplazar el set).
        """
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="reales")
        buf.seek(0)
        return self.upload_real_excel(
            variety, buf.getvalue(), "reales_editado.xlsx", replace=True,
        )

    # ---- emparejamiento proyectado ↔ real --------------------------------

    def build_accuracy(
        self,
        variety: str,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 2000,
        with_decomposition: bool = True,
    ) -> list[AccuracyPoint]:
        """Pares (proyectado, real) ordenados por fecha.

        Solo incluye claves `(fundo, formato, fecha)` que existen en AMBOS
        lados. Si hay varios pronósticos para la misma clave, gana el más
        reciente (`created_at`). `with_decomposition` dispara una predicción
        dry-run por punto para separar error de datos vs error de modelo.
        """
        forecasts = self._forecasts.list(
            variety=variety, date_from=date_from, date_to=date_to, limit=limit,
        ).items
        real_map = {
            (h.fundo, h.formato, h.fecha[:10]): h
            for h in self.list_history(variety)
        }

        # Dedupe de pronósticos por clave, conservando el más reciente.
        latest: dict[tuple[str, str, str], ForecastRecord] = {}
        for fc in forecasts:
            key = (fc.fundo, fc.formato, fc.fecha[:10])
            prev = latest.get(key)
            if prev is None or fc.created_at > prev.created_at:
                latest[key] = fc

        points: list[AccuracyPoint] = []
        for key, fc in latest.items():
            hist = real_map.get(key)
            if hist is None:
                continue
            pred_on_real = (
                self._predict_on_real(variety, fc, hist)
                if with_decomposition
                else None
            )
            points.append(
                AccuracyPoint(
                    variety=variety,
                    fundo=fc.fundo,
                    formato=fc.formato,
                    fecha=fc.fecha[:10],
                    pred_original=fc.kghora_pred,
                    real=hist.kg_jr_h,
                    pred_on_real=pred_on_real,
                )
            )
        points.sort(key=lambda p: p.fecha)
        return points

    def _predict_on_real(
        self, variety: str, fc: ForecastRecord, hist: HistoricalObservation
    ) -> float | None:
        """Re-predice usando las features REALES del cosechado.

        Cada feature usa su valor real si el Excel de reales lo trae; si no,
        cae al valor proyectado del pronóstico. Cuando la observación real
        incluye todas las features, la descomposición es 100% exacta; si solo
        trae KG/HA, `error_data` aísla únicamente la proyección de KG/HA.
        """
        payload = build_prediction_payload(
            fecha=fc.fecha[:10],
            kg_ha=hist.kg_ha,
            dpc=hist.dpc if hist.dpc is not None else fc.dpc,
            ha=hist.ha if hist.ha is not None else fc.ha,
            dia_cosecha=(
                hist.dia_cosecha if hist.dia_cosecha is not None else fc.dia_cosecha
            ),
            fundo=fc.fundo,
            formato=fc.formato,
            indus_pct=hist.indus_pct if hist.indus_pct is not None else fc.indus_pct,
            p_baya=hist.p_baya if hist.p_baya is not None else fc.p_baya,
        )
        try:
            return self._forecasts.predict_dry(variety, payload).kghora
        except (ApiConnectionError, ApiResponseError) as exc:
            logger.warning("predict_on_real falló (%s/%s): %s", variety, fc.fecha, exc)
            return None

    # ---- agregación semanal (cierre de semana) ---------------------------

    @staticmethod
    def weekly_aggregate(points: list[AccuracyPoint]) -> list[WeekAggregate]:
        """Suma proyectada vs real por semana ISO, ordenada cronológicamente."""
        buckets: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        for p in points:
            wk = _iso_week_label(p.fecha)
            if not wk:
                continue
            acc = buckets[wk]
            acc[0] += p.pred_original
            acc[1] += p.real
            acc[2] += 1
        return [
            WeekAggregate(
                week=wk, proj_sum=acc[0], real_sum=acc[1], n=int(acc[2]),
            )
            for wk, acc in sorted(buckets.items())
        ]
