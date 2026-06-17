"""Construye payloads (dict en formato API) a partir de inputs del UI."""

from __future__ import annotations

from datetime import date

import pandas as pd


def build_forecast_payload(
    *,
    fecha: date,
    kg_ha: float,
    dpc: float,
    ha: float,
    dia_cosecha: int,
    formato: str,
    fundo: str,
    indus_pct: float | None = None,
    p_baya: float | None = None,
    horas: float | None = None,
    external_id: str | None = None,
) -> dict:
    """Payload para POST/PATCH de pronóstico (incluye opcionales > 0)."""
    payload: dict = {
        "FECHA": fecha.isoformat(),
        "KG/HA": kg_ha,
        "DPC": dpc,
        "HA": ha,
        "DIA_COSECHA": dia_cosecha,
        "FORMATO": formato,
        "FUNDO": fundo,
    }
    # `is not None` (no `> 0`): si el usuario escribió un valor explícito en el
    # form — incluso 0 — debe viajar al backend para que aparezca en el panel
    # de drift. El default vacío del form se traduce a None y queda fuera del
    # payload, dejando que el imputer del modelo rellene con la mediana.
    if indus_pct is not None:
        payload["%INDUS"] = indus_pct
    if p_baya is not None:
        payload["P/BAYA"] = p_baya
    if horas is not None:
        payload["HORAS_EFECTIVAS"] = horas
    if external_id:
        payload["EXTERNAL_ID"] = external_id
    return payload


def build_prediction_payload(
    *,
    fecha: date | str,
    kg_ha: float,
    dpc: float,
    ha: float,
    dia_cosecha: int,
    fundo: str,
    formato: str = "FRESCO",
    indus_pct: float | None = None,
    p_baya: float | None = None,
    horas_efectivas: float | None = None,
    external_id: str | None = None,
) -> dict:
    """Payload para predicción (dry-run) — acepta str|date, sin filtro `>0`."""
    payload: dict = {
        "FECHA": fecha.isoformat() if isinstance(fecha, date) else str(fecha),
        "KG/HA": kg_ha,
        "DPC": dpc,
        "HA": ha,
        "DIA_COSECHA": dia_cosecha,
        "FORMATO": formato,
        "FUNDO": fundo,
    }
    if indus_pct is not None:
        payload["%INDUS"] = indus_pct
    if p_baya is not None:
        payload["P/BAYA"] = p_baya
    if horas_efectivas is not None:
        payload["HORAS_EFECTIVAS"] = horas_efectivas
    if external_id:
        payload["EXTERNAL_ID"] = external_id
    return payload


def row_to_record(row: pd.Series) -> dict:
    """Convierte una fila del DataFrame batch al payload esperado por la API."""
    rec: dict = {
        "FECHA": str(row["FECHA"]),
        "KG/HA": float(row["KG/HA"]),
        "DPC": float(row["DPC"]),
        "HA": float(row["HA"]),
        "DIA_COSECHA": int(row["DIA_COSECHA"]),
        "FORMATO": str(row["FORMATO"]) if pd.notna(row.get("FORMATO")) else "FRESCO",
        "FUNDO": str(row["FUNDO"]),
    }
    for src_col, dst_key, caster in (
        ("%INDUS", "%INDUS", float),
        ("P/BAYA", "P/BAYA", float),
        ("HORAS_EFECTIVAS", "HORAS_EFECTIVAS", float),
        ("EXTERNAL_ID", "EXTERNAL_ID", str),
    ):
        val = row.get(src_col)
        if pd.notna(val) and val != "":
            rec[dst_key] = caster(val)
    return rec
