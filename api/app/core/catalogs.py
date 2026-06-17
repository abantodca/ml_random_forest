"""
Catálogos de valores válidos
=============================
Fuente única de verdad para FORMATO y FUNDO. Modelados como `StrEnum`
para que Pydantic los reconozca y los exponga en la spec OpenAPI sin
duplicar listas/sets manuales.

Las funciones `normalize_*` siguen existiendo porque el input externo
(Excel, JSON con espacios irregulares) puede no coincidir exactamente
con el valor del enum: normalizan ANTES de que Pydantic ejecute el cast.

Cualquier cambio aquí se propaga automáticamente a:
- Validación de Pydantic (`ForecastCreate` / `ForecastUpdate`)
- Validación de Excel/CSV en `excel_service`
- Endpoint `GET /api/varieties/catalogs` (consumido por el frontend)
"""

from __future__ import annotations

from enum import StrEnum


class Formato(StrEnum):
    """Presentación del producto. Catálogo cerrado."""

    CLAMSHELL_44 = "CLAMSHELL 4.4 OZ"
    CLAMSHELL_11 = "CLAMSHELL 11 OZ"
    CLAMSHELL_18 = "CLAMSHELL 18 OZ"
    GRANEL = "GRANEL"
    OTROS = "OTROS"


class Fundo(StrEnum):
    """Parcela / lote. Catálogo cerrado."""

    C5 = "C5"
    A9 = "A9"
    LN = "LN"
    C6 = "C6"


FORMATO_DEFAULT: Formato = Formato.CLAMSHELL_44


def normalize_formato(value: object) -> Formato:
    """Normaliza espacios y mayúsculas; devuelve el miembro enum o lanza ValueError."""
    if value is None:
        raise ValueError("FORMATO es obligatorio")
    candidate = " ".join(str(value).strip().upper().split())
    try:
        return Formato(candidate)
    except ValueError as exc:
        valid = ", ".join(f.value for f in Formato)
        raise ValueError(
            f"FORMATO '{value}' no válido. Valores aceptados: {valid}"
        ) from exc


def normalize_fundo(value: object) -> Fundo:
    """Normaliza espacios y mayúsculas; devuelve el miembro enum o lanza ValueError."""
    if value is None:
        raise ValueError("FUNDO es obligatorio")
    candidate = str(value).strip().upper()
    try:
        return Fundo(candidate)
    except ValueError as exc:
        valid = ", ".join(f.value for f in Fundo)
        raise ValueError(
            f"FUNDO '{value}' no válido. Valores aceptados: {valid}"
        ) from exc
