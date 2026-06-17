"""
Catálogo de variedades soportadas
==================================
`Variety` (StrEnum) es la fuente única de verdad. Los nombres de los
miembros usan `_` cuando el valor original tiene guiones o espacios
(ej. `FCM14_057` -> `"FCM14-057"`); el VALOR persiste exactamente como
está en la base de datos y en el registry de MLflow.
"""

from __future__ import annotations

from enum import StrEnum


class Variety(StrEnum):
    """Variedades del catálogo de productividad."""

    ALBUS = "ALBUS"
    ARANA = "ARANA"
    ATLAS = "ATLAS"
    AVANTI = "AVANTI"
    AZRA = "AZRA"
    BELLA = "BELLA"
    BIANCA = "BIANCA"
    COLOSSUS = "COLOSSUS"
    EMERALD = "EMERALD"
    FALCO = "FALCO"
    FCM14_057 = "FCM14-057"
    FCM17_132 = "FCM17-132"
    FL_10_179 = "FL-10-179"
    FL_11_158 = "FL-11-158"
    JUPITER = "JUPITER"
    KEECRISP = "KEECRISP"
    KIRRA = "KIRRA"
    MAGICA = "MAGICA"
    MAGNUS = "MAGNUS"
    MEGACRISP = "MEGACRISP"
    MEGAEARLY = "MEGAEARLY"
    MEGAGEM = "MEGAGEM"
    MEGAGRAND = "MEGAGRAND"
    MEGAONE = "MEGAONE"
    MEGASTAR = "MEGASTAR"
    POP = "POP"
    RAYMI = "RAYMI"
    REGINA = "REGINA"
    ROSITA = "ROSITA"
    SEKOYA_BEAUTY = "SEKOYA BEAUTY"
    SEKOYA_POP = "SEKOYA POP"
    TERRAPIN = "TERRAPIN"
    VENTURA = "VENTURA"


def validate_variety(variety: str) -> Variety:
    """
    Valida y normaliza el nombre de una variedad.

    Args:
        variety: Nombre crudo (case-insensitive)

    Returns:
        Miembro `Variety` correspondiente (StrEnum, también usable como str).

    Raises:
        ValueError: Si la variedad no está en el catálogo.
    """
    normalized = str(variety).upper()
    try:
        return Variety(normalized)
    except ValueError as exc:
        valid = ", ".join(sorted(v.value for v in Variety))
        raise ValueError(
            f"Variedad '{variety}' no válida. Variedades disponibles: {valid}"
        ) from exc
