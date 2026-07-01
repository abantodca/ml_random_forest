"""Veredicto ejecutivo del modelo (4 niveles + thresholds)."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import REPORT_VERDICT_THRESHOLDS


@dataclass(frozen=True)
class Verdict:
    """Veredicto del modelo en formato presentable.

    icon      : emoji para badge (✅ / 🟢 / ⚠ / 🚫)
    title     : titulo corto para el hero (e.g. "Listo para producción")
    headline  : 1 frase de cabecera (e.g. "Modelo apto para uso operacional")
    body      : 1-2 frases con la recomendacion accionable
    color_key : 'green' | 'green-2' | 'amber' | 'red' (para CSS class)

    El nivel ('alta_confianza' | ... | 'no_recomendado') es la KEY del dict
    _VERDICTS; ningun consumidor lo leia como campo, asi que no se duplica aca.
    """

    icon: str
    title: str
    headline: str
    body: str
    color_key: str


_VERDICTS: dict[str, Verdict] = {
    "alta_confianza": Verdict(
        icon="✅",
        title="Listo para producción",
        headline="Modelo apto para integrarse al flujo operacional.",
        body=(
            "Los errores son consistentes y la diferencia entre lo que el "
            "modelo aprendió y lo que predice en datos nuevos es mínima. "
            "Recomendamos desplegarlo y monitorear mensualmente."
        ),
        color_key="green",
    ),
    "confianza_aceptable": Verdict(
        icon="🟢",
        title="Apto con monitoreo",
        headline="Modelo útil para producción con seguimiento cercano.",
        body=(
            "Los errores están dentro de límites aceptables. Recomendamos "
            "usarlo como apoyo a las decisiones operativas y revisar las "
            "métricas mensualmente para detectar deterioro."
        ),
        color_key="green-2",
    ),
    "confianza_limitada": Verdict(
        icon="⚠",
        title="Usar como referencia",
        headline="Modelo informativo, no como autoridad operativa.",
        body=(
            "El error es manejable pero hay señales de inconsistencia "
            "entre lo aprendido y lo predicho. Combinar siempre con "
            "criterio humano hasta acumular más datos o reentrenar."
        ),
        color_key="amber",
    ),
    "no_recomendado": Verdict(
        icon="🚫",
        title="No recomendado para producción",
        headline="Modelo con variabilidad alta — investigar antes de usar.",
        body=(
            "El error es elevado o la diferencia entre entrenamiento y "
            "predicción real es grande. Recomendamos NO desplegar hasta "
            "diagnosticar la causa (datos, features, segmentación)."
        ),
        color_key="red",
    ),
}


def compute_verdict(*, full_mape_pct: float, abs_gap: float) -> Verdict:
    """Devuelve el veredicto ejecutivo segun los umbrales de config.

    El modelo cae en el nivel MAS CONSERVADOR donde ambas metricas entren.
    Es decir, basta que UNA de las dos viole el limite del nivel para
    bajar al siguiente. Los thresholds se leen de
    `REPORT_VERDICT_THRESHOLDS` para poder tunearlos sin tocar codigo.
    """
    levels_in_order = [
        "alta_confianza",
        "confianza_aceptable",
        "confianza_limitada",
    ]
    for level in levels_in_order:
        thr = REPORT_VERDICT_THRESHOLDS[level]
        if full_mape_pct <= thr["max_mape_pct"] and abs_gap <= thr["max_abs_gap"]:
            return _VERDICTS[level]
    return _VERDICTS["no_recomendado"]
