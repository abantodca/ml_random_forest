"""Glosario unificado de terminos tecnicos -> definicion para lector no-tecnico.

Se renderiza tanto en el HTML (tooltips + tabla) como en el Excel
(hoja "Glosario"). Mantener corto y claro.
"""
from __future__ import annotations

GLOSSARY: dict[str, str] = {
    "MAPE": (
        "Error porcentual promedio. Por ejemplo, MAPE=15% significa que en "
        "promedio cada predicción se desvía un 15% del valor real."
    ),
    "MAE": (
        "Error absoluto promedio en las unidades originales (kg/jornal). "
        "Por ejemplo, MAE=5 significa que el modelo se equivoca en "
        "promedio en 5 kg por jornal."
    ),
    "R² (R cuadrado)": (
        "Porcentaje de la variabilidad de los datos que el modelo logra "
        "explicar. Va de 0 a 100%: 100% = modelo perfecto, 0% = el modelo "
        "no aporta nada vs predecir el promedio."
    ),
    "OOF (Out-Of-Fold)": (
        "Métrica calculada con predicciones que el modelo NUNCA vio "
        "durante su entrenamiento. Es la forma honesta de medir error: "
        "lo que esperamos en producción real."
    ),
    "In-sample / Aplicación Total": (
        "Métrica calculada cuando el modelo predice los mismos datos con "
        "los que se entrenó. Es OPTIMISTA — sirve solo como sanity check."
    ),
    "Train (Entrenamiento)": (
        "Datos que el modelo ve durante el aprendizaje. El error en train "
        "siempre es bajo; lo que importa es cómo se comporta en datos "
        "nuevos (Test)."
    ),
    "Test (Prueba)": (
        "Datos que el modelo NO vio en entrenamiento, usados para medir "
        "qué tan bien generaliza a casos nuevos."
    ),
    "Brecha Train-Test (gap, overfitting)": (
        "Diferencia entre el error en datos vistos vs datos nuevos. "
        "Brecha grande = el modelo memorizó el pasado pero no generaliza. "
        "Brecha chica = aprendió patrones reales y reproducibles."
    ),
    "KG/JR_H": (
        "Kilogramos cosechados por jornal-hora. Es la unidad técnica que "
        "predice el modelo (elimina el efecto de la duración de la jornada)."
    ),
    "KG/JR": (
        "Kilogramos cosechados por jornal completo. Es la unidad de "
        "negocio: KG/JR = KG/JR_H × duración de la jornada."
    ),
    "Nested Cross-Validation": (
        "Protocolo de evaluación donde la búsqueda de hiperparámetros "
        "ocurre dentro de cada partición, sin filtrar información del "
        "test. Es la forma rigurosa de medir generalización."
    ),
    "Baseline": (
        "Predicción ingenua usada como referencia (e.g., predecir siempre "
        "el promedio). Si un modelo no es mejor que el baseline, no "
        "aporta valor."
    ),
}


def glossary_terms() -> list[tuple[str, str]]:
    """Devuelve [(term, definition), ...] en orden de presentacion."""
    return list(GLOSSARY.items())
