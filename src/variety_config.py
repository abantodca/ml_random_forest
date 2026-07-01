"""Configuracion POR VARIEDAD (P0.2 del plan docs/PLAN_REFACTOR_2026-06-12.md).

Problema que resuelve: `config.py` es global y varios umbrales son en
realidad conocimiento de POP (meses de temporada alta, umbral KNN validado
con P/BAYA de POP, minimo de muestra para colapsar categorias raras...).
Al escalar a la segunda variedad, esos valores se convierten en bugs
silenciosos. Ademas los env vars son globales al proceso: dos variedades
en el mismo run no podian tener settings distintos.

Diseno (explicito, no env):
  - `VarietyConfig` guarda SOLO overrides. `None` significa "usar el
    default global de hoy" (literal hardcodeado, env var o config.py,
    segun el knob) — asi el comportamiento actual de POP queda
    bit-identico y no se duplica la logica de defaults en dos lugares.
  - `variety_runner`/`single_run` obtienen el config con `for_variety()`
    y lo PASAN explicito al pipeline factory / nested CV / data loader.
    Los componentes sklearn guardan el valor en __init__ (clone-safe) y
    queda serializado dentro del pickle (mismo contrato self-contained
    que LagFeatureTransformer.flags_).
  - Precedencia: override de variedad > env var global > default de codigo.

Para una variedad nueva con estacionalidad distinta, agregar una entrada:

    VARIETY_OVERRIDES["ARANDANO_X"] = {
        "high_season_months": (11, 12, 1),
        "sample_weight_high_season_months": (11, 12, 1),
    }

Antes de fijar overrides, re-validar la evidencia por variedad (los
umbrales de POP salieron de su EDA/ACF; no extrapolar sin medir).
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class VarietyConfig:
    """Overrides por variedad. None = usar el default global de hoy.

    Campos y donde se resuelven (consumidor del None):
      high_season_months / low_season_months:
          dummies TEMPORADA_ALTA/BAJA en FeatureGenerator
          (default literal POP: jun-oct / dic-abr).
      sample_weight_high_season_months:
          boost de temporada alta en tuning._maybe_sample_weights
          (default: env SAMPLE_WEIGHT_HIGH_SEASON_MONTHS, hoy "8,9,10").
      sample_weight_high_season:
          toggle POR VARIEDAD del boost de temporada alta. None = usa el env
          global SAMPLE_WEIGHT_HIGH_SEASON (hoy OFF). Permite activar el boost
          solo en variedades donde ayuda, sin encenderlo para todo el proceso.
      imputer_knn_threshold:
          fallback_threshold de CustomKNNImputer
          (default: env IMPUTER_KNN_THRESHOLD, hoy 0.30).
      rare_min_count:
          colapso de categorias raras a 'OTROS' en data_loader
          (default: config.RARE_MIN_COUNT, hoy 50).
    """

    variety: str
    high_season_months: tuple[int, ...] | None = None
    low_season_months: tuple[int, ...] | None = None
    sample_weight_high_season_months: tuple[int, ...] | None = None
    sample_weight_high_season: bool | None = None
    imputer_knn_threshold: float | None = None
    rare_min_count: int | None = None


# Overrides explicitos por hoja del Excel. Variedad sin entrada = defaults
# globales (hoy DATA-DRIVEN: temporada autodetectada, capacidad/folds por n).
#
# POP fija sus meses de temporada EXPLICITAMENTE (2026-07-01): son los valores
# historicos validados con su EDA. Sin este pin, SEASON_AUTODETECT derivaria
# (7,8,9,10)/(1,2,3,4) — parecido pero NO identico al literal (6-10)/(12-4) —
# y cambiaria silenciosamente el comportamiento de la variedad de referencia.
# Exactamente el escenario que el comentario original de este dict advertia.
VARIETY_OVERRIDES: dict[str, dict[str, object]] = {
    "POP": {
        "high_season_months": (6, 7, 8, 9, 10),
        "low_season_months": (12, 1, 2, 3, 4),
    },
}

_VALID_FIELDS = {f.name for f in fields(VarietyConfig)} - {"variety"}


def for_variety(variety: str) -> VarietyConfig:
    """Config de la variedad: defaults globales + overrides declarados.

    Variedad sin entrada en VARIETY_OVERRIDES -> todos los campos None
    (defaults globales), que es el comportamiento correcto para una
    variedad nueva sin evidencia propia todavia.
    """
    overrides = VARIETY_OVERRIDES.get(variety, {})
    desconocidos = set(overrides) - _VALID_FIELDS
    if desconocidos:
        raise ValueError(
            f"VARIETY_OVERRIDES[{variety!r}] tiene campos invalidos: "
            f"{sorted(desconocidos)} (validos: {sorted(_VALID_FIELDS)})"
        )
    return VarietyConfig(variety=variety, **overrides)
