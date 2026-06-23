"""Single source of truth para los backends de modelo entrenables.

Antes habia DOS dicts paralelos:
    `_MODEL_FACTORIES`     en tuning.py        (str -> factory)
    `SEARCH_SPACE_REGISTRY` en search_spaces.py  (str -> suggest_*)

Cualquier nuevo backend (ngboost, tabnet, ...) requeria editar dos
archivos coordinadamente y arriesgar que divergieran. Peor: cli.py
listaba `VALID_MODELS = sorted(SEARCH_SPACE_REGISTRY)` -- es decir,
considera "modelo valido" al que tenga search_space, sin verificar que
tenga factory. Una incoherencia silenciosa fallaba en runtime.

Este modulo centraliza ambos en un dataclass tipado. El registry vive
en el mismo `step_04_train/` que sus piezas, y `cli.py`/`tuning.py`/
`search_spaces.py` lo importan.

Para AGREGAR UN BACKEND NUEVO (ngboost, tabnet, ...) son 4 cambios
localizados, sin tocar el orchestrator ni el champion:
  1. Crear `model_<name>.py` con `get_<name>_model()` factory que
     devuelve un sklearn-compatible regressor (idealmente envuelto en
     `wrap_with_log_target`).
  2. Agregar `suggest_<name>_params(trial)` en `search_spaces.py`.
  3. Importar ambos aqui y agregar al `BACKEND_REGISTRY`.
  4. CLI los recoge via `valid_backends()` automaticamente.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.step_04_train.model_lgb import get_lgb_model
from src.step_04_train.model_xgb import get_xgb_model
from src.step_04_train.search_spaces import (
    suggest_lgb_params,
    suggest_xgb_params,
)


@dataclass(frozen=True)
class ModelBackend:
    """Un backend entrenable = factory + search space + nombre.

    Inmutable a proposito: el registry debe ser readonly. Si en el
    futuro se necesitan defaults distintos por backend, agregarlos como
    campo `defaults: dict` aqui (un solo lugar de cambio).
    """

    name: str
    factory: Callable[..., Any]
    search_space: Callable[..., dict[str, object]]


BACKEND_REGISTRY: dict[str, ModelBackend] = {
    "xgb": ModelBackend("xgb", get_xgb_model, suggest_xgb_params),
    "lgb": ModelBackend("lgb", get_lgb_model, suggest_lgb_params),
}


def get_backend(name: str) -> ModelBackend:
    """Devuelve el ModelBackend o levanta ValueError con la lista valida."""
    if name not in BACKEND_REGISTRY:
        raise ValueError(f"Backend '{name}' no soportado. Disponibles: {sorted(BACKEND_REGISTRY)}")
    return BACKEND_REGISTRY[name]


def valid_backends() -> list[str]:
    """Lista de nombres validos (para argparse choices y validacion)."""
    return sorted(BACKEND_REGISTRY)
