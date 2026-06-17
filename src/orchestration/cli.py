"""CLI parsing y resolucion de argumentos para el pipeline de training.

Vive aqui (separado de `main.py`) porque es la unica capa que:
  - Conoce el contrato CLI (argparse).
  - Traduce strings ("all") a listas tipadas.
  - Resuelve overrides de tuning profile (n_trials, folds, etc.).

`main.py` queda thin: parse + delega en `orchestration.runners`.

Nota: el pipeline SIEMPRE entrena todos los backends del registry
(`valid_backends()`) y delega la eleccion del campeon a
`champion.select_champion`. No hay flag de usuario para forzar un modelo:
ese es el contrato del proyecto (la maquina elige, no el operador).
"""
from __future__ import annotations

import argparse
import sys

from src.config import (
    DEFAULT_TUNING,
    DEFAULT_VARIETIES,
    MLFLOW_EXPERIMENT_PREFIX,
    TUNING_PROFILES,
)
from src.step_01_load.data_loader import list_varieties


def _positive_int(value: str) -> int:
    """argparse type para enteros >= 1 con mensaje accionable."""
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"valor invalido: {value!r}; se esperaba un entero >= 1",
        ) from exc
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            f"valor invalido: {ivalue}; debe ser >= 1",
        )
    return ivalue


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline ML - Nested CV + Optuna multi-variedad + multi-modelo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tuning",
        choices=list(TUNING_PROFILES),
        default=DEFAULT_TUNING,
        help="Presupuesto de Optuna: smoke (~1 min) | dev (~10 min) | prod (~1.5 h).",
    )
    parser.add_argument(
        "--varieties",
        default=DEFAULT_VARIETIES,
        help='Lista CSV de variedades (= hojas) o "all" para todas.',
    )
    parser.add_argument("--n-trials", type=_positive_int, default=None)
    parser.add_argument("--final-trials", type=_positive_int, default=None)
    parser.add_argument("--outer-folds", type=_positive_int, default=None)
    parser.add_argument("--inner-folds", type=_positive_int, default=None)
    parser.add_argument("--skip-final-tuning", action="store_true")
    parser.add_argument(
        "--experiment-prefix",
        default=MLFLOW_EXPERIMENT_PREFIX,
        help=(
            "Prefijo de experimentos. Cada variedad va a su propio experimento "
            "(prefix + variety). Ej: 'productivity_' -> 'productivity_POP'."
        ),
    )
    parser.add_argument(
        "--no-register",
        dest="register_model",
        action="store_false",
        default=True,
        help="Desactiva el registro en MLflow Model Registry (default: registra)",
    )
    parser.add_argument(
        "--registry-stage",
        choices=["None", "Staging", "Production"],
        default="None",
        help=(
            "Stage para la nueva version registrada (default: None = registra "
            "sin promover). Pasa 'Staging' o 'Production' explicitamente cuando "
            "quieras promover. Asi evitamos que entrenamientos dev ensucien el "
            "Registry con versiones en Staging."
        ),
    )
    parser.add_argument(
        "--parallel-varieties",
        type=_positive_int,
        default=1,
        help=(
            "Variedades a entrenar en PARALELO (procesos independientes). "
            "Cada subproceso libera su memoria al terminar (cache cleanup "
            "natural). Recomendado: 2-4 segun cores disponibles."
        ),
    )
    parser.add_argument(
        "--inner-cv-n-jobs",
        type=int,
        default=-1,
        help=(
            "n_jobs para cross_val_score interno. -1 = todos los cores. "
            "Si --parallel-varieties > 1, se ajusta automaticamente para "
            "evitar oversubscription."
        ),
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help=(
            "Modo EDA: en vez de entrenar, corre el analisis exploratorio "
            "(src.diagnostics.eda.run_eda) por variedad y sincroniza los HTML "
            "a reports/. Standalone y repetible; ignora flags de tuning/modelo."
        ),
    )
    return parser.parse_args(argv)


def resolve_settings(args: argparse.Namespace) -> dict[str, int | bool | str]:
    """Aplica overrides CLI sobre el tuning profile base."""
    base = dict(TUNING_PROFILES[args.tuning])
    overrides = {
        "n_trials": args.n_trials,
        "final_trials": args.final_trials,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
    }
    for k, v in overrides.items():
        if v is not None:
            base[k] = v
    base["skip_final_tuning"] = args.skip_final_tuning
    return base


def resolve_varieties(arg: str) -> list[str]:
    """Resuelve la lista de variedades validando contra el catalogo del Excel.

    Cualquier variedad desconocida aborta con `sys.exit(2)` y lista las
    validas, para que el error se vea en la CLI antes de gastar workers
    en jobs que de todos modos fallarian al cargar la hoja del Excel.
    """
    if arg.strip().lower() == "all":
        return list_varieties()
    requested = [v.strip() for v in arg.split(",") if v.strip()]
    try:
        available = list_varieties()
    except FileNotFoundError:
        # Sin Excel no podemos validar; dejamos pasar y que el worker
        # reporte el error de I/O con su contexto.
        return requested
    available_set = set(available)
    unknown = [v for v in requested if v not in available_set]
    if unknown:
        print(
            f"error: variedades desconocidas: {unknown}. "
            f"Validas: {available}",
            file=sys.stderr,
        )
        sys.exit(2)
    return requested
