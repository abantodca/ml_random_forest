"""
Configuración de logging
========================
Define el formato y nivel de logs para toda la aplicación
"""

import logging
import sys

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def setup_logger(name: str = "rnd-forest-backend", level: str = "INFO") -> logging.Logger:
    """
    Configura el logger de la aplicación.

    Args:
        name: Nombre del logger
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Logger configurado

    Raises:
        ValueError: si `level` no es un nivel de logging válido. Sin esta guarda,
            un valor mal escrito reventaba con un AttributeError opaco en
            `getattr(logging, ...)`.
    """
    normalized_level = level.upper()
    if normalized_level not in _VALID_LEVELS:
        raise ValueError(
            f"level '{level}' no válido. Valores aceptados: {', '.join(sorted(_VALID_LEVELS))}"
        )

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, normalized_level))

    # Silenciar el ruidoso "Detected one or more mismatches" del sublogger
    # `mlflow.utils.requirements_utils`. Ese mensaje lista deps OPCIONALES
    # del entorno conda de training (bottleneck, defusedxml, distributed,
    # lz4, matplotlib, xarray, zstandard) que el backend NO necesita en el
    # path de inferencia. Las que SI nos importan (cloudpickle, scipy) ya
    # estan pineadas en requirements.txt al MLmodel y no aparecen.
    #
    # Se aplica fuera del check `if logger.handlers` porque:
    #   1. Es idempotente (`disabled=True` setado dos veces sigue True).
    #   2. MLflow re-configura sus loggers durante `import mlflow` y puede
    #      resetear `setLevel`, pero `disabled=True` sobrevive porque no
    #      es un atributo que `_configure_mlflow_loggers` toque.
    logging.getLogger("mlflow.utils.requirements_utils").disabled = True

    # Evitar duplicar handlers si ya existe
    if logger.handlers:
        return logger

    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Formato de los logs
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Silenciar logs verbosos de librerías externas
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    return logger
