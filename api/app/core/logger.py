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

    logging.getLogger("mlflow.utils.requirements_utils").disabled = True

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    return logger
