"""Logger raíz del frontend."""

from __future__ import annotations

import logging
import sys

from app.core.constants import DEFAULT_LOG_LEVEL, LOGGER_NAME

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=DEFAULT_LOG_LEVEL,
    stream=sys.stdout,
)

logger = logging.getLogger(LOGGER_NAME)
