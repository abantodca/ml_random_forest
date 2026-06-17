"""
Middlewares personalizados de la aplicación
============================================
Se registran en main.py usando app.add_middleware().
"""

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Registra cada request con método, path, status code y duración.

    Los paths en EXCLUDED_PATHS no generan log para no saturar
    los registros con health checks frecuentes.
    """

    EXCLUDED_PATHS: frozenset[str] = frozenset({"/api/health"})

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if request.url.path not in self.EXCLUDED_PATHS:
            logger.info(
                "%s %s → %d (%.0fms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )

        return response
