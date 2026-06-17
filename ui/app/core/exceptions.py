"""Excepciones de aplicación para errores del backend."""

from __future__ import annotations


class ApiConnectionError(Exception):
    """No se puede conectar al backend API."""


class ApiResponseError(Exception):
    """El backend devolvió un error HTTP."""

    def __init__(self, detail: str, status_code: int = 0) -> None:
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)
