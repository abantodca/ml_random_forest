"""Cliente HTTP minimalista para el backend FastAPI.

Solo se ocupa del transporte: construye URLs, ejecuta requests y normaliza
errores. La traducción JSON ↔ entidades vive en `app.client.mappers`.
"""

from __future__ import annotations

import requests

from app.core import ApiConnectionError, ApiResponseError, Configuracion


class ApiClient:
    """Wrapper sobre `requests` con manejo unificado de errores."""

    def __init__(self, cfg: Configuracion) -> None:
        self._base_url = cfg.api_url.rstrip("/")
        self.timeout_health = cfg.timeout_health
        self.timeout_read = cfg.timeout_read
        self.timeout_write = cfg.timeout_write
        self.timeout_batch = cfg.timeout_batch

    @property
    def base_url(self) -> str:
        return self._base_url

    # ---- Métodos de conveniencia ----------------------------------------

    def get(self, path: str, *, timeout: int, params: dict | None = None) -> dict:
        return self.request("GET", path, timeout=timeout, params=params)

    def post(
        self,
        path: str,
        *,
        timeout: int,
        json: dict | None = None,
        files: dict | None = None,
    ) -> dict:
        return self.request("POST", path, timeout=timeout, json=json, files=files)

    def patch(self, path: str, *, timeout: int, json: dict | None = None) -> dict:
        return self.request("PATCH", path, timeout=timeout, json=json)

    def delete(self, path: str, *, timeout: int) -> dict:
        return self.request("DELETE", path, timeout=timeout)

    def request(self, method: str, path: str, *, timeout: int, **kwargs) -> dict:
        url = f"{self._base_url}{path}"
        try:
            r = requests.request(method, url, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as exc:
            detail = self._extract_error_detail(exc)
            status = exc.response.status_code if exc.response is not None else 0
            raise ApiResponseError(detail, status) from exc
        except requests.RequestException as exc:
            raise ApiConnectionError(str(exc)) from exc

    def get_text(self, path: str, *, timeout: int) -> tuple[int, str]:
        """Variante para descargar texto/HTML; devuelve (status_code, body)."""
        url = f"{self._base_url}{path}"
        try:
            r = requests.get(url, timeout=timeout)
        except requests.RequestException as exc:
            raise ApiConnectionError(str(exc)) from exc
        return r.status_code, r.text

    @staticmethod
    def _extract_error_detail(exc: requests.exceptions.HTTPError) -> str:
        detail = str(exc)
        if exc.response is None:
            return detail
        try:
            body = exc.response.json()
            return body.get("message") or body.get("detail") or detail
        except Exception:
            return detail
