"""Casos de uso para variedades (listado, detalle, dashboard)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.client import endpoints
from app.client.api_client import ApiClient
from app.client.mappers import to_variety
from app.core import (
    WORKERS_VARIETY_DETAIL_MAX,
    WORKERS_VARIETY_ROOT,
    ApiConnectionError,
    ApiResponseError,
    logger,
)
from app.schemas import Catalogs, VarietyViewModel


class VarietyService:
    def __init__(self, client: ApiClient) -> None:
        self._client = client

    def list_all(self) -> list[VarietyViewModel]:
        try:
            all_names, available_names = self._fetch_root_lists()
        except (ApiConnectionError, ApiResponseError) as exc:
            logger.error("Error al obtener variedades: %s", exc)
            return []

        # Solo las variedades con modelo entrenado (presentes en el registry,
        # vía /available) deben aparecer en la UI. Las del catálogo sin modelo
        # se descartan: no se pueden pronosticar ni tienen reporte/dashboard.
        names_with_model = [n for n in all_names if n in available_names]
        detail_map = self._fetch_details(names_with_model)
        return [detail_map[name] for name in names_with_model if name in detail_map]

    def loaded_names(self) -> list[str]:
        return [v.name for v in self.list_all() if v.model_loaded]

    def get_catalogs(self) -> Catalogs:
        """Trae catálogos cerrados (FORMATO, FUNDO) del backend."""
        data = self._client.get(
            endpoints.CATALOGS, timeout=self._client.timeout_read
        )
        return Catalogs(
            formatos=tuple(data.get("formatos", ())),
            formato_default=str(data.get("formato_default", "")),
            fundos=tuple(data.get("fundos", ())),
        )

    def get_dashboard_html(self, variety: str) -> str:
        status, text = self._client.get_text(
            endpoints.variety_dashboard(variety),
            timeout=self._client.timeout_batch,
        )
        if status == 404:
            return ""
        if status != 200:
            raise ApiResponseError(self._extract_text_detail(text), status)
        return text

    # ---- helpers privados -----------------------------------------------

    def _fetch_root_lists(self) -> tuple[list[str], set[str]]:
        with ThreadPoolExecutor(max_workers=WORKERS_VARIETY_ROOT) as executor:
            f_all = executor.submit(
                self._client.get,
                endpoints.VARIETIES,
                timeout=self._client.timeout_read,
            )
            f_avail = executor.submit(
                self._client.get,
                endpoints.VARIETIES_AVAILABLE,
                timeout=self._client.timeout_read,
            )
            d_all = f_all.result()
            d_avail = f_avail.result()
        return list(d_all.get("varieties", [])), set(d_avail.get("varieties", []))

    def _fetch_details(self, names: list[str]) -> dict[str, VarietyViewModel]:
        if not names:
            return {}
        max_workers = min(WORKERS_VARIETY_DETAIL_MAX, len(names))
        detail_map: dict[str, VarietyViewModel] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_one_detail, n): n for n in names
            }
            for future in as_completed(futures):
                vm = future.result()
                detail_map[vm.name] = vm
        return detail_map

    def _fetch_one_detail(self, name: str) -> VarietyViewModel:
        try:
            data = self._client.get(
                endpoints.variety_detail(name),
                timeout=self._client.timeout_read,
            )
        except (ApiConnectionError, ApiResponseError):
            return VarietyViewModel(name=name, model_loaded=True, metrics={})
        vm = to_variety(data, fallback_name=name)
        # `name` viene de los disponibles en el REGISTRY → TIENE modelo. El
        # `model_loaded` del backend es estado in-memory (lazy-load): tras un
        # reinicio queda False aunque el modelo exista, lo que vaciaba el
        # dashboard y hacía que Seguimiento abriera en una variedad sin datos.
        # Para el dashboard, "tiene modelo" = está en el registry → forzamos True.
        return vm if vm.model_loaded else vm.model_copy(update={"model_loaded": True})

    @staticmethod
    def _extract_text_detail(text: str) -> str:
        try:
            body = json.loads(text)
            return body.get("message") or body.get("detail") or text
        except (ValueError, TypeError):
            return text
