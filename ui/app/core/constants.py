"""Constantes globales del frontend (paleta, timeouts, columnas)."""

from __future__ import annotations

DEFAULT_API_URL: str = "http://localhost:8000"
DEFAULT_TIMEOUT_HEALTH: int = 5
DEFAULT_TIMEOUT_READ: int = 10
DEFAULT_TIMEOUT_WRITE: int = 15
DEFAULT_TIMEOUT_BATCH: int = 60
API_BATCH_MAX_ROWS: int = 1000

DEFAULT_CACHE_TTL_HEALTH: int = 60
DEFAULT_CACHE_TTL_VARIETIES: int = 120
DEFAULT_CACHE_TTL_FORECASTS: int = 30
CACHE_TTL_DASHBOARD_HTML: int = 900

DEFAULT_LOG_LEVEL: str = "INFO"
LOGGER_NAME: str = "rnd-forest-frontend"

WORKERS_VARIETY_ROOT: int = 2
WORKERS_VARIETY_DETAIL_MAX: int = 10

LONGITUD_VISIBLE_API_URL: int = 22

TEMA: dict[str, str] = {
    "primary": "#4F46E5",
    "primary_dark": "#3730A3",
    "primary_light": "#818CF8",
    "accent": "#7C3AED",
    "success": "#047857",
    "warning": "#B45309",
    "danger": "#B91C1C",
    "info": "#0E7490",
    "bg": "#F8FAFC",
    "bg_alt": "#F1F5F9",
    "card": "#FFFFFF",
    "border": "#E2E8F0",
    "border_strong": "#CBD5E1",
    "text": "#0F172A",
    "text_body": "#1E293B",
    "text_secondary": "#334155",
    "text_tertiary": "#475569",
    "muted": "#64748B",
    "purple": "#7C3AED",
    "blue": "#3B82F6",
}

PALETA_SERIES: tuple[str, ...] = (
    "#4F46E5",
    "#7C3AED",
    "#0E7490",
    "#047857",
    "#B45309",
    "#BE185D",
    "#1D4ED8",
    "#7E22CE",
)

COLUMNAS_REQUERIDAS: tuple[str, ...] = (
    "VARIEDAD",
    "FECHA",
    "KG/HA",
    "DPC",
    "HA",
    "DIA_COSECHA",
    "FORMATO",
    "FUNDO",
)
COLUMNAS_OPCIONALES: tuple[str, ...] = (
    "%INDUS",
    "P/BAYA",
    "HORAS_EFECTIVAS",
    "EXTERNAL_ID",
)

FORMATOS_FALLBACK: tuple[str, ...] = ()
FORMATO_DEFAULT_FALLBACK: str = ""
FUNDOS_FALLBACK: tuple[str, ...] = ()
