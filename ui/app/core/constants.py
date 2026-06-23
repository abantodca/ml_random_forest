"""Constantes globales del frontend (paleta, timeouts, columnas)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Backend / red
# ---------------------------------------------------------------------------
DEFAULT_API_URL: str = "http://localhost:8000"
DEFAULT_TIMEOUT_HEALTH: int = 5
DEFAULT_TIMEOUT_READ: int = 10
DEFAULT_TIMEOUT_WRITE: int = 15
# Cubre operaciones pesadas: subidas masivas y el 1er fetch (en frío) del
# reporte del modelo (~4MB descargados de S3 vía MLflow). La API cachea el
# HTML por variedad, así que solo el primer request paga el costo de red.
DEFAULT_TIMEOUT_BATCH: int = 60

# ---------------------------------------------------------------------------
# Cache TTL (segundos)
# ---------------------------------------------------------------------------
DEFAULT_CACHE_TTL_HEALTH: int = 60
DEFAULT_CACHE_TTL_VARIETIES: int = 120
DEFAULT_CACHE_TTL_FORECASTS: int = 30
CACHE_TTL_DASHBOARD_HTML: int = 900

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
DEFAULT_LOG_LEVEL: str = "INFO"
LOGGER_NAME: str = "rnd-forest-frontend"

# ---------------------------------------------------------------------------
# Concurrencia HTTP
# ---------------------------------------------------------------------------
WORKERS_VARIETY_ROOT: int = 2
WORKERS_VARIETY_DETAIL_MAX: int = 10

# ---------------------------------------------------------------------------
# Sidebar / UI
# ---------------------------------------------------------------------------
LONGITUD_VISIBLE_API_URL: int = 22

# ---------------------------------------------------------------------------
# Paleta de colores (WCAG AA contrast ratio ≥ 4.5:1 sobre blanco/bg)
# ---------------------------------------------------------------------------
TEMA: dict[str, str] = {
    # Marca
    "primary": "#4F46E5",  # indigo-600
    "primary_dark": "#3730A3",  # indigo-800 (para hover/text énfasis)
    "primary_light": "#818CF8",  # indigo-400
    "accent": "#7C3AED",  # violet-600
    # Semánticos
    "success": "#047857",  # emerald-700 (más oscuro, mejor contraste)
    "warning": "#B45309",  # amber-700
    "danger": "#B91C1C",  # red-700
    "info": "#0E7490",  # cyan-700
    # Superficies
    "bg": "#F8FAFC",  # slate-50
    "bg_alt": "#F1F5F9",  # slate-100
    "card": "#FFFFFF",
    "border": "#E2E8F0",  # slate-200
    "border_strong": "#CBD5E1",  # slate-300
    # Texto (sobre fondo claro)
    "text": "#0F172A",  # slate-900 — títulos
    "text_body": "#1E293B",  # slate-800 — cuerpo
    "text_secondary": "#334155",  # slate-700 — labels
    "text_tertiary": "#475569",  # slate-600 — meta
    "muted": "#64748B",  # slate-500 — placeholders/grids (antes #94A3B8 → ilegible)
    # Compat (alias antiguos)
    "purple": "#7C3AED",
    "blue": "#3B82F6",
}

# Paleta de series para gráficos (8 categorías)
PALETA_SERIES: tuple[str, ...] = (
    "#4F46E5",  # indigo
    "#7C3AED",  # violet
    "#0E7490",  # cyan
    "#047857",  # emerald
    "#B45309",  # amber
    "#BE185D",  # pink
    "#1D4ED8",  # blue
    "#7E22CE",  # purple
)

# ---------------------------------------------------------------------------
# Validación batch (Excel/CSV)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Catálogos cerrados — se cargan desde el endpoint /catalogs del backend.
# Las constantes vacías son el "fallback degradado": si el backend no
# responde, los selectboxes quedan vacíos y la validación falla con un
# mensaje claro en lugar de aceptar valores stale.
# ---------------------------------------------------------------------------
FORMATOS_FALLBACK: tuple[str, ...] = ()
FORMATO_DEFAULT_FALLBACK: str = ""
FUNDOS_FALLBACK: tuple[str, ...] = ()
