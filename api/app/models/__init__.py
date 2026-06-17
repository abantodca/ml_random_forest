"""
app.models - Fachada pública del paquete models
================================================
Re-exporta el motor de base de datos y los modelos ORM para que los
consumidores (CRUD, services, main) los importen desde `app.models`
sin acoplarse al archivo interno.
"""

from app.models.database import (
    Base,
    dispose_engine,
    get_engine,
    get_session,
    init_db,
)
from app.models.forecast import Forecast
from app.models.historical_observation import HistoricalObservation

__all__ = [
    # Database
    "Base",
    "dispose_engine",
    "get_engine",
    "get_session",
    "init_db",
    # ORM models
    "Forecast",
    "HistoricalObservation",
]
