"""
app.crud - Fachada del paquete CRUD
====================================
Re-exporta los módulos de operaciones (`forecast`, `historical_observation`)
para que los routers usen el patrón `crud.forecast.fn(...)` sin importar
cada archivo manualmente.
"""

from app.crud import forecast, historical_observation

__all__ = ["forecast", "historical_observation"]
