"""
app.routers - Fachada del paquete de routers
=============================================
Re-exporta los módulos de cada router. `main.py` los registra mediante
`app.include_router(<modulo>.router, ...)`, así que aquí solo
exponemos los módulos como atributos del paquete.
"""

from app.routers import forecasts, health, history, varieties

__all__ = ["forecasts", "health", "history", "varieties"]
