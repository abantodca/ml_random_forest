"""app - Paquete raíz de la UI (Streamlit).

Espeja la arquitectura en capas de `api/app`:

    core/         configuración, constantes, excepciones, logging
    client/       adaptador HTTP hacia el backend FastAPI
    schemas/      DTOs de presentación (view-models de datos)
    services/     lógica de negocio del frontend
    presenters/   armado de view-models para vistas finas
    components/   widgets de render (layout, charts, forms)
    pages/        páginas Streamlit (composición delgada)
    dependencies  composition root + caché (≈ api/app/dependencies.py)
    app           punto de entrada / navegación
"""
