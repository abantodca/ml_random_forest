"""presenters - Arman los view-models que consumen las páginas.

Cada presenter toma datos crudos de `services`/`dependencies`, los agrega y
los devuelve como view-models listos para render, de modo que las vistas
(`app/views`) queden delgadas (solo composición de componentes), igual que
los routers de `api/` delegan en los services.
"""
