"""Excepciones personalizadas del dominio."""


class VarietyNotFoundError(ValueError):
    """Se solicitó una variedad fuera del catálogo."""

    def __init__(self, variety: str):
        self.variety = variety
        super().__init__(f"Variedad '{variety}' no encontrada en el catálogo")


class ModelNotAvailableError(RuntimeError):
    """El modelo MLflow no está cargado o no se pudo cargar."""

    def __init__(self, variety: str):
        self.variety = variety
        super().__init__(
            f"Modelo para la variedad '{variety}' no está disponible. "
            f"Verifica que MLflow esté corriendo y el modelo exista."
        )


class PredictionError(RuntimeError):
    """La inferencia del modelo falló."""

    def __init__(self, variety: str, detail: str):
        self.variety = variety
        self.detail = detail
        super().__init__(f"Error en predicción para '{variety}': {detail}")


class ForecastNotFoundError(ValueError):
    """No existe un pronóstico con el ID indicado."""

    def __init__(self, forecast_id: int):
        self.forecast_id = forecast_id
        super().__init__(f"Pronóstico con ID {forecast_id} no encontrado")
