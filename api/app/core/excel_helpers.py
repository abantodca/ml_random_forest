"""Helpers compartidos para parseo de archivos Excel."""

import io
from datetime import date

import pandas as pd

from app.core.config import settings


def read_excel_dataframe(contents: bytes, required_columns: frozenset[str]) -> pd.DataFrame:
    """Lee el Excel a DataFrame, normaliza nombres de columna y valida
    que estén las columnas requeridas.

    Centraliza el boilerplate que compartían el parser de pronósticos
    (`excel_service`) y el de historial (`history` router). NO valida
    extensión ni tamaño: eso es responsabilidad de `validate_excel_file`,
    que el caller debe invocar antes.

    Raises:
        ValueError: si falta alguna columna requerida.
    """
    df = pd.read_excel(io.BytesIO(contents))
    df.columns = [str(c).strip() for c in df.columns]
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(sorted(missing))}")
    return df


def validate_excel_file(contents: bytes, filename: str | None) -> None:
    """Valida extensión y tamaño máximo de un Excel cargado por endpoint."""
    if not filename or not filename.lower().endswith((".xlsx", ".xls")):
        raise ValueError("El archivo debe ser Excel (.xlsx o .xls)")
    max_bytes = settings.max_excel_file_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise ValueError(f"El archivo excede el limite de {settings.max_excel_file_size_mb} MB")


def validate_upload_size(size: int | None, filename: str | None) -> None:
    """Valida el tamaño declarado del upload ANTES de leerlo a memoria.

    `UploadFile.size` lo provee Starlette a partir del Content-Length del part
    multipart; cuando viene, rechazamos un archivo demasiado grande sin
    bufferizarlo entero (defensa contra OOM). Si `size` es `None` (cliente que
    no lo declara), no podemos decidir aquí y delegamos en `validate_excel_file`,
    que vuelve a chequear sobre los bytes ya leídos — comportamiento previo
    intacto. Valida también la extensión para fallar barato (sin leer) cuando
    el nombre ya delata un archivo no-Excel.
    """
    if not filename or not filename.lower().endswith((".xlsx", ".xls")):
        raise ValueError("El archivo debe ser Excel (.xlsx o .xls)")
    if size is None:
        return
    max_bytes = settings.max_excel_file_size_mb * 1024 * 1024
    if size > max_bytes:
        raise ValueError(f"El archivo excede el limite de {settings.max_excel_file_size_mb} MB")


def parse_date_value(value) -> date:
    """Acepta str ISO, datetime/Timestamp o date y devuelve `date`."""
    if isinstance(value, str):
        return date.fromisoformat(value)
    if hasattr(value, "date"):
        return value.date()
    return value
