"""Validación de uploads batch (Excel/CSV) usando pandera.

`pandera.DataFrameSchema.validate(lazy=True)` colecta TODAS las
violaciones por celda (fila, columna, valor, regla violada) en lugar
de fallar al primer error. Se traducen al shape `ValidationIssue` que
consume el UI (tabla de errores con botón de descarga).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors

from app.core import COLUMNAS_OPCIONALES, COLUMNAS_REQUERIDAS


@dataclass(frozen=True)
class ValidationIssue:
    fila: int  # 1-indexed (encabezado = 1)
    columna: str
    valor: str
    motivo: str


class BatchValidationError(Exception):
    """Lanzada cuando el DataFrame de upload tiene issues bloqueantes."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__(f"{len(issues)} errores de validación")


# ---------------------------------------------------------------------------
# Normalización (uppercase, strip, NA)
# ---------------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in ("VARIEDAD", "FUNDO"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    if "FORMATO" in df.columns:
        df["FORMATO"] = (
            df["FORMATO"].astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
        )
    for col in COLUMNAS_OPCIONALES:
        if col in df.columns:
            df[col] = df[col].replace({"": pd.NA, " ": pd.NA, "nan": pd.NA})
    return df


# ---------------------------------------------------------------------------
# Schema (declarativo)
# ---------------------------------------------------------------------------
def _is_valid_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").notna()


def _build_schema(
    formatos: tuple[str, ...],
    fundos: tuple[str, ...],
    valid_varieties: list[str] | None,
) -> pa.DataFrameSchema:
    variety_check = (
        [
            pa.Check.isin(
                [v.upper() for v in valid_varieties],
                error="Variedad no reconocida en el catálogo",
            )
        ]
        if valid_varieties is not None
        else []
    )
    columns: dict[str, pa.Column] = {
        "FECHA": pa.Column(
            object,
            pa.Check(_is_valid_date, error="Formato de fecha inválido (use YYYY-MM-DD)"),
            nullable=False,
        ),
        "VARIEDAD": pa.Column(str, variety_check, nullable=False),
        "KG/HA": pa.Column(
            float,
            pa.Check.in_range(0.001, 100_000, error="Debe estar entre 0 (excl.) y 100000"),
            nullable=False,
            coerce=True,
        ),
        "DPC": pa.Column(
            float,
            pa.Check.in_range(0, 400, error="Debe estar entre 0 y 400"),
            nullable=False,
            coerce=True,
        ),
        "HA": pa.Column(
            float,
            pa.Check.in_range(0.001, 10_000, error="Debe estar entre 0 (excl.) y 10000"),
            nullable=False,
            coerce=True,
        ),
        "DIA_COSECHA": pa.Column(
            int,
            pa.Check.in_range(0, 365, error="Debe estar entre 0 y 365"),
            nullable=False,
            coerce=True,
        ),
        "FORMATO": pa.Column(
            str,
            pa.Check.isin(formatos, error=f"Valor no permitido. Aceptados: {', '.join(formatos)}"),
            nullable=False,
        ),
        "FUNDO": pa.Column(
            str,
            pa.Check.isin(fundos, error=f"Valor no permitido. Aceptados: {', '.join(fundos)}"),
            nullable=False,
        ),
        "%INDUS": pa.Column(
            float,
            pa.Check.in_range(0, 100, error="Debe estar entre 0 y 100"),
            nullable=True,
            coerce=True,
            required=False,
        ),
        "P/BAYA": pa.Column(
            float,
            pa.Check.in_range(0.001, 100, error="Debe estar entre 0 (excl.) y 100"),
            nullable=True,
            coerce=True,
            required=False,
        ),
        "HORAS_EFECTIVAS": pa.Column(
            float,
            pa.Check.in_range(0, 24, error="Debe estar entre 0 y 24"),
            nullable=True,
            coerce=True,
            required=False,
        ),
    }
    return pa.DataFrameSchema(columns, strict="filter")


# ---------------------------------------------------------------------------
# Traducción failure_cases → ValidationIssue
# ---------------------------------------------------------------------------
def _failure_to_issue(row: pd.Series) -> ValidationIssue:
    raw_idx = row.get("index")
    # +2: pandas 0-indexed + header
    fila = int(raw_idx) + 2 if pd.notna(raw_idx) else 0
    valor_raw = row.get("failure_case")
    valor = (
        "(vacío)"
        if valor_raw is None or (isinstance(valor_raw, float) and pd.isna(valor_raw))
        else str(valor_raw)
    )
    return ValidationIssue(
        fila=fila,
        columna=str(row.get("column") or "—"),
        valor=valor,
        motivo=str(row.get("check") or "Regla incumplida"),
    )


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def validate_batch_upload(
    df: pd.DataFrame, *, valid_varieties: list[str] | None = None
) -> pd.DataFrame:
    """Normaliza y valida el DataFrame; lanza BatchValidationError con detalles.

    Si `valid_varieties` se provee, valida también la columna VARIEDAD.
    """
    df = _normalize_columns(df)

    missing = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if missing:
        raise BatchValidationError(
            [
                ValidationIssue(
                    fila=0,
                    columna=c,
                    valor="(ausente)",
                    motivo="Columna obligatoria faltante",
                )
                for c in missing
            ]
        )

    # Import local: rompe el ciclo app.services ⇄ app.dependencies
    # (dependencies importa los servicios; este es el único servicio que
    # necesita el composition root, y solo en runtime, no en import).
    from app.dependencies import get_cached_catalogs

    catalogs = get_cached_catalogs()
    schema = _build_schema(
        formatos=catalogs.formatos,
        fundos=catalogs.fundos,
        valid_varieties=valid_varieties,
    )
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        issues = [_failure_to_issue(row) for _, row in exc.failure_cases.iterrows()]
        raise BatchValidationError(issues) from exc
