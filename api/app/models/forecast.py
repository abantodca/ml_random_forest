"""
Modelos ORM de SQLAlchemy
==========================
Define las tablas de la base de datos usando SQLAlchemy ORM.

Las columnas reflejan los inputs raw del modelo MLflow (KG/HA, %INDUS,
DPC, P/BAYA, HA, DIA_COSECHA, FORMATO, FUNDO, FECHA) mas:
  - HORAS_EFECTIVAS y EXTERNAL_ID como metadatos del request,
  - kghora_pred / kgjn_pred como salidas persistidas.
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class Forecast(Base):
    """Pronostico de productividad (kg cosechados por jornal-hora)."""

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Identificadores
    variety: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    fecha: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    external_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True
    )

    # Datos de entrada (features RAW que el pipeline MLflow espera)
    kg_ha: Mapped[float] = mapped_column(Float, nullable=False)
    indus_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    dpc: Mapped[float] = mapped_column(Float, nullable=False)
    p_baya: Mapped[float | None] = mapped_column(Float, nullable=True)
    ha: Mapped[float] = mapped_column(Float, nullable=False)
    dia_cosecha: Mapped[int] = mapped_column(Integer, nullable=False)
    formato: Mapped[str] = mapped_column(String(40), nullable=False, default="FRESCO")
    fundo: Mapped[str] = mapped_column(String(80), nullable=False)

    # Metadato del request (no entra al modelo)
    horas_efectivas: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Predicciones
    kghora_pred: Mapped[float] = mapped_column(Float, nullable=False)
    kgjn_pred: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Metadatos
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (Index("ix_forecasts_variety_fecha", "variety", "fecha"),)

    def __repr__(self) -> str:
        return (
            f"<Forecast(id={self.id}, variety={self.variety}, "
            f"fecha={self.fecha}, kghora_pred={self.kghora_pred})>"
        )
