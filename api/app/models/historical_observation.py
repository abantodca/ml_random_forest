"""ORM HistoricalObservation
=============================
Almacena observaciones historicas (KG/HA, KG/JR_H) por (variety, fundo,
formato, fecha) que el `FeaturePipeline` usa para calcular los 31 lag
features que el modelo MLflow espera en su signature.

Se siembra una vez desde el Excel BD_HISTORICO_ACUMULADO mediante el
endpoint `POST /history/{variety}/upload` y se mantiene actualizada
cada vez que el negocio capture nuevas observaciones reales.
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class HistoricalObservation(Base):
    """Observacion historica con target real (KG/JR_H)."""

    __tablename__ = "historical_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    variety: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    fundo: Mapped[str] = mapped_column(String(80), nullable=False)
    formato: Mapped[str] = mapped_column(String(40), nullable=False)
    fecha: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    kg_ha: Mapped[float] = mapped_column(Float, nullable=False)
    kg_jr_h: Mapped[float] = mapped_column(Float, nullable=False)

    # Features REALES del input al cosechar (opcionales). Espejan los inputs
    # del pronóstico y permiten la descomposición exacta de error: re-predecir
    # sobre estos valores reales aísla el error del modelo del error de datos.
    # NULL cuando el Excel de reales trae solo el formato mínimo (KG/HA+KG/JR_H).
    dpc: Mapped[float | None] = mapped_column(Float, nullable=True)
    indus_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_baya: Mapped[float | None] = mapped_column(Float, nullable=True)
    ha: Mapped[float | None] = mapped_column(Float, nullable=True)
    dia_cosecha: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_history_variety_group_fecha",
            "variety", "fundo", "formato", "fecha",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<HistoricalObservation(id={self.id}, variety={self.variety}, "
            f"fundo={self.fundo}, formato={self.formato}, fecha={self.fecha})>"
        )
