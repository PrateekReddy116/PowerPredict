"""Empty schema stubs for later phases. Do not populate in Phase 1."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.user import utcnow


class WeatherObservation(Base):
    __tablename__ = "weather"
    __table_args__ = (Index("ix_weather_site_ts", "site_id", "ts"),)

    id: Mapped[str] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    site_id: Mapped[str] = mapped_column(String(64), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    air_temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    dew_temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    sea_level_pressure: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_origin: Mapped[str] = mapped_column(String(32), nullable=False, default="historical")


class OccupancyObservation(Base):
    __tablename__ = "occupancy"
    __table_args__ = (Index("ix_occupancy_building_ts", "building_id", "ts"),)

    id: Mapped[str] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    building_id: Mapped[str] = mapped_column(String(64), ForeignKey("buildings.id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    occupancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    occupancy_source: Mapped[str] = mapped_column(String(32), nullable=False, default="simulated")


class Explanation(Base):
    __tablename__ = "explanations"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    model_metrics_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("model_metrics.id"), nullable=True
    )
    forecast_id: Mapped[str | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("forecasts.id"), nullable=True)
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
