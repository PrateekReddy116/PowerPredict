from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.user import utcnow


class SensorReading(Base):
    """Energy observations. Named sensor_readings (plan text sometimes said 'readings')."""

    __tablename__ = "sensor_readings"
    __table_args__ = (
        UniqueConstraint(
            "building_id",
            "ts",
            "data_origin",
            "meter_type",
            name="uq_sensor_readings_building_ts_origin_meter",
        ),
        Index("ix_sensor_readings_building_ts", "building_id", "ts"),
        CheckConstraint(
            "data_origin IN ('historical', 'simulated_iot', 'real_iot')",
            name="ck_sensor_readings_origin",
        ),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    building_id: Mapped[str] = mapped_column(String(64), ForeignKey("buildings.id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    energy_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    data_origin: Mapped[str] = mapped_column(String(32), nullable=False)
    meter_type: Mapped[str] = mapped_column(String(32), nullable=False, default="electricity")
    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    occupancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    occupancy_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    schema_version: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
