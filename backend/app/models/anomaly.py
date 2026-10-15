from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
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


class Anomaly(Base):
    __tablename__ = "anomalies"
    __table_args__ = (
        Index("ix_anomalies_building_ts", "building_id", "ts"),
        UniqueConstraint("building_id", "ts", "data_origin", name="uq_anomalies_building_ts_origin"),
        CheckConstraint(
            "data_origin IN ('historical', 'simulated_iot', 'real_iot')",
            name="ck_anomalies_origin",
        ),
        CheckConstraint(
            "severity IS NULL OR severity IN ('normal', 'warning', 'critical')",
            name="ck_anomalies_severity",
        ),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    building_id: Mapped[str] = mapped_column(String(64), ForeignKey("buildings.id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    model_metrics_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("model_metrics.id"), nullable=True
    )
    data_origin: Mapped[str] = mapped_column(String(32), nullable=False)
    deviation_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity: Mapped[str | None] = mapped_column(String(16), nullable=True)
    actual_kwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_kwh: Mapped[float | None] = mapped_column(Float, nullable=True)
