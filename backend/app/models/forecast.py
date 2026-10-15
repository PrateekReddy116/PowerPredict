from datetime import datetime
from uuid import uuid4

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Index, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Forecast(Base):
    __tablename__ = "forecasts"
    __table_args__ = (
        Index("ix_forecasts_building_ts", "building_id", "target_ts"),
        CheckConstraint("split IN ('val', 'test', 'live')", name="ck_forecasts_split"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    building_id: Mapped[str] = mapped_column(String(64), ForeignKey("buildings.id"), nullable=False)
    model_metrics_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("model_metrics.id"), nullable=True
    )
    target_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    y_true: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_pred: Mapped[float] = mapped_column(Float, nullable=False)
    split: Mapped[str] = mapped_column(String(8), nullable=False)
