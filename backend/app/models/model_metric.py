from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.user import utcnow


class ModelMetric(Base):
    """One offline training run. Plan text sometimes said model_runs; table is model_metrics."""

    __tablename__ = "model_metrics"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    train_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    train_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    val_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    val_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    test_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    test_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    rmse: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2: Mapped[float | None] = mapped_column(Float, nullable=True)
    mape: Mapped[float | None] = mapped_column(Float, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    training_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
