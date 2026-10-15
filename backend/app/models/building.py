from sqlalchemy import Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Building(Base):
    __tablename__ = "buildings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    site_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    primary_use: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sqm: Mapped[float | None] = mapped_column(Float, nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")
    selected_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
