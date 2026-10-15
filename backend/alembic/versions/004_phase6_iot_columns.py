"""Phase 6: optional IoT columns on sensor_readings and weather.

Revision ID: 004_phase6
Revises: 003_phase5
Create Date: 2026-08-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "004_phase6"
down_revision: str | None = "003_phase5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("sensor_readings", sa.Column("temperature_c", sa.Float(), nullable=True))
    op.add_column("sensor_readings", sa.Column("humidity_pct", sa.Float(), nullable=True))
    op.add_column("sensor_readings", sa.Column("occupancy", sa.Float(), nullable=True))
    op.add_column("sensor_readings", sa.Column("occupancy_source", sa.String(length=32), nullable=True))
    op.add_column("sensor_readings", sa.Column("schema_version", sa.String(length=16), nullable=True))
    op.add_column("weather", sa.Column("humidity_pct", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("weather", "humidity_pct")
    op.drop_column("sensor_readings", "schema_version")
    op.drop_column("sensor_readings", "occupancy_source")
    op.drop_column("sensor_readings", "occupancy")
    op.drop_column("sensor_readings", "humidity_pct")
    op.drop_column("sensor_readings", "temperature_c")
