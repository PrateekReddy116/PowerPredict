"""Phase 5: Isolation Forest extras on anomalies (deviation, severity, actual vs expected).

Revision ID: 003_phase5
Revises: 002_phase3
Create Date: 2026-08-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "003_phase5"
down_revision: str | None = "002_phase3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("anomalies", sa.Column("deviation_pct", sa.Float(), nullable=True))
    op.add_column("anomalies", sa.Column("severity", sa.String(length=16), nullable=True))
    op.add_column("anomalies", sa.Column("actual_kwh", sa.Float(), nullable=True))
    op.add_column("anomalies", sa.Column("expected_kwh", sa.Float(), nullable=True))
    op.create_check_constraint(
        "ck_anomalies_severity",
        "anomalies",
        "severity IS NULL OR severity IN ('normal', 'warning', 'critical')",
    )
    op.create_unique_constraint(
        "uq_anomalies_building_ts_origin",
        "anomalies",
        ["building_id", "ts", "data_origin"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_anomalies_building_ts_origin", "anomalies", type_="unique")
    op.drop_constraint("ck_anomalies_severity", "anomalies", type_="check")
    op.drop_column("anomalies", "expected_kwh")
    op.drop_column("anomalies", "actual_kwh")
    op.drop_column("anomalies", "severity")
    op.drop_column("anomalies", "deviation_pct")
