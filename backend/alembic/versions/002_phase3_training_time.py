"""Phase 3: optional training_time_sec on model_metrics.

Revision ID: 002_phase3
Revises: 001_phase1
Create Date: 2026-08-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "002_phase3"
down_revision: str | None = "001_phase1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("model_metrics", sa.Column("training_time_sec", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("model_metrics", "training_time_sec")
