"""Phase 1 schema: users, buildings, sensor_readings, model_metrics, forecasts, anomalies, stubs.

Revision ID: 001_phase1
Revises:
Create Date: 2026-08-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "001_phase1"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("role IN ('viewer', 'admin')", name="ck_users_role"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "buildings",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("site_id", sa.String(length=64), nullable=False),
        sa.Column("primary_use", sa.String(length=64), nullable=True),
        sa.Column("sqm", sa.Float(), nullable=True),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("selected_reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_buildings_site_id"), "buildings", ["site_id"], unique=False)

    op.create_table(
        "model_metrics",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("train_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("train_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("val_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("val_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("test_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("test_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("r2", sa.Float(), nullable=True),
        sa.Column("mape", sa.Float(), nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=True),
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.Column("random_seed", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "sensor_readings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("building_id", sa.String(length=64), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("energy_kwh", sa.Float(), nullable=False),
        sa.Column("data_origin", sa.String(length=32), nullable=False),
        sa.Column("meter_type", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "data_origin IN ('historical', 'simulated_iot', 'real_iot')",
            name="ck_sensor_readings_origin",
        ),
        sa.ForeignKeyConstraint(["building_id"], ["buildings.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "building_id",
            "ts",
            "data_origin",
            "meter_type",
            name="uq_sensor_readings_building_ts_origin_meter",
        ),
    )
    op.create_index("ix_sensor_readings_building_ts", "sensor_readings", ["building_id", "ts"])

    op.create_table(
        "forecasts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("building_id", sa.String(length=64), nullable=False),
        sa.Column("model_metrics_id", sa.Uuid(), nullable=True),
        sa.Column("target_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("y_true", sa.Float(), nullable=True),
        sa.Column("y_pred", sa.Float(), nullable=False),
        sa.Column("split", sa.String(length=8), nullable=False),
        sa.CheckConstraint("split IN ('val', 'test', 'live')", name="ck_forecasts_split"),
        sa.ForeignKeyConstraint(["building_id"], ["buildings.id"]),
        sa.ForeignKeyConstraint(["model_metrics_id"], ["model_metrics.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_forecasts_building_ts", "forecasts", ["building_id", "target_ts"])

    op.create_table(
        "anomalies",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("building_id", sa.String(length=64), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("is_anomaly", sa.Boolean(), nullable=False),
        sa.Column("model_metrics_id", sa.Uuid(), nullable=True),
        sa.Column("data_origin", sa.String(length=32), nullable=False),
        sa.CheckConstraint(
            "data_origin IN ('historical', 'simulated_iot', 'real_iot')",
            name="ck_anomalies_origin",
        ),
        sa.ForeignKeyConstraint(["building_id"], ["buildings.id"]),
        sa.ForeignKeyConstraint(["model_metrics_id"], ["model_metrics.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_anomalies_building_ts", "anomalies", ["building_id", "ts"])

    op.create_table(
        "weather",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("site_id", sa.String(length=64), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("air_temperature_c", sa.Float(), nullable=True),
        sa.Column("dew_temperature_c", sa.Float(), nullable=True),
        sa.Column("sea_level_pressure", sa.Float(), nullable=True),
        sa.Column("wind_speed", sa.Float(), nullable=True),
        sa.Column("data_origin", sa.String(length=32), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_weather_site_ts", "weather", ["site_id", "ts"])

    op.create_table(
        "occupancy",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("building_id", sa.String(length=64), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("occupancy", sa.Float(), nullable=True),
        sa.Column("occupancy_source", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["building_id"], ["buildings.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_occupancy_building_ts", "occupancy", ["building_id", "ts"])

    op.create_table(
        "explanations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("model_metrics_id", sa.Uuid(), nullable=True),
        sa.Column("forecast_id", sa.Uuid(), nullable=True),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["forecast_id"], ["forecasts.id"]),
        sa.ForeignKeyConstraint(["model_metrics_id"], ["model_metrics.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("explanations")
    op.drop_table("occupancy")
    op.drop_table("weather")
    op.drop_table("anomalies")
    op.drop_table("forecasts")
    op.drop_table("sensor_readings")
    op.drop_table("model_metrics")
    op.drop_table("buildings")
    op.drop_table("users")
