from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.enums import DataOrigin


class BuildingOut(BaseModel):
    id: str
    site_id: str
    primary_use: str | None
    sqm: float | None
    timezone: str
    selected_reason: str | None

    model_config = {"from_attributes": True}


class ReadingOut(BaseModel):
    id: UUID
    building_id: str
    ts: datetime
    energy_kwh: float
    data_origin: DataOrigin
    meter_type: str
    temperature_c: float | None = None
    humidity_pct: float | None = None
    occupancy: float | None = None
    occupancy_source: str | None = None
    schema_version: str | None = None

    model_config = {"from_attributes": True}


class ModelRunOut(BaseModel):
    """Serialized model_metrics row. Metrics are null until a real training run."""

    id: UUID
    model_name: str
    created_at: datetime
    train_start: datetime | None
    train_end: datetime | None
    val_start: datetime | None
    val_end: datetime | None
    test_start: datetime | None
    test_end: datetime | None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    mape: float | None = None
    artifact_path: str | None = None
    random_seed: int | None = None
    training_time_sec: float | None = None

    model_config = {"from_attributes": True}


class ForecastOut(BaseModel):
    id: UUID
    building_id: str
    model_run_id: UUID | None = Field(validation_alias="model_metrics_id")
    target_ts: datetime
    y_true: float | None
    y_pred: float
    split: str
    model_name: str | None = None

    model_config = {"from_attributes": True, "populate_by_name": True}


class AnomalyOut(BaseModel):
    id: UUID
    building_id: str
    ts: datetime
    score: float
    is_anomaly: bool
    model_run_id: UUID | None = Field(validation_alias="model_metrics_id")
    data_origin: DataOrigin
    deviation_pct: float | None = None
    severity: str | None = None
    actual_kwh: float | None = None
    expected_kwh: float | None = None

    model_config = {"from_attributes": True, "populate_by_name": True}


class AnomalyTimelinePoint(BaseModel):
    ts: datetime
    actual_kwh: float | None = None
    expected_kwh: float | None = None
    score: float
    is_anomaly: bool
    severity: str
    deviation_pct: float | None = None
    data_origin: DataOrigin


class AnomalyTimelineOut(BaseModel):
    building_id: str
    points: list[AnomalyTimelinePoint]


class AnomalySeverityConfigOut(BaseModel):
    warning_deviation_pct: float
    critical_deviation_pct: float
    warning_score_cutoff: float | None = None
    critical_score_cutoff: float | None = None
    flag_if_outlier: bool
    expected_floor_kwh: float


class AnomalySummaryOut(BaseModel):
    building_id: str
    n_scored: int
    n_flagged: int
    by_severity: dict[str, int]
    by_origin: dict[str, int]
    ts_min: datetime | None = None
    ts_max: datetime | None = None
    severity_config: AnomalySeverityConfigOut


class ModelPerformanceOut(BaseModel):
    """Test metrics from model_metrics or artifacts/metrics.json. Nulls if that field was not computed."""

    model_name: str
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    mape: float | None = None
    id: UUID | None = None
    created_at: datetime | None = None
    train_start: datetime | None = None
    train_end: datetime | None = None
    val_start: datetime | None = None
    val_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None
    artifact_path: str | None = None
    random_seed: int | None = None
    training_time_sec: float | None = None
    n: int | None = None
    source: str
    data_source: str | None = None
    smoke: bool | None = None

