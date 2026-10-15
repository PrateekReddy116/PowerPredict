"""Building-scoped Isolation Forest results. Empty lists when nothing has been scored."""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import BuildingIdPath, ensure_time_range, require_building
from app.core.config import Settings, get_settings
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.enums import DataOrigin
from app.models.user import User
from app.schemas.catalog import AnomalyOut, AnomalySummaryOut, AnomalyTimelineOut
from app.services.anomalies import (
    anomaly_settings_from_app,
    query_anomalies,
    serialize_row,
    summary_payload,
)

router = APIRouter(prefix="/buildings", tags=["anomalies"])


@router.get("/{building_id}/anomalies/summary", response_model=AnomalySummaryOut)
def building_anomaly_summary(
    building_id: BuildingIdPath,
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    origin: DataOrigin | None = Query(None),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    ensure_time_range(from_ts, to_ts)
    require_building(building_id, db)
    cfg = anomaly_settings_from_app(settings)
    origin_value = origin.value if origin is not None else None
    rows = query_anomalies(db, building_id, from_ts=from_ts, to_ts=to_ts, origin=origin_value)
    return summary_payload(building_id, rows, cfg)


@router.get("/{building_id}/anomalies/timeline", response_model=AnomalyTimelineOut)
def building_anomaly_timeline(
    building_id: BuildingIdPath,
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    origin: DataOrigin | None = Query(None),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    ensure_time_range(from_ts, to_ts)
    require_building(building_id, db)
    cfg = anomaly_settings_from_app(settings)
    origin_value = origin.value if origin is not None else None
    rows = query_anomalies(db, building_id, from_ts=from_ts, to_ts=to_ts, origin=origin_value)
    points = []
    for row in rows:
        payload = serialize_row(row, cfg)
        points.append(
            {
                "ts": payload["ts"],
                "actual_kwh": payload["actual_kwh"],
                "expected_kwh": payload["expected_kwh"],
                "score": payload["score"],
                "is_anomaly": payload["is_anomaly"],
                "severity": payload["severity"],
                "deviation_pct": payload["deviation_pct"],
                "data_origin": payload["data_origin"],
            }
        )
    return {"building_id": building_id, "points": points}


@router.get("/{building_id}/anomalies", response_model=list[AnomalyOut])
def list_building_anomalies(
    building_id: BuildingIdPath,
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    origin: DataOrigin | None = Query(None),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[dict]:
    ensure_time_range(from_ts, to_ts)
    require_building(building_id, db)
    cfg = anomaly_settings_from_app(settings)
    origin_value = origin.value if origin is not None else None
    rows = query_anomalies(db, building_id, from_ts=from_ts, to_ts=to_ts, origin=origin_value)
    return [serialize_row(row, cfg) for row in rows]
