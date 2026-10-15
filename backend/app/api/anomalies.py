from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import OptionalBuildingIdQuery, ensure_time_range, require_building
from app.core.config import Settings, get_settings
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.enums import DataOrigin
from app.models.user import User
from app.schemas.catalog import AnomalyOut
from app.services.anomalies import anomaly_settings_from_app, query_anomalies, serialize_row

router = APIRouter(tags=["anomalies"])

_UNSCOPED_LIMIT = 500
_SCOPED_LIMIT = 10_000


@router.get("/anomalies", response_model=list[AnomalyOut])
def list_anomalies(
    building_id: OptionalBuildingIdQuery = None,
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    origin: DataOrigin | None = Query(None),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[dict]:
    ensure_time_range(from_ts, to_ts)
    if building_id is not None:
        require_building(building_id, db)
        limit = _SCOPED_LIMIT
    else:
        limit = _UNSCOPED_LIMIT
    origin_value = origin.value if origin is not None else None
    cfg = anomaly_settings_from_app(settings)
    rows = query_anomalies(
        db,
        building_id,
        from_ts=from_ts,
        to_ts=to_ts,
        origin=origin_value,
        limit=limit,
    )
    return [serialize_row(row, cfg) for row in rows]
