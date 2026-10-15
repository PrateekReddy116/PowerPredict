from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import BuildingIdQuery, ensure_time_range, require_building
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.enums import DataOrigin
from app.models.reading import SensorReading
from app.models.user import User
from app.schemas.catalog import ReadingOut
from app.services.readings import HISTORY_LIMIT, query_readings

router = APIRouter(prefix="/energy", tags=["energy"])


@router.get("/history", response_model=list[ReadingOut])
def energy_history(
    building_id: BuildingIdQuery,
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    origin: DataOrigin | None = Query(None),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SensorReading]:
    ensure_time_range(from_ts, to_ts)
    require_building(building_id, db)
    origin_value = origin.value if origin is not None else None
    return query_readings(
        db,
        building_id=building_id,
        from_ts=from_ts,
        to_ts=to_ts,
        origin=origin_value,
        limit=HISTORY_LIMIT,
    )
