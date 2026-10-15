from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import OptionalBuildingIdQuery
from app.core.config import Settings, get_settings
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.enums import IotDataOrigin
from app.models.reading import SensorReading
from app.models.user import User
from app.schemas.catalog import ReadingOut
from app.schemas.iot import IotStatusOut
from app.services.iot_query import query_latest_iot
from app.services.mqtt_consumer import status_payload

router = APIRouter(prefix="/iot", tags=["iot"])


@router.get("/status", response_model=IotStatusOut)
def iot_status(
    _user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> dict:
    return status_payload(settings)


@router.get("/latest", response_model=list[ReadingOut])
def iot_latest(
    building_id: OptionalBuildingIdQuery = None,
    origin: IotDataOrigin | None = Query(default=None),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SensorReading]:
    origin_value = origin.value if origin is not None else None
    return query_latest_iot(db, building_id=building_id, origin=origin_value)
