"""Latest MQTT-origin readings from Postgres. Broker down does not matter if rows already exist."""

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.enums import IotDataOrigin
from app.models.reading import SensorReading

_IOT_ORIGINS = (IotDataOrigin.simulated_iot.value, IotDataOrigin.real_iot.value)


def query_latest_iot(
    db: Session,
    *,
    building_id: str | None = None,
    origin: str | None = None,
) -> list[SensorReading]:
    origins: tuple[str, ...]
    if origin is not None:
        origins = (origin,)
    else:
        origins = _IOT_ORIGINS

    q = db.query(
        SensorReading.building_id,
        SensorReading.data_origin,
        func.max(SensorReading.ts).label("max_ts"),
    ).filter(SensorReading.data_origin.in_(origins))
    if building_id is not None:
        q = q.filter(SensorReading.building_id == building_id)
    grouped = q.group_by(SensorReading.building_id, SensorReading.data_origin).subquery()

    rows = (
        db.query(SensorReading)
        .join(
            grouped,
            (SensorReading.building_id == grouped.c.building_id)
            & (SensorReading.data_origin == grouped.c.data_origin)
            & (SensorReading.ts == grouped.c.max_ts),
        )
        .order_by(SensorReading.building_id.asc(), SensorReading.data_origin.asc())
        .all()
    )
    return rows
