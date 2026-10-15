"""Query stored energy observations. The API does not invent kWh values."""

from datetime import datetime

from sqlalchemy.orm import Session

from app.models.reading import SensorReading

HISTORY_LIMIT = 10_000
LIST_LIMIT = 500


def query_readings(
    db: Session,
    *,
    building_id: str | None = None,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
    origin: str | None = None,
    meter_type: str | None = None,
    limit: int = HISTORY_LIMIT,
) -> list[SensorReading]:
    q = db.query(SensorReading)
    if building_id is not None:
        q = q.filter(SensorReading.building_id == building_id)
    if from_ts is not None:
        q = q.filter(SensorReading.ts >= from_ts)
    if to_ts is not None:
        q = q.filter(SensorReading.ts <= to_ts)
    if origin is not None:
        q = q.filter(SensorReading.data_origin == origin)
    if meter_type is not None:
        q = q.filter(SensorReading.meter_type == meter_type)
    return q.order_by(SensorReading.ts.asc()).limit(limit).all()
