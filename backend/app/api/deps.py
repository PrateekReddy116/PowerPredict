"""Shared query/path validation and building lookup. Routers stay thin."""

from datetime import datetime
from typing import Annotated

from fastapi import Path, Query
from sqlalchemy.orm import Session

from app.core.errors import not_found, unprocessable
from app.models.building import Building
from app.services.catalog import get_building_or_none

# BDG2 ids (e.g. Hog_education_Casandra) and simulator ids (building_001).
BUILDING_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"

BuildingIdPath = Annotated[
    str,
    Path(min_length=1, max_length=64, pattern=BUILDING_ID_PATTERN),
]
BuildingIdQuery = Annotated[
    str,
    Query(min_length=1, max_length=64, pattern=BUILDING_ID_PATTERN),
]
OptionalBuildingIdQuery = Annotated[
    str | None,
    Query(min_length=1, max_length=64, pattern=BUILDING_ID_PATTERN),
]


def ensure_time_range(from_ts: datetime | None, to_ts: datetime | None) -> None:
    if from_ts is not None and to_ts is not None and from_ts > to_ts:
        raise unprocessable("`from` must be less than or equal to `to`")


def require_building(building_id: str, db: Session) -> Building:
    row = get_building_or_none(db, building_id)
    if row is None:
        raise not_found("Building not found")
    return row
