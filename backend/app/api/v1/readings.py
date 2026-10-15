from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.reading import SensorReading
from app.models.user import User
from app.schemas.catalog import ReadingOut
from app.services.readings import LIST_LIMIT, query_readings

router = APIRouter(prefix="/readings", tags=["readings"])


@router.get("", response_model=list[ReadingOut])
def list_readings(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SensorReading]:
    return query_readings(db, limit=LIST_LIMIT)
