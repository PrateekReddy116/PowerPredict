from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.catalog import ForecastOut
from app.services.forecasts import LIST_LIMIT, query_forecasts

forecasts_router = APIRouter(prefix="/forecasts", tags=["forecasts"])


@forecasts_router.get("", response_model=list[ForecastOut])
def list_forecasts(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    return query_forecasts(db, limit=LIST_LIMIT)
