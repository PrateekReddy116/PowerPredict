from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import BuildingIdQuery, require_building
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.enums import ModelName
from app.models.user import User
from app.schemas.catalog import ForecastOut
from app.services.forecasts import FORECAST_LIMIT, query_forecasts

router = APIRouter(tags=["forecasts"])


@router.get("/forecast", response_model=list[ForecastOut])
def get_forecast(
    building_id: BuildingIdQuery,
    model: ModelName | None = Query(None),
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[dict]:
    require_building(building_id, db)
    model_value = model.value if model is not None else None
    return query_forecasts(
        db,
        building_id=building_id,
        model=model_value,
        limit=FORECAST_LIMIT,
    )
