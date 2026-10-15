from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.catalog import ModelPerformanceOut
from app.services.models import list_model_performance

router = APIRouter(tags=["models"])


@router.get("/model-performance", response_model=list[ModelPerformanceOut])
def model_performance(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> list[dict]:
    """MAE/RMSE/R²/MAPE from stored runs or artifacts/metrics.json. Empty if neither exists."""
    return list_model_performance(db, Path(settings.artifacts_dir))
