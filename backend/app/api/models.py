from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.model_metric import ModelMetric
from app.models.user import User
from app.schemas.catalog import ModelRunOut
from app.services.models import list_model_runs

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelRunOut])
def get_models(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ModelMetric]:
    """Stored training runs only. Empty until offline training writes rows — never fabricated scores."""
    return list_model_runs(db)
