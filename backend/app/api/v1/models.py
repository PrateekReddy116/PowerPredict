from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.enums import ModelName
from app.models.model_metric import ModelMetric
from app.models.user import User
from app.schemas.catalog import ModelRunOut
from app.schemas.explanation import GlobalExplanationOut, LocalExplanationOut
from app.services.explanations import find_local, load_global, load_locals
from app.services.models import list_model_runs

router = APIRouter(prefix="/models", tags=["models"])


def _artifacts_dir(settings: Settings) -> Path:
    return Path(settings.artifacts_dir)


@router.get("", response_model=list[ModelRunOut])
def list_models(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ModelMetric]:
    """Returns stored training runs only. Empty until offline training writes rows — never fabricated scores."""
    return list_model_runs(db)


@router.get("/{name}/explanations/global", response_model=GlobalExplanationOut)
def get_global_explanation(
    name: ModelName,
    _user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> dict:
    payload = load_global(_artifacts_dir(settings), name.value)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No global explanation artifact for this model",
        )
    return payload


@router.get("/{name}/explanations", response_model=LocalExplanationOut)
def get_local_explanation(
    name: ModelName,
    building_id: str | None = Query(default=None),
    ts: str | None = Query(default=None),
    row_index: int | None = Query(default=None),
    _user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> dict:
    items = load_locals(_artifacts_dir(settings), name.value)
    if items is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No local explanation artifact for this model",
        )
    match = find_local(items, building_id=building_id, ts=ts, row_index=row_index)
    if match is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No stored local explanation matches building_id/ts/row_index",
        )
    return match
