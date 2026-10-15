from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import BuildingIdPath, require_building
from app.core.security import get_current_user
from app.db.session import get_db
from app.models.building import Building
from app.models.user import User
from app.schemas.catalog import BuildingOut
from app.services.catalog import list_buildings

router = APIRouter(prefix="/buildings", tags=["buildings"])


@router.get("", response_model=list[BuildingOut])
def get_buildings(
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Building]:
    return list_buildings(db)


@router.get("/{id}", response_model=BuildingOut)
def get_building(
    id: BuildingIdPath,
    _user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Building:
    return require_building(id, db)
