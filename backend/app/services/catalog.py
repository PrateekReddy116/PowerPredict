"""Building catalog. Empty list when nothing has been ingested — never invented rows."""

from sqlalchemy.orm import Session

from app.models.building import Building


def list_buildings(db: Session) -> list[Building]:
    return db.query(Building).order_by(Building.id).all()


def get_building_or_none(db: Session, building_id: str) -> Building | None:
    return db.get(Building, building_id)
