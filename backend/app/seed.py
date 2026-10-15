from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.security import hash_password
from app.models.enums import UserRole
from app.models.user import User


def seed_admin(db: Session, settings: Settings) -> None:
    if not settings.seed_admin_email or not settings.seed_admin_password:
        return
    email = settings.seed_admin_email.strip().lower()
    existing = db.query(User).filter(User.email == email).one_or_none()
    if existing is not None:
        return
    db.add(
        User(
            email=email,
            password_hash=hash_password(settings.seed_admin_password),
            role=UserRole.admin.value,
        )
    )
    db.commit()
