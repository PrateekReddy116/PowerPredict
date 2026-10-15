from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.enums import UserRole
from app.models.user import User

_bearer = HTTPBearer(auto_error=False)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(
    *,
    user_id: UUID,
    role: str,
    settings: Settings | None = None,
) -> str:
    settings = settings or get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])


def try_decode_user_id(authorization: str | None) -> str | None:
    """Best-effort JWT peek for user-based rate limiting. Invalid tokens are ignored here."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        return None
    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError:
        return None
    sub = payload.get("sub")
    return str(sub) if sub else None


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None or creds.scheme.lower() != "bearer":
        raise unauthorized
    try:
        payload = decode_access_token(creds.credentials, settings)
        user_id = payload.get("sub")
        if not user_id:
            raise unauthorized
    except jwt.PyJWTError as exc:
        raise unauthorized from exc
    user = db.get(User, UUID(str(user_id)))
    if user is None:
        raise unauthorized
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != UserRole.admin.value:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user
