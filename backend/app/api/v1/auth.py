from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.security import create_access_token, get_current_user, verify_password
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import LoginRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    email = str(body.email).strip().lower()
    user = db.query(User).filter(User.email == email).one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    token = create_access_token(user_id=user.id, role=user.role, settings=settings)
    return TokenResponse(access_token=token, role=user.role, email=user.email)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user
