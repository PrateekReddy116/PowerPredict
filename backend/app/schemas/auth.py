from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from app.models.enums import UserRole


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: UserRole
    email: EmailStr


class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    role: UserRole
    created_at: datetime

    model_config = {"from_attributes": True}
