from fastapi import APIRouter

from app.schemas.health import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health() -> dict[str, str]:
    return {"status": "ok"}
