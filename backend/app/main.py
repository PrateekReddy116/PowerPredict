from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.api.v1 import api_v1_router
from app.core.config import get_settings
from app.core.rate_limit import RateLimitMiddleware
from app.db.base import Base
from app.db.session import get_engine, get_session_factory
from app.models import load_all_models
from app.schemas.health import HealthOut
from app.seed import seed_admin
from app.services.mqtt_consumer import start_mqtt_consumer, stop_mqtt_consumer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_all_models()
    engine = get_engine()
    # Phase 1 convenience so `uvicorn` works on a fresh DB. Prefer `alembic upgrade head` in later phases.
    Base.metadata.create_all(bind=engine)
    db = get_session_factory()()
    try:
        seed_admin(db, get_settings())
    finally:
        db.close()
    # MQTT is fail-soft: broker down must not prevent /health or historical queries.
    start_mqtt_consumer()
    try:
        yield
    finally:
        stop_mqtt_consumer()


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="PowerPredict",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(RateLimitMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    application.include_router(api_router, prefix="/api")
    application.include_router(api_v1_router, prefix="/api/v1")

    @application.get("/health", response_model=HealthOut)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
