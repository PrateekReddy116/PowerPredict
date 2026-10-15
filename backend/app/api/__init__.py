from fastapi import APIRouter

from app.api import (
    anomalies,
    buildings,
    energy,
    explainability,
    forecast,
    health,
    iot,
    models,
    performance,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(buildings.router)
api_router.include_router(energy.router)
api_router.include_router(forecast.router)
api_router.include_router(anomalies.router)
api_router.include_router(models.router)
api_router.include_router(performance.router)
api_router.include_router(explainability.router)
api_router.include_router(iot.router)
