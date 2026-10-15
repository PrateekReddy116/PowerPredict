from fastapi import APIRouter

from app.api import anomalies as anomalies_index
from app.api import energy, explainability, forecast, health, performance
from app.api.v1 import anomalies, auth, buildings, iot, models, predict, readings, series

api_v1_router = APIRouter()
api_v1_router.include_router(auth.router)
api_v1_router.include_router(health.router)
api_v1_router.include_router(buildings.router)
api_v1_router.include_router(anomalies.router)
api_v1_router.include_router(anomalies_index.router)
api_v1_router.include_router(readings.router)
api_v1_router.include_router(iot.router)
api_v1_router.include_router(models.router)
api_v1_router.include_router(series.forecasts_router)
api_v1_router.include_router(predict.router)
api_v1_router.include_router(energy.router)
api_v1_router.include_router(forecast.router)
api_v1_router.include_router(explainability.router)
api_v1_router.include_router(performance.router)
