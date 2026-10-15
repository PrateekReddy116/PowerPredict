"""Live what-if prediction and multi-hour recursive forecast endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.security import get_current_user
from app.models.user import User
from app.services.predict import (
    ArtifactNotFound,
    InsufficientHistory,
    predict_horizon,
    predict_single,
)

router = APIRouter(prefix="/predict", tags=["predict"])


class PredictRequest(BaseModel):
    building_id: str
    model: str = Field(description="linear_regression | random_forest | xgboost | lstm | cnn_lstm")
    temperature: float | None = None
    humidity: float | None = None
    occupancy: float | None = None
    energy_kwh: float | None = None
    wind_speed: float | None = None
    hour: int | None = Field(None, ge=0, le=23)
    day_of_week: int | None = Field(None, ge=0, le=6)
    is_weekend: bool | None = None


class HorizonRequest(BaseModel):
    building_id: str
    model: str
    horizon_hours: int = Field(24, ge=1, le=48)
    temperature: float | None = None
    humidity: float | None = None
    occupancy: float | None = None
    energy_kwh: float | None = None


@router.post("")
def post_predict(
    body: PredictRequest,
    _user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    valid_models = ("linear_regression", "random_forest", "xgboost", "lstm", "cnn_lstm")
    if body.model not in valid_models:
        raise HTTPException(422, f"model must be one of {valid_models}")
    user_inputs = {
        "temperature": body.temperature,
        "humidity": body.humidity,
        "occupancy": body.occupancy,
        "energy_kwh": body.energy_kwh,
        "wind_speed": body.wind_speed,
    }
    try:
        result = predict_single(
            building_id=body.building_id,
            model_name=body.model,
            user_inputs=user_inputs,
            hour=body.hour,
            day_of_week=body.day_of_week,
            is_weekend=body.is_weekend,
            artifacts_dir=Path(settings.artifacts_dir),
        )
    except ArtifactNotFound as e:
        raise HTTPException(404, str(e))
    except InsufficientHistory as e:
        raise HTTPException(422, str(e))
    return result


@router.post("/horizon")
def post_predict_horizon(
    body: HorizonRequest,
    _user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
):
    valid_models = ("linear_regression", "random_forest", "xgboost", "lstm", "cnn_lstm")
    if body.model not in valid_models:
        raise HTTPException(422, f"model must be one of {valid_models}")
    user_inputs = {
        "temperature": body.temperature,
        "humidity": body.humidity,
        "occupancy": body.occupancy,
        "energy_kwh": body.energy_kwh,
    }
    try:
        result = predict_horizon(
            building_id=body.building_id,
            model_name=body.model,
            horizon_hours=body.horizon_hours,
            user_inputs=user_inputs,
            artifacts_dir=Path(settings.artifacts_dir),
        )
    except ArtifactNotFound as e:
        raise HTTPException(404, str(e))
    except InsufficientHistory as e:
        raise HTTPException(422, str(e))
    return result
