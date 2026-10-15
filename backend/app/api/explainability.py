from pathlib import Path

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.api.deps import OptionalBuildingIdQuery
from app.core.config import Settings, get_settings
from app.core.errors import not_found, not_implemented
from app.core.security import get_current_user
from app.models.enums import ModelName
from app.models.user import User
from app.schemas.explanation import GlobalExplanationOut, LiveLocalExplanationOut, LocalExplanationOut
from app.services.explanations import find_local, load_global, load_locals
from app.services.local_explain import LiveShapNotSupported, explain_local_whatif
from app.services.predict import ArtifactNotFound

router = APIRouter(tags=["explainability"])


class LocalExplainRequest(BaseModel):
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


@router.post(
    "/explainability/local",
    response_model=LiveLocalExplanationOut,
)
def post_local_explainability(
    body: LocalExplainRequest,
    _user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Live local SHAP for one what-if row (same features as POST /predict). Never stored local.json."""
    valid_models = tuple(m.value for m in ModelName)
    if body.model not in valid_models:
        from fastapi import HTTPException

        raise HTTPException(422, f"model must be one of {valid_models}")
    user_inputs = {
        "temperature": body.temperature,
        "humidity": body.humidity,
        "occupancy": body.occupancy,
        "energy_kwh": body.energy_kwh,
        "wind_speed": body.wind_speed,
    }
    try:
        return explain_local_whatif(
            building_id=body.building_id,
            model_name=body.model,
            user_inputs=user_inputs,
            hour=body.hour,
            day_of_week=body.day_of_week,
            is_weekend=body.is_weekend,
            artifacts_dir=Path(settings.artifacts_dir),
        )
    except ArtifactNotFound as exc:
        raise not_found(str(exc)) from exc
    except LiveShapNotSupported as exc:
        raise not_implemented(exc.detail) from exc


@router.get(
    "/explainability",
    response_model=GlobalExplanationOut | LocalExplanationOut,
)
def explainability(
    model: ModelName,
    building_id: OptionalBuildingIdQuery = None,
    ts: str | None = Query(default=None),
    row: int | None = Query(default=None, ge=0),
    row_index: int | None = Query(default=None, ge=0),
    _user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Load stored SHAP JSON. Missing artifacts → 404, never dummy contributions.

    Without row/building/ts: global mean |SHAP|. With those query params: stored CLI local example.
    """
    artifacts = Path(settings.artifacts_dir)
    local_index = row if row is not None else row_index
    wants_local = building_id is not None or ts is not None or local_index is not None
    if wants_local:
        items = load_locals(artifacts, model.value)
        if items is None:
            raise not_found("No local explanation artifact for this model")
        match = find_local(
            items,
            building_id=building_id,
            ts=ts,
            row_index=local_index,
        )
        if match is None:
            raise not_found("No stored local explanation matches building_id/ts/row_index")
        return match
    payload = load_global(artifacts, model.value)
    if payload is None:
        raise not_found("No global explanation artifact for this model")
    return payload
