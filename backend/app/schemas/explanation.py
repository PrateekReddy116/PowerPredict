from datetime import datetime

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    name: str
    value: float
    contribution: float
    shap_value: float | None = None


class LocalExplanationOut(BaseModel):
    """One stored local SHAP payload. Numbers come from artifacts, never from dummy arrays."""

    model_config = {"extra": "ignore"}

    prediction: float
    features: list[FeatureContribution]
    model_name: str
    building_id: str | None = None
    ts: datetime | str | None = None
    explainer: str
    explainer_mode: str | None = None
    data_origin: str | None = None
    occupancy_source: str | None = None
    expected_value: float | None = None
    row_index: int | None = None
    data_label: str | None = None
    narrative: str | None = None
    positive_contributions: list[FeatureContribution] = Field(default_factory=list)
    negative_contributions: list[FeatureContribution] = Field(default_factory=list)


class FeatureRank(BaseModel):
    name: str
    mean_abs_shap: float


class GlobalExplanationOut(BaseModel):
    model_config = {"extra": "ignore"}
    model_name: str
    explainer: str
    explainer_mode: str | None = None
    rationale: str | None = None
    n_samples: int
    ranking: list[FeatureRank]
    expected_value: float | None = None
    data_origin: str | None = None
    occupancy_source: str | None = None
    data_label: str | None = None
    summary_plot: str | None = None
    importance_plot: str | None = None
    narrative: str | None = None
    scope: str = "global"


class LiveLocalExplanationOut(BaseModel):
    """Computed on this request's feature vector. Not stored local.json."""

    model_config = {"extra": "ignore"}

    prediction: float
    prediction_kwh: float
    base_value: float
    features: list[FeatureContribution]
    model: str
    model_name: str
    building_id: str
    explainer: str
    explainer_mode: str | None = "live"
    occupancy_source: str = "simulated"
    narrative: str | None = None
    live: bool = True
    warning: str | None = None
    scope: str = "local"
