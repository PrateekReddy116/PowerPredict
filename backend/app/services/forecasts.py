"""Query stored next-hour forecasts. Empty when no training run wrote rows — never dummy y_pred."""

from sqlalchemy.orm import Session

from app.models.forecast import Forecast
from app.models.model_metric import ModelMetric

FORECAST_LIMIT = 10_000
LIST_LIMIT = 500


def serialize_forecast(row: Forecast, model_name: str | None = None) -> dict:
    return {
        "id": row.id,
        "building_id": row.building_id,
        "model_run_id": row.model_metrics_id,
        "model_metrics_id": row.model_metrics_id,
        "target_ts": row.target_ts,
        "y_true": row.y_true,
        "y_pred": row.y_pred,
        "split": row.split,
        "model_name": model_name,
    }


def query_forecasts(
    db: Session,
    *,
    building_id: str | None = None,
    model: str | None = None,
    limit: int = FORECAST_LIMIT,
) -> list[dict]:
    q = db.query(Forecast, ModelMetric.model_name).outerjoin(
        ModelMetric, Forecast.model_metrics_id == ModelMetric.id
    )
    if building_id is not None:
        q = q.filter(Forecast.building_id == building_id)
    if model is not None:
        q = q.filter(ModelMetric.model_name == model)
    rows = q.order_by(Forecast.target_ts.asc()).limit(limit).all()
    return [serialize_forecast(forecast, name) for forecast, name in rows]
