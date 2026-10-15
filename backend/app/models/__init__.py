from app.models.anomaly import Anomaly
from app.models.building import Building
from app.models.forecast import Forecast
from app.models.model_metric import ModelMetric
from app.models.reading import SensorReading
from app.models.stubs import Explanation, OccupancyObservation, WeatherObservation
from app.models.user import User


def load_all_models() -> None:
    """Import side-effect so metadata includes every table."""
    _ = (
        User,
        Building,
        SensorReading,
        ModelMetric,
        Forecast,
        Anomaly,
        WeatherObservation,
        OccupancyObservation,
        Explanation,
    )


__all__ = [
    "Anomaly",
    "Building",
    "Explanation",
    "Forecast",
    "ModelMetric",
    "OccupancyObservation",
    "SensorReading",
    "User",
    "WeatherObservation",
    "load_all_models",
]
