from enum import StrEnum


class UserRole(StrEnum):
    viewer = "viewer"
    admin = "admin"


class DataOrigin(StrEnum):
    historical = "historical"
    simulated_iot = "simulated_iot"
    real_iot = "real_iot"


class IotDataOrigin(StrEnum):
    """Origins that may appear on MQTT / IoT latest (never historical)."""

    simulated_iot = "simulated_iot"
    real_iot = "real_iot"


class ModelName(StrEnum):
    """Offline training model ids. The API does not train; it only names stored artifacts."""

    linear_regression = "linear_regression"
    random_forest = "random_forest"
    xgboost = "xgboost"
    lstm = "lstm"
    cnn_lstm = "cnn_lstm"


class ForecastSplit(StrEnum):
    val = "val"
    test = "test"
    live = "live"
