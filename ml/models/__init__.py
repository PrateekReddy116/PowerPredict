"""Model registry. Training is offline CLI (`python -m ml.train`); the API only loads artifacts."""

from __future__ import annotations

from ml.models.cnn_lstm import CNNLSTMForecaster
from ml.models.lstm import LSTMForecaster
from ml.models.tabular import LinearRegressionForecaster, RandomForestForecaster, XGBoostForecaster

TABULAR_MODEL_NAMES = ("linear_regression", "random_forest", "xgboost")
SEQUENCE_MODEL_NAMES = ("lstm", "cnn_lstm")
ALL_MODEL_NAMES = (*TABULAR_MODEL_NAMES, *SEQUENCE_MODEL_NAMES)

FORECASTER_CLASSES = {
    "linear_regression": LinearRegressionForecaster,
    "random_forest": RandomForestForecaster,
    "xgboost": XGBoostForecaster,
    "lstm": LSTMForecaster,
    "cnn_lstm": CNNLSTMForecaster,
}


def is_sequence_model(name: str) -> bool:
    return name in SEQUENCE_MODEL_NAMES
