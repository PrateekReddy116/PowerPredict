"""Live what-if inference and multi-hour recursive forecasting.

Loads saved model artifacts from the offline training CLI. Never fabricates predictions.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np

LOGGER = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts"

_TABULAR_MODELS = ("linear_regression", "random_forest", "xgboost")
_SEQUENCE_MODELS = ("lstm", "cnn_lstm")
ALL_MODEL_NAMES = (*_TABULAR_MODELS, *_SEQUENCE_MODELS)

# The feature vector order used during training (from ml.pipeline.FEATURE_COLUMNS).
# We replicate it here so the backend doesn't import the ml package at runtime.
_LAGS = (1, 2, 3, 6, 12, 24, 168)
_ROLLING_WINDOWS = (3, 6, 24, 168)
_LAG_COLUMNS = [f"lag_{lag}" for lag in _LAGS]
_ROLLING_COLUMNS = [f"rolling_mean_{window}" for window in _ROLLING_WINDOWS]
_ROLLING_EXTRA = ["rolling_std_24"]
_TEMPORAL_COLUMNS = [
    "hour", "day", "day_of_week", "month", "weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
_WEATHER_COLUMNS = ("air_temperature", "dew_temperature", "humidity", "wind_speed", "precip_depth_1hr")
_DEFAULT_FEATURE_COLUMNS = [
    "energy_kwh",
    *_LAG_COLUMNS,
    *_ROLLING_COLUMNS,
    *_ROLLING_EXTRA,
    *_TEMPORAL_COLUMNS,
    *_WEATHER_COLUMNS,
    "occupancy",
    "sqm",
]


class ArtifactNotFound(Exception):
    pass


class InsufficientHistory(Exception):
    pass


def _load_scaler(artifacts_dir: Path) -> dict[str, Any]:
    path = artifacts_dir / "feature_scaler.joblib"
    if not path.exists():
        raise ArtifactNotFound(f"feature_scaler.joblib not found in {artifacts_dir}")
    return joblib.load(path)


def _load_model(model_name: str, artifacts_dir: Path):
    if model_name in _TABULAR_MODELS:
        artifact = artifacts_dir / model_name / "model.joblib"
        if not artifact.exists():
            raise ArtifactNotFound(f"Model artifact not found: {artifact}")
        payload = joblib.load(artifact)
        return payload["model"], "tabular"
    else:
        artifact = artifacts_dir / model_name / "model.pt"
        if not artifact.exists():
            raise ArtifactNotFound(f"Model artifact not found: {artifact}")
        import torch
        try:
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(artifact, map_location="cpu")
        from ml.models import FORECASTER_CLASSES
        cls = FORECASTER_CLASSES[model_name]
        obj = cls.load(artifact)
        return obj, "sequence"


def _temporal_features(hour: int, day_of_week: int) -> dict[str, float]:
    import math
    is_weekend = 1.0 if day_of_week >= 5 else 0.0
    return {
        "hour": float(hour),
        "day": 1.0,  # day-of-month not critical for prediction
        "day_of_week": float(day_of_week),
        "month": float(datetime.now(UTC).month),
        "weekend": is_weekend,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * day_of_week / 7),
        "dow_cos": math.cos(2 * math.pi * day_of_week / 7),
    }


def _build_feature_vector(
    feature_names: list[str],
    user_inputs: dict[str, float | None],
    hour: int,
    day_of_week: int,
) -> tuple[dict[str, float], list[str]]:
    """Build a single feature dict from user inputs + defaults. Returns (features, warnings)."""
    warnings: list[str] = []
    temporal = _temporal_features(hour, day_of_week)

    feature_map: dict[str, float] = {}
    for col in feature_names:
        if col in temporal:
            feature_map[col] = temporal[col]
        elif col == "energy_kwh":
            val = user_inputs.get("energy_kwh")
            feature_map[col] = val if val is not None else 0.0
            if val is None:
                warnings.append("energy_kwh defaulted to 0")
        elif col == "occupancy":
            val = user_inputs.get("occupancy")
            feature_map[col] = val if val is not None else 50.0
        elif col == "air_temperature":
            val = user_inputs.get("temperature")
            feature_map[col] = val if val is not None else 20.0
            if val is None:
                warnings.append("temperature defaulted to 20°C")
        elif col == "humidity":
            val = user_inputs.get("humidity")
            feature_map[col] = val if val is not None else 50.0
            if val is None:
                warnings.append("humidity defaulted to 50%")
        elif col == "wind_speed":
            val = user_inputs.get("wind_speed")
            feature_map[col] = val if val is not None else 3.0
            if val is None:
                warnings.append("wind_speed defaulted to 3 m/s")
        elif col == "dew_temperature":
            feature_map[col] = (user_inputs.get("temperature") or 20.0) - 5.0
        elif col == "precip_depth_1hr":
            feature_map[col] = 0.0
        elif col == "sqm":
            feature_map[col] = 5000.0
        elif col.startswith("lag_"):
            energy = user_inputs.get("energy_kwh")
            feature_map[col] = energy if energy is not None else 0.0
        elif col.startswith("rolling_mean_") or col.startswith("rolling_std_"):
            energy = user_inputs.get("energy_kwh")
            feature_map[col] = energy if energy is not None else 0.0
        else:
            feature_map[col] = 0.0
            warnings.append(f"{col} defaulted to 0")

    return feature_map, warnings


def prepare_inference_row(
    model_name: str,
    user_inputs: dict[str, float | None],
    hour: int | None = None,
    day_of_week: int | None = None,
    is_weekend: bool | None = None,
    artifacts_dir: Path | None = None,
    *,
    load_model: bool = True,
) -> dict[str, Any]:
    """Build the same scaled feature row used by live predict. Shared with local SHAP."""
    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    scaler_data = _load_scaler(artifacts_dir)
    scaler = scaler_data["scaler"]
    feature_names: list[str] = list(scaler_data["feature_names"])
    if "target_energy_kwh" in feature_names:
        raise ValueError("target_energy_kwh must not appear in inference features")
    seq_len: int = int(scaler_data.get("seq_len", 24))

    now = datetime.now(UTC)
    if hour is None:
        hour = now.hour
    if day_of_week is None:
        if is_weekend is not None:
            day_of_week = 5 if is_weekend else 2
        else:
            day_of_week = now.weekday()

    feature_map, warnings = _build_feature_vector(feature_names, user_inputs, hour, day_of_week)
    x = np.array([[feature_map[col] for col in feature_names]], dtype=np.float64)
    x_scaled = scaler.transform(x)

    model = None
    model_type = "sequence" if model_name in _SEQUENCE_MODELS else "tabular"
    if load_model:
        model, model_type = _load_model(model_name, artifacts_dir)

    return {
        "artifacts_dir": artifacts_dir,
        "scaler": scaler,
        "feature_names": feature_names,
        "seq_len": seq_len,
        "hour": hour,
        "day_of_week": day_of_week,
        "feature_map": feature_map,
        "warnings": warnings,
        "x_unscaled": x,
        "x_scaled": x_scaled,
        "model": model,
        "model_type": model_type,
    }


def tabular_artifact_path(model_name: str, artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / model_name / "model.joblib"


def sequence_artifact_path(model_name: str, artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / model_name / "model.pt"


def predict_single(
    building_id: str,
    model_name: str,
    user_inputs: dict[str, float | None],
    hour: int | None = None,
    day_of_week: int | None = None,
    is_weekend: bool | None = None,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Single next-hour what-if prediction."""
    packed = prepare_inference_row(
        model_name,
        user_inputs,
        hour=hour,
        day_of_week=day_of_week,
        is_weekend=is_weekend,
        artifacts_dir=artifacts_dir,
    )
    model = packed["model"]
    model_type = packed["model_type"]
    feature_map = packed["feature_map"]
    warnings = packed["warnings"]
    feature_names = packed["feature_names"]
    seq_len = packed["seq_len"]
    x_scaled = packed["x_scaled"]
    hour = packed["hour"]
    day_of_week = packed["day_of_week"]

    if model_type == "sequence":
        x_seq = np.tile(x_scaled, (1, seq_len, 1)).reshape(1, seq_len, len(feature_names))
        warnings.append(f"sequence model padded with repeated input for {seq_len} steps (no real history)")
        prediction = float(model.predict(x_seq).reshape(-1)[0])
    else:
        prediction = float(model.predict(x_scaled).reshape(-1)[0])

    occ_warning = "occupancy is simulated"

    inputs_used = {
        "temperature": feature_map.get("air_temperature"),
        "humidity": feature_map.get("humidity"),
        "occupancy": feature_map.get("occupancy"),
        "energy_kwh": feature_map.get("energy_kwh"),
        "wind_speed": feature_map.get("wind_speed"),
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": day_of_week >= 5,
    }

    return {
        "prediction_kwh": round(prediction, 2),
        "model": model_name,
        "building_id": building_id,
        "horizon_hours": 1,
        "inputs_used": inputs_used,
        "warning": occ_warning,
        "defaults_note": "Missing lags/rolling features filled with energy_kwh or 0. Missing weather defaults: temp=20C, humidity=50%, wind=3m/s.",
    }


def predict_horizon(
    building_id: str,
    model_name: str,
    horizon_hours: int,
    user_inputs: dict[str, float | None],
    hour: int | None = None,
    day_of_week: int | None = None,
    artifacts_dir: Path | None = None,
) -> dict[str, Any]:
    """Recursive autoregressive multi-hour forecast."""
    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    scaler_data = _load_scaler(artifacts_dir)
    scaler = scaler_data["scaler"]
    feature_names: list[str] = scaler_data["feature_names"]
    seq_len: int = scaler_data.get("seq_len", 24)

    now = datetime.now(UTC)
    if hour is None:
        hour = now.hour
    if day_of_week is None:
        day_of_week = now.weekday()

    model, model_type = _load_model(model_name, artifacts_dir)

    feature_map, _ = _build_feature_vector(feature_names, user_inputs, hour, day_of_week)
    predictions: list[dict[str, Any]] = []

    # For sequence models, maintain a window of feature vectors
    if model_type == "sequence":
        x_row = np.array([feature_map[col] for col in feature_names], dtype=np.float64)
        x_scaled_row = scaler.transform(x_row.reshape(1, -1)).reshape(-1)
        window = np.tile(x_scaled_row, (seq_len, 1))  # (seq_len, n_features)

    current_energy = feature_map.get("energy_kwh", 0.0)
    current_hour = hour
    current_dow = day_of_week
    base_ts = now.replace(minute=0, second=0, microsecond=0)

    for step in range(1, horizon_hours + 1):
        # Advance calendar
        current_hour = (hour + step) % 24
        if step > 0 and current_hour == 0:
            current_dow = (current_dow + 1) % 7

        import math
        temporal = _temporal_features(current_hour, current_dow)

        step_map: dict[str, float] = {}
        for col in feature_names:
            if col in temporal:
                step_map[col] = temporal[col]
            elif col == "energy_kwh":
                step_map[col] = current_energy
            elif col.startswith("lag_1"):
                step_map[col] = current_energy
            elif col.startswith("lag_") or col.startswith("rolling_"):
                step_map[col] = feature_map.get(col, current_energy)
            elif col in ("air_temperature", "humidity", "wind_speed", "dew_temperature", "precip_depth_1hr", "occupancy", "sqm"):
                step_map[col] = feature_map[col]
            else:
                step_map[col] = feature_map.get(col, 0.0)

        x_row = np.array([step_map[col] for col in feature_names], dtype=np.float64)
        x_scaled_row = scaler.transform(x_row.reshape(1, -1))

        if model_type == "sequence":
            window = np.roll(window, -1, axis=0)
            window[-1] = x_scaled_row.reshape(-1)
            x_input = window.reshape(1, seq_len, len(feature_names))
            pred = float(model.predict(x_input).reshape(-1)[0])
        else:
            pred = float(model.predict(x_scaled_row).reshape(-1)[0])

        current_energy = pred
        ts = base_ts + timedelta(hours=step)
        predictions.append({
            "hour_ahead": step,
            "prediction_kwh": round(pred, 2),
            "timestamp_utc": ts.isoformat(),
        })

    return {
        "building_id": building_id,
        "model": model_name,
        "horizon_hours": horizon_hours,
        "method": "recursive",
        "note": "weather/occupancy held constant across horizon",
        "warning": "occupancy is simulated",
        "predictions": predictions,
    }
