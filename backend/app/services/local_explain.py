"""Live local SHAP for one what-if row. Never returns stored local.json. Never KernelSHAP."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from app.services.predict import (
    ArtifactNotFound,
    _SEQUENCE_MODELS,
    prepare_inference_row,
    sequence_artifact_path,
    tabular_artifact_path,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LIVE_SHAP_UNAVAILABLE = (
    "Live local SHAP is not served for LSTM/CNN-LSTM. KernelSHAP is never run on an HTTP "
    "request. Flattened permutation SHAP on a sequence window is too expensive for interactive "
    "what-if. Use the Explain page for stored offline permutation SHAP from `python -m ml.explain`."
)


class LiveShapNotSupported(Exception):
    def __init__(self, detail: str = LIVE_SHAP_UNAVAILABLE) -> None:
        super().__init__(detail)
        self.detail = detail


def _scaled_background(scaler: Any, n_features: int) -> np.ndarray:
    mean = getattr(scaler, "mean_", None)
    if mean is not None:
        mean_row = np.asarray(mean, dtype=np.float64).reshape(1, -1)
        if mean_row.shape[1] == n_features:
            return np.asarray(scaler.transform(mean_row), dtype=np.float64)
    return np.zeros((1, n_features), dtype=np.float64)


def explain_local_whatif(
    *,
    building_id: str,
    model_name: str,
    user_inputs: dict[str, float | None],
    hour: int | None = None,
    day_of_week: int | None = None,
    is_weekend: bool | None = None,
    artifacts_dir: Path,
) -> dict[str, Any]:
    """TreeExplainer / LinearExplainer on the same feature vector as predict_single."""
    from ml.explain.compute import shap_tabular
    from ml.explain.narrative import narrative_from_contributions
    from ml.explain.registry import is_live_explainer

    if model_name in _SEQUENCE_MODELS or not is_live_explainer(model_name):
        prepare_inference_row(
            model_name,
            user_inputs,
            hour=hour,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            artifacts_dir=artifacts_dir,
            load_model=False,
        )
        seq_path = sequence_artifact_path(model_name, artifacts_dir)
        if not seq_path.exists():
            raise ArtifactNotFound(f"Model artifact not found: {seq_path}")
        raise LiveShapNotSupported()

    tab_path = tabular_artifact_path(model_name, artifacts_dir)
    if not tab_path.exists():
        raise ArtifactNotFound(f"Model artifact not found: {tab_path}")

    packed = prepare_inference_row(
        model_name,
        user_inputs,
        hour=hour,
        day_of_week=day_of_week,
        is_weekend=is_weekend,
        artifacts_dir=artifacts_dir,
        load_model=True,
    )
    feature_names: list[str] = packed["feature_names"]
    if "target_energy_kwh" in feature_names:
        raise ValueError("target_energy_kwh must not appear in inference features")

    sklearn_model = packed["model"]
    x_scaled = packed["x_scaled"]
    x_unscaled = packed["x_unscaled"]
    scaler = packed["scaler"]
    n_features = len(feature_names)
    background = _scaled_background(scaler, n_features)

    shap_values, base_values, used = shap_tabular(
        model_name=model_name,
        sklearn_model=sklearn_model,
        X=x_scaled,
        background=background,
    )
    shap_row = np.asarray(shap_values[0], dtype=np.float64)
    values_row = np.asarray(x_unscaled[0], dtype=np.float64)
    prediction = float(sklearn_model.predict(x_scaled).reshape(-1)[0])
    base_value = float(base_values[0]) if len(base_values) else float("nan")

    features: list[dict[str, Any]] = []
    for name, value, shap_value in zip(feature_names, values_row, shap_row, strict=True):
        phi = float(shap_value)
        features.append(
            {
                "name": str(name),
                "value": float(value),
                "shap_value": phi,
                "contribution": phi,
            }
        )

    occupancy_source = "simulated"
    narrative = narrative_from_contributions(
        features,
        occupancy_source=occupancy_source,
        prediction=prediction,
    )
    return {
        "prediction": round(prediction, 2),
        "prediction_kwh": round(prediction, 2),
        "base_value": base_value,
        "features": features,
        "model": model_name,
        "model_name": model_name,
        "building_id": building_id,
        "explainer": used,
        "explainer_mode": "live",
        "occupancy_source": occupancy_source,
        "narrative": narrative,
        "live": True,
        "warning": "occupancy is simulated",
    }
