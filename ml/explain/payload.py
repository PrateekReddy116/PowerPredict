"""Structured explanation payloads. Numbers must come from an explainer, never literals."""

from __future__ import annotations

from typing import Any

from ml.explain.narrative import narrative_from_contributions, narrative_from_mean_abs
from ml.models.base import json_safe


def _float(value: Any) -> float:
    return float(value)


def feature_entry(*, name: str, value: float, contribution: float) -> dict[str, Any]:
    return {
        "name": str(name),
        "value": _float(value),
        "contribution": _float(contribution),
    }


def split_signed(features: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive = [row for row in features if float(row["contribution"]) > 0]
    negative = [row for row in features if float(row["contribution"]) < 0]
    positive.sort(key=lambda row: float(row["contribution"]), reverse=True)
    negative.sort(key=lambda row: float(row["contribution"]))
    return positive, negative


def local_payload(
    *,
    prediction: float,
    feature_names: list[str],
    feature_values: list[float] | Any,
    contributions: list[float] | Any,
    model_name: str,
    explainer: str,
    building_id: str | None = None,
    ts: str | None = None,
    data_origin: str | None = None,
    occupancy_source: str | None = None,
    expected_value: float | None = None,
    row_index: int | None = None,
    data_label: str | None = None,
    explainer_mode: str | None = None,
) -> dict[str, Any]:
    values = [_float(v) for v in list(feature_values)]
    contribs = [_float(v) for v in list(contributions)]
    if len(feature_names) != len(values) or len(feature_names) != len(contribs):
        raise ValueError("feature names, values, and contributions must be the same length")
    features = [
        feature_entry(name=name, value=val, contribution=contrib)
        for name, val, contrib in zip(feature_names, values, contribs, strict=True)
    ]
    positive, negative = split_signed(features)
    payload: dict[str, Any] = {
        "prediction": _float(prediction),
        "features": features,
        "positive_contributions": positive,
        "negative_contributions": negative,
        "model_name": model_name,
        "building_id": building_id,
        "ts": ts,
        "explainer": explainer,
        "explainer_mode": explainer_mode,
        "data_origin": data_origin,
        "occupancy_source": occupancy_source,
        "expected_value": None if expected_value is None else _float(expected_value),
        "row_index": row_index,
        "data_label": data_label,
        "narrative": narrative_from_contributions(
            features,
            occupancy_source=occupancy_source,
            prediction=_float(prediction),
        ),
    }
    return json_safe(payload)


def ranking_from_shap(feature_names: list[str], mean_abs: list[float] | Any) -> list[dict[str, Any]]:
    pairs = [(str(name), _float(val)) for name, val in zip(feature_names, list(mean_abs), strict=True)]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [{"name": name, "mean_abs_shap": score} for name, score in pairs]


def global_payload(
    *,
    model_name: str,
    explainer: str,
    feature_names: list[str],
    mean_abs_shap: list[float] | Any,
    n_samples: int,
    data_origin: str | None = None,
    occupancy_source: str | None = None,
    data_label: str | None = None,
    explainer_mode: str | None = None,
    expected_value: float | None = None,
    summary_plot: str | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    ranking = ranking_from_shap(feature_names, mean_abs_shap)
    top_as_features = [
        {"name": row["name"], "contribution": row["mean_abs_shap"], "value": None} for row in ranking[:5]
    ]
    payload: dict[str, Any] = {
        "model_name": model_name,
        "explainer": explainer,
        "explainer_mode": explainer_mode,
        "rationale": rationale,
        "n_samples": int(n_samples),
        "ranking": ranking,
        "expected_value": None if expected_value is None else _float(expected_value),
        "data_origin": data_origin,
        "occupancy_source": occupancy_source,
        "data_label": data_label,
        "summary_plot": summary_plot,
        "narrative": narrative_from_mean_abs(
            top_as_features,
            occupancy_source=occupancy_source,
            top_k=3,
            n_samples=int(n_samples),
        ),
    }
    return json_safe(payload)
