"""Natural-language text from an already-computed contribution list.

This module does not call SHAP. Unit tests may pass a fake list to check the
template. Production text is generated from real explainer output.
"""

from __future__ import annotations

from typing import Any

# Friendly phrases for known Phase 2 columns. Unknown names fall back to the column id.
FEATURE_PHRASES: dict[str, str] = {
    "air_temperature": "temperature",
    "air_temperature_c": "temperature",
    "dew_temperature": "dew point",
    "dew_temperature_c": "dew point",
    "humidity": "humidity",
    "wind_speed": "wind speed",
    "precip_depth_1hr": "precipitation",
    "occupancy": "occupancy",
    "energy_kwh": "recent energy consumption",
    "lag_1": "recent energy consumption",
    "lag_2": "recent energy consumption",
    "lag_3": "recent energy consumption",
    "lag_6": "recent energy consumption",
    "lag_12": "recent energy consumption",
    "lag_24": "yesterday's energy consumption",
    "lag_168": "last week's energy consumption",
    "rolling_mean_3": "the short-term energy average",
    "rolling_mean_6": "the short-term energy average",
    "rolling_mean_24": "the 24-hour energy average",
    "rolling_mean_168": "the week-long energy average",
    "rolling_std_24": "recent energy variability",
    "hour": "hour of day",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
    "dow_sin": "day of week",
    "dow_cos": "day of week",
    "day_of_week": "day of week",
    "day": "calendar day",
    "month": "month",
    "weekend": "weekend versus weekday",
    "is_weekend": "weekend versus weekday",
    "sqm": "floor area",
}

_OCCUPANCY_NAMES = {"occupancy"}


def _as_feature_dicts(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in features:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            contribution = float(item["contribution"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"name": name, "contribution": contribution, "value": item.get("value")})
    return out


def phrase_for_feature(name: str, *, occupancy_source: str | None = None) -> str:
    source = (occupancy_source or "simulated").strip().lower()
    if name in _OCCUPANCY_NAMES:
        if source == "simulated" or source == "":
            return "simulated occupancy profile"
        return "occupancy"
    return FEATURE_PHRASES.get(name, name.replace("_", " "))


def _join_phrases(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])} and {phrases[-1]}"


def top_contributions(
    features: list[dict[str, Any]],
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    items = _as_feature_dicts(features)
    items.sort(key=lambda row: abs(float(row["contribution"])), reverse=True)
    return items[: max(0, int(top_k))]


def narrative_from_contributions(
    features: list[dict[str, Any]],
    *,
    occupancy_source: str | None = "simulated",
    top_k: int = 3,
    prediction: float | None = None,
) -> str:
    """Build a sentence from the provided contributions only. No extra features, no causality."""
    items = _as_feature_dicts(features)
    if not items:
        return "No feature contributions were provided."

    ranked = top_contributions(items, top_k=top_k)
    phrases: list[str] = []
    seen: set[str] = set()
    for row in ranked:
        phrase = phrase_for_feature(row["name"], occupancy_source=occupancy_source)
        if phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
    if not phrases:
        return "No feature contributions were provided."

    parts: list[str] = []
    if prediction is not None:
        parts.append(f"The predicted next-hour energy is {float(prediction):.2f} kWh.")
    parts.append(f"The prediction is primarily influenced by {_join_phrases(phrases)}.")

    signed: list[str] = []
    for row in ranked[: min(3, len(ranked))]:
        phrase = phrase_for_feature(row["name"], occupancy_source=occupancy_source)
        contrib = float(row["contribution"])
        sign = "+" if contrib >= 0 else ""
        label = phrase[:1].upper() + phrase[1:] if phrase else phrase
        signed.append(
            f"{label} contributed approximately {sign}{contrib:.2f} kWh to this prediction"
        )
    if signed:
        parts.append(" ".join(s + "." for s in signed))
        parts.append("These figures are calculated associations with the model output, not proven causes.")
    return " ".join(parts)


def narrative_from_mean_abs(
    features: list[dict[str, Any]],
    *,
    occupancy_source: str | None = "simulated",
    top_k: int = 3,
    n_samples: int | None = None,
) -> str:
    """Global wording: mean |SHAP| is average influence, not a signed contribution to one row."""
    items = _as_feature_dicts(features)
    if not items:
        return "No global mean |SHAP| ranking was provided."

    ranked = top_contributions(items, top_k=top_k)
    phrases: list[str] = []
    seen: set[str] = set()
    for row in ranked:
        phrase = phrase_for_feature(row["name"], occupancy_source=occupancy_source)
        if phrase in seen:
            continue
        seen.add(phrase)
        phrases.append(phrase)
    if not phrases:
        return "No global mean |SHAP| ranking was provided."

    sample_bit = f" across {int(n_samples)} explained rows" if n_samples else ""
    parts = [
        f"The features with the highest average influence (mean |SHAP|){sample_bit} are {_join_phrases(phrases)}.",
        "Mean |SHAP| is the average magnitude of contributions on the explained subset, not a signed contribution to one prediction.",
        "These figures are calculated associations with the model output, not proven causes.",
    ]
    return " ".join(parts)
