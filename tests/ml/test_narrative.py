from ml.explain.narrative import (
    narrative_from_contributions,
    narrative_from_mean_abs,
    phrase_for_feature,
    top_contributions,
)


def test_narrative_uses_only_provided_names():
    features = [
        {"name": "air_temperature", "value": 31.2, "contribution": 3.8},
        {"name": "occupancy", "value": 0.9, "contribution": 1.1},
        {"name": "lag_1", "value": 40.0, "contribution": 2.0},
        {"name": "wind_speed", "value": 3.0, "contribution": 0.01},
    ]
    text = narrative_from_contributions(features, occupancy_source="simulated", prediction=47.8)
    assert "47.80" in text
    assert "contributed approximately" in text
    assert "kWh to this prediction" in text
    assert "Temperature contributed" in text or "temperature" in text
    assert "recent energy consumption" in text
    assert "simulated occupancy profile" in text
    assert "measured occupancy" not in text.lower()
    assert "caused" not in text.lower()
    assert "wind" not in text  # not in top-3 after ranking... wait wind is smallest so not in top 3
    assert "humidity" not in text  # never provided
    assert "sqm" not in text


def test_narrative_does_not_invent_features_from_empty_list():
    text = narrative_from_contributions([], occupancy_source="simulated")
    assert text == "No feature contributions were provided."
    assert "temperature" not in text


def test_occupancy_phrase_is_simulated_unless_measured():
    assert phrase_for_feature("occupancy", occupancy_source="simulated") == "simulated occupancy profile"
    assert phrase_for_feature("occupancy", occupancy_source=None) == "simulated occupancy profile"
    assert phrase_for_feature("occupancy", occupancy_source="measured") == "occupancy"


def test_top_contributions_order_by_abs():
    features = [
        {"name": "a", "contribution": -5.0},
        {"name": "b", "contribution": 2.0},
        {"name": "c", "contribution": 4.0},
    ]
    top = top_contributions(features, top_k=2)
    assert [row["name"] for row in top] == ["a", "c"]


def test_global_mean_abs_wording_is_not_signed_local_contribution():
    features = [
        {"name": "air_temperature", "contribution": 1.25},
        {"name": "occupancy", "contribution": 0.40},
    ]
    text = narrative_from_mean_abs(features, occupancy_source="simulated", n_samples=8)
    assert "highest average influence" in text
    assert "mean |SHAP|" in text
    lowered = text.lower()
    assert "contributed approximately" not in lowered
    assert "to this prediction" not in lowered
    assert "simulated occupancy profile" in text
