"""Live POST /explainability/local — computed SHAP, not stored local.json."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

shap = pytest.importorskip("shap")

FEATURE_NAMES = ["air_temperature", "humidity", "occupancy", "energy_kwh"]
SCENARIO_A = {"temperature": 25.0, "humidity": 50.0, "occupancy": 30.0, "energy_kwh": 100.0}
SCENARIO_B = {"temperature": 35.0, "humidity": 80.0, "occupancy": 90.0, "energy_kwh": 200.0}


def _auth(client: TestClient) -> dict[str, str]:
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "test-admin-password"},
    )
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _write_scaler_and_model(artifacts: Path, *, model_name: str, tree: bool = False) -> None:
    rng = np.random.default_rng(0)
    n = 120
    x = np.column_stack(
        [
            rng.normal(22.0, 6.0, n),
            rng.uniform(30.0, 85.0, n),
            rng.uniform(5.0, 95.0, n),
            rng.uniform(40.0, 200.0, n),
        ]
    )
    y = 0.45 * x[:, 0] + 0.08 * x[:, 1] + 0.22 * x[:, 2] + 0.75 * x[:, 3] + 12.0
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    if tree:
        est = RandomForestRegressor(n_estimators=12, max_depth=5, random_state=0)
    else:
        est = LinearRegression()
    est.fit(x_scaled, y)
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "feature_names": list(FEATURE_NAMES), "seq_len": 24},
        artifacts / "feature_scaler.joblib",
    )
    dest = artifacts / model_name
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "name": model_name,
            "model": est,
            "feature_names": list(FEATURE_NAMES),
            "config": {},
            "random_seed": 0,
        },
        dest / "model.joblib",
    )
    stored = artifacts / "explanations" / model_name
    stored.mkdir(parents=True, exist_ok=True)
    (stored / "local.json").write_text(
        json.dumps(
            [
                {
                    "prediction": 999.0,
                    "features": [
                        {"name": name, "value": 0.0, "contribution": 100.0} for name in FEATURE_NAMES
                    ],
                    "model_name": model_name,
                    "explainer": "stored_stub",
                    "occupancy_source": "simulated",
                    "row_index": 0,
                    "narrative": "stored stub must not be returned by live local SHAP",
                }
            ]
        ),
        encoding="utf-8",
    )


def _post_local(client: TestClient, headers: dict[str, str], model: str, scenario: dict) -> object:
    return client.post(
        "/api/v1/explainability/local",
        headers=headers,
        json={
            "building_id": "fixture_office_1",
            "model": model,
            "hour": 10,
            "day_of_week": 2,
            **scenario,
        },
    )


def test_local_explain_requires_auth(client: TestClient):
    resp = client.post("/api/v1/explainability/local", json={"building_id": "x", "model": "linear_regression"})
    assert resp.status_code == 401


def test_local_explain_404_without_artifacts(client: TestClient):
    headers = _auth(client)
    resp = _post_local(client, headers, "linear_regression", SCENARIO_A)
    assert resp.status_code == 404
    assert "ranking" not in resp.json()
    assert "features" not in resp.json() or "detail" in resp.json()


def test_two_scenarios_differ_and_are_not_stored_json(client: TestClient, tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_scaler_and_model(artifacts, model_name="linear_regression", tree=False)
    headers = _auth(client)
    resp_a = _post_local(client, headers, "linear_regression", SCENARIO_A)
    resp_b = _post_local(client, headers, "linear_regression", SCENARIO_B)
    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    body_a, body_b = resp_a.json(), resp_b.json()
    assert body_a["live"] is True
    assert body_a["occupancy_source"] == "simulated"
    assert body_a["prediction"] != pytest.approx(999.0)
    assert body_b["prediction"] != pytest.approx(999.0)
    assert body_a["prediction"] != pytest.approx(body_b["prediction"])
    names_a = [row["name"] for row in body_a["features"]]
    assert names_a == FEATURE_NAMES
    assert "target_energy_kwh" not in names_a
    shap_a = np.array([row["shap_value"] for row in body_a["features"]], dtype=np.float64)
    shap_b = np.array([row["shap_value"] for row in body_b["features"]], dtype=np.float64)
    assert shap_a.shape == (len(FEATURE_NAMES),)
    assert not np.allclose(shap_a, shap_b)
    assert not np.allclose(shap_a, 100.0)
    assert body_a["narrative"] != "stored stub must not be returned by live local SHAP"
    assert "contributed approximately" in body_a["narrative"]
    assert "kWh to this prediction" in body_a["narrative"]
    assert "highest average influence" not in body_a["narrative"]
    stored = client.get(
        "/api/v1/explainability",
        params={"model": "linear_regression", "row": 0},
        headers=headers,
    )
    assert stored.status_code == 200
    assert stored.json()["prediction"] == pytest.approx(999.0)
    assert stored.json()["prediction"] != pytest.approx(body_a["prediction"])


def test_random_forest_live_treeshap_two_scenarios(client: TestClient, tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_scaler_and_model(artifacts, model_name="random_forest", tree=True)
    headers = _auth(client)
    resp_a = _post_local(client, headers, "random_forest", SCENARIO_A)
    resp_b = _post_local(client, headers, "random_forest", SCENARIO_B)
    assert resp_a.status_code == 200
    assert resp_b.status_code == 200
    assert resp_a.json()["explainer"] == "TreeExplainer"
    assert resp_a.json()["prediction"] != pytest.approx(resp_b.json()["prediction"])
    shap_a = [row["shap_value"] for row in resp_a.json()["features"]]
    shap_b = [row["shap_value"] for row in resp_b.json()["features"]]
    assert shap_a != shap_b


def test_lstm_live_shap_returns_501(client: TestClient, tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_scaler_and_model(artifacts, model_name="linear_regression", tree=False)
    (artifacts / "lstm").mkdir(parents=True, exist_ok=True)
    (artifacts / "lstm" / "model.pt").write_bytes(b"not-a-real-torch-checkpoint")
    headers = _auth(client)
    resp = _post_local(client, headers, "lstm", SCENARIO_A)
    assert resp.status_code == 501
    detail = str(resp.json()["detail"]).lower()
    assert "kernelshap" in detail or "lstm" in detail
    assert "999" not in str(resp.json())
