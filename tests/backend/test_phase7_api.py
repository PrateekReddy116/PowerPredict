from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient
from ml.explain.payload import global_payload, local_payload

from app.db.session import get_session_factory
from app.models.building import Building
from app.models.forecast import Forecast
from app.models.reading import SensorReading


def _auth(client: TestClient) -> dict[str, str]:
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "test-admin-password"},
    )
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_api_health_public(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-RateLimit-Limit" in response.headers


def test_phase7_routes_require_auth(client: TestClient):
    for path in (
        "/api/buildings",
        "/api/buildings/Hog_education_Casandra",
        "/api/energy/history?building_id=Hog_education_Casandra",
        "/api/forecast?building_id=Hog_education_Casandra",
        "/api/anomalies",
        "/api/models",
        "/api/model-performance",
        "/api/explainability?model=xgboost",
        "/api/iot/status",
        "/api/iot/latest",
    ):
        response = client.get(path)
        assert response.status_code == 401, path


def test_buildings_empty_then_seeded(client: TestClient):
    headers = _auth(client)
    empty = client.get("/api/buildings", headers=headers)
    assert empty.status_code == 200
    assert empty.json() == []

    db = get_session_factory()()
    try:
        db.add(
            Building(
                id="Hog_education_Casandra",
                site_id="Hog",
                primary_use="Education",
                timezone="US/Central",
                selected_reason="test seed",
            )
        )
        db.commit()
    finally:
        db.close()

    listed = client.get("/api/buildings", headers=headers)
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == "Hog_education_Casandra"
    one = client.get("/api/buildings/Hog_education_Casandra", headers=headers)
    assert one.status_code == 200
    assert one.json()["site_id"] == "Hog"
    missing = client.get("/api/buildings/does-not-exist", headers=headers)
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Building not found"


def test_energy_history_validation_and_seed(client: TestClient):
    headers = _auth(client)
    missing_id = client.get("/api/energy/history", headers=headers)
    assert missing_id.status_code == 422

    bad_origin = client.get(
        "/api/energy/history",
        params={"building_id": "Hog_education_Casandra", "origin": "not-an-origin"},
        headers=headers,
    )
    assert bad_origin.status_code == 422

    inverted = client.get(
        "/api/energy/history",
        params={
            "building_id": "Hog_education_Casandra",
            "from": "2017-02-01T00:00:00Z",
            "to": "2017-01-01T00:00:00Z",
        },
        headers=headers,
    )
    assert inverted.status_code == 422

    unknown = client.get(
        "/api/energy/history",
        params={"building_id": "Hog_education_Casandra"},
        headers=headers,
    )
    assert unknown.status_code == 404

    db = get_session_factory()()
    try:
        db.add(
            Building(
                id="Hog_education_Casandra",
                site_id="Hog",
                primary_use="Education",
                timezone="UTC",
            )
        )
        db.add(
            SensorReading(
                building_id="Hog_education_Casandra",
                ts=datetime(2017, 1, 15, 4, tzinfo=UTC),
                energy_kwh=41.2,
                data_origin="historical",
                occupancy=0.4,
                occupancy_source="simulated",
            )
        )
        db.commit()
    finally:
        db.close()

    history = client.get(
        "/api/energy/history",
        params={"building_id": "Hog_education_Casandra", "origin": "historical"},
        headers=headers,
    )
    assert history.status_code == 200
    rows = history.json()
    assert len(rows) == 1
    assert rows[0]["energy_kwh"] == 41.2
    assert rows[0]["data_origin"] == "historical"
    assert rows[0]["occupancy_source"] == "simulated"


def test_forecast_empty_not_fabricated(client: TestClient):
    headers = _auth(client)
    db = get_session_factory()()
    try:
        db.add(
            Building(
                id="Hog_education_Donnie",
                site_id="Hog",
                primary_use="Education",
                timezone="UTC",
            )
        )
        db.commit()
    finally:
        db.close()

    missing_id = client.get("/api/forecast", headers=headers)
    assert missing_id.status_code == 422
    bad_model = client.get(
        "/api/forecast",
        params={"building_id": "Hog_education_Donnie", "model": "not-a-model"},
        headers=headers,
    )
    assert bad_model.status_code == 422
    empty = client.get(
        "/api/forecast",
        params={"building_id": "Hog_education_Donnie", "model": "xgboost"},
        headers=headers,
    )
    assert empty.status_code == 200
    assert empty.json() == []

    db = get_session_factory()()
    try:
        db.add(
            Forecast(
                building_id="Hog_education_Donnie",
                target_ts=datetime(2017, 10, 1, 12, tzinfo=UTC),
                y_true=80.0,
                y_pred=79.5,
                split="test",
            )
        )
        db.commit()
    finally:
        db.close()
    stored = client.get(
        "/api/forecast",
        params={"building_id": "Hog_education_Donnie"},
        headers=headers,
    )
    assert stored.status_code == 200
    body = stored.json()
    assert len(body) == 1
    assert body[0]["y_pred"] == 79.5
    assert body[0]["y_true"] == 80.0


def test_models_and_performance_empty_without_artifacts(client: TestClient):
    headers = _auth(client)
    models = client.get("/api/models", headers=headers)
    assert models.status_code == 200
    assert models.json() == []
    performance = client.get("/api/model-performance", headers=headers)
    assert performance.status_code == 200
    assert performance.json() == []


def test_model_performance_reads_metrics_json_not_dummies(tmp_path: Path, monkeypatch):
    metrics = {
        "data_source": "TEST FIXTURE",
        "smoke": True,
        "models": [
            {
                "model_name": "linear_regression",
                "mae": 0.8670769658896456,
                "rmse": 0.8918454934215417,
                "r2": 0.9525793863237679,
                "mape": 6.121034622704997,
                "n": 118,
                "train_start": "2017-01-08T00:00:00+00:00",
                "train_end": "2017-01-12T15:00:00+00:00",
            }
        ],
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-key-32-chars-min")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'perf.db').as_posix()}")
    monkeypatch.setenv("SEED_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("SEED_ADMIN_PASSWORD", "test-admin-password")
    monkeypatch.setenv("MQTT_ENABLED", "false")
    from app.core.config import reset_settings_cache
    from app.core.rate_limit import limiter
    from app.db.session import reset_engine
    from app.main import create_app
    from app.services.mqtt_consumer import reset_runtime_status

    reset_settings_cache()
    reset_engine()
    limiter.reset()
    reset_runtime_status()
    app = create_app()
    with TestClient(app) as test_client:
        headers = _auth(test_client)
        listed = test_client.get("/api/models", headers=headers)
        assert listed.status_code == 200
        assert listed.json() == []
        perf = test_client.get("/api/model-performance", headers=headers)
        assert perf.status_code == 200
        rows = perf.json()
        assert len(rows) == 1
        assert rows[0]["mae"] == 0.8670769658896456
        assert rows[0]["source"] == "artifacts"
        assert rows[0]["data_source"] == "TEST FIXTURE"
        assert rows[0]["smoke"] is True
    reset_engine()
    reset_settings_cache()
    limiter.reset()
    reset_runtime_status()


def test_explainability_404_has_no_dummy_shap(client: TestClient):
    headers = _auth(client)
    missing_model = client.get("/api/explainability", headers=headers)
    assert missing_model.status_code == 422
    missing = client.get("/api/explainability", params={"model": "xgboost"}, headers=headers)
    assert missing.status_code == 404
    body = missing.json()
    assert "detail" in body
    assert "ranking" not in body
    assert "features" not in body
    assert "0.123456789" not in str(body)


def test_explainability_returns_stored_payload(tmp_path: Path, monkeypatch):
    dest = tmp_path / "explanations" / "linear_regression"
    dest.mkdir(parents=True)
    global_body = global_payload(
        model_name="linear_regression",
        explainer="LinearExplainer",
        feature_names=["air_temperature", "occupancy"],
        mean_abs_shap=[1.25, 0.40],
        n_samples=4,
        occupancy_source="simulated",
        data_label="TEST FIXTURE",
        explainer_mode="live",
    )
    local_body = local_payload(
        prediction=12.5,
        feature_names=["air_temperature", "occupancy"],
        feature_values=[18.0, 0.8],
        contributions=[1.1, -0.4],
        model_name="linear_regression",
        explainer="LinearExplainer",
        building_id="fixture_office_1",
        ts="2017-01-15T04:00:00+00:00",
        occupancy_source="simulated",
        data_origin="historical",
        row_index=0,
        data_label="TEST FIXTURE",
        explainer_mode="live",
        expected_value=11.8,
    )
    (dest / "global.json").write_text(json.dumps(global_body), encoding="utf-8")
    (dest / "local.json").write_text(json.dumps([local_body]), encoding="utf-8")

    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret-key-32-chars-min")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'exp.db').as_posix()}")
    monkeypatch.setenv("SEED_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("SEED_ADMIN_PASSWORD", "test-admin-password")
    monkeypatch.setenv("MQTT_ENABLED", "false")
    from app.core.config import reset_settings_cache
    from app.core.rate_limit import limiter
    from app.db.session import reset_engine
    from app.main import create_app
    from app.services.mqtt_consumer import reset_runtime_status

    reset_settings_cache()
    reset_engine()
    limiter.reset()
    reset_runtime_status()
    app = create_app()
    with TestClient(app) as test_client:
        headers = _auth(test_client)
        global_resp = test_client.get(
            "/api/explainability",
            params={"model": "linear_regression"},
            headers=headers,
        )
        assert global_resp.status_code == 200
        assert global_resp.json()["ranking"][0]["mean_abs_shap"] == 1.25
        local_resp = test_client.get(
            "/api/explainability",
            params={
                "model": "linear_regression",
                "building_id": "fixture_office_1",
                "ts": "2017-01-15T04:00:00+00:00",
            },
            headers=headers,
        )
        assert local_resp.status_code == 200
        assert local_resp.json()["prediction"] == 12.5
        assert "simulated occupancy" in local_resp.json()["narrative"]
    reset_engine()
    reset_settings_cache()
    limiter.reset()
    reset_runtime_status()


def test_iot_status_offline_and_latest_empty(client: TestClient):
    headers = _auth(client)
    status = client.get("/api/iot/status", headers=headers)
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "OFFLINE"
    assert body["broker_connected"] is False
    latest = client.get("/api/iot/latest", headers=headers)
    assert latest.status_code == 200
    assert latest.json() == []
    historical_origin = client.get(
        "/api/iot/latest",
        params={"origin": "historical"},
        headers=headers,
    )
    assert historical_origin.status_code == 422


def test_v1_aliases_still_work(client: TestClient):
    headers = _auth(client)
    assert client.get("/api/v1/buildings", headers=headers).status_code == 200
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/iot/status", headers=headers).json()["status"] == "OFFLINE"
    assert client.get("/api/v1/anomalies", headers=headers).json() == []
    assert client.get("/api/v1/models", headers=headers).json() == []
    assert client.get("/api/v1/forecasts", headers=headers).json() == []
