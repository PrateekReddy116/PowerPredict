import json
from pathlib import Path

from ml.explain.payload import global_payload, local_payload


def _auth(client) -> dict[str, str]:
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "test-admin-password"},
    )
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_explanations_require_auth(client):
    assert client.get("/api/v1/models/xgboost/explanations/global").status_code == 401
    assert client.get("/api/v1/models/xgboost/explanations").status_code == 401


def test_explanations_404_when_artifact_missing(client):
    headers = _auth(client)
    missing = client.get("/api/v1/models/xgboost/explanations/global", headers=headers)
    assert missing.status_code == 404
    local = client.get("/api/v1/models/xgboost/explanations", headers=headers)
    assert local.status_code == 404


def test_explanations_return_stored_payload(tmp_path: Path, monkeypatch):
    """API reads files written by the explainer payload helper — not canned SHAP literals."""
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
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    monkeypatch.setenv("SEED_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("SEED_ADMIN_PASSWORD", "test-admin-password")
    from fastapi.testclient import TestClient

    from app.core.config import reset_settings_cache
    from app.core.rate_limit import limiter
    from app.db.session import reset_engine
    from app.main import create_app

    reset_settings_cache()
    reset_engine()
    limiter.reset()
    app = create_app()
    with TestClient(app) as test_client:
        headers = _auth(test_client)
        global_resp = test_client.get(
            "/api/v1/models/linear_regression/explanations/global", headers=headers
        )
        assert global_resp.status_code == 200
        body = global_resp.json()
        assert body["explainer"] == "LinearExplainer"
        assert body["ranking"][0]["name"] == "air_temperature"
        assert body["ranking"][0]["mean_abs_shap"] == 1.25
        assert body["occupancy_source"] == "simulated"
        local_resp = test_client.get(
            "/api/v1/models/linear_regression/explanations",
            params={"building_id": "fixture_office_1", "ts": "2017-01-15T04:00:00+00:00"},
            headers=headers,
        )
        assert local_resp.status_code == 200
        local = local_resp.json()
        assert local["prediction"] == 12.5
        assert local["features"][0]["name"] == "air_temperature"
        assert local["features"][0]["value"] == 18.0
        assert local["features"][0]["contribution"] == 1.1
        assert "simulated occupancy" in local["narrative"]
        missing_row = test_client.get(
            "/api/v1/models/linear_regression/explanations",
            params={"building_id": "no-such-building"},
            headers=headers,
        )
        assert missing_row.status_code == 404
    reset_engine()
    reset_settings_cache()
    limiter.reset()
