from __future__ import annotations

import os

from fastapi.testclient import TestClient
from iot.simulator.payload import validate_mqtt_payload

from app.core.config import reset_settings_cache
from app.core.rate_limit import limiter
from app.db.session import get_session_factory, reset_engine
from app.main import create_app
from app.models.reading import SensorReading
from app.models.stubs import OccupancyObservation, WeatherObservation
from app.services.iot_ingest import persist_mqtt_reading
from app.services.mqtt_consumer import IotRuntimeStatus, reset_runtime_status, status_payload

_TEST_SECRET = "test-jwt-secret-key-32-chars-min"

VALID = {
    "building_id": "building_001",
    "timestamp": "2026-01-01T12:00:00Z",
    "temperature": 31.2,
    "humidity": 67.4,
    "occupancy": 82,
    "energy_kwh": 42.8,
    "data_origin": "simulated_iot",
    "schema_version": "1",
    "occupancy_source": "simulated",
}


def _auth(client: TestClient) -> dict[str, str]:
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "test-admin-password"},
    )
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_status_semantics_live_only_for_real_iot():
    snap = IotRuntimeStatus(broker_connected=False, last_data_origin="simulated_iot")
    assert snap.connection_status() == "OFFLINE"
    snap.broker_connected = True
    assert snap.connection_status() == "SIMULATED"
    snap.last_data_origin = "real_iot"
    assert snap.connection_status() == "LIVE"
    snap.last_data_origin = "simulated_iot"
    assert snap.connection_status() == "SIMULATED"


def test_iot_status_requires_auth(client):
    response = client.get("/api/v1/iot/status")
    assert response.status_code == 401


def test_iot_status_offline_without_broker(client):
    headers = _auth(client)
    response = client.get("/api/v1/iot/status", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "OFFLINE"
    assert body["broker_connected"] is False
    assert body["last_message_at"] is None
    assert "SIMULATED" in body["note"]
    assert "LIVE" in body["note"]


def test_ingestion_maps_to_db_fields(client):
    headers = _auth(client)
    db = get_session_factory()()
    try:
        canonical = validate_mqtt_payload(VALID)
        persist_mqtt_reading(db, canonical)
        db.commit()
        reading = db.query(SensorReading).one()
        assert reading.building_id == "building_001"
        assert reading.energy_kwh == 42.8
        assert reading.temperature_c == 31.2
        assert reading.humidity_pct == 67.4
        assert reading.occupancy == 82
        assert reading.data_origin == "simulated_iot"
        assert reading.occupancy_source == "simulated"
        assert reading.schema_version == "1"
        weather = db.query(WeatherObservation).one()
        assert weather.air_temperature_c == 31.2
        assert weather.humidity_pct == 67.4
        assert weather.data_origin == "simulated_iot"
        occ = db.query(OccupancyObservation).one()
        assert occ.occupancy == 82
        assert occ.occupancy_source == "simulated"
    finally:
        db.close()

    listed = client.get("/api/v1/readings", headers=headers)
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 1
    assert rows[0]["data_origin"] == "simulated_iot"
    assert rows[0]["energy_kwh"] == 42.8
    assert rows[0]["temperature_c"] == 31.2


def test_unknown_origin_not_persisted(client):
    db = get_session_factory()()
    try:
        payload = dict(validate_mqtt_payload(VALID))
        payload["data_origin"] = "not_a_real_origin"
        try:
            persist_mqtt_reading(db, payload)
        except ValueError as exc:
            assert "unknown MQTT origin" in str(exc)
        else:
            raise AssertionError("expected unknown origin to be rejected")
        db.rollback()
        assert db.query(SensorReading).count() == 0
    finally:
        db.close()


def test_status_offline_when_paho_connect_fails(tmp_path, monkeypatch):
    class FakeMqtt:
        CallbackAPIVersion = type("V", (), {"VERSION2": 2})
        MQTTv311 = 4

        class Client:
            def __init__(self, *args, **kwargs):
                self.on_connect = None
                self.on_disconnect = None
                self.on_message = None

            def username_pw_set(self, *args, **kwargs):
                return None

            def connect_async(self, *args, **kwargs):
                raise OSError("broker unreachable")

            def loop_start(self):
                return None

            def loop_stop(self):
                return None

            def disconnect(self):
                return None

            def subscribe(self, *args, **kwargs):
                return None

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'iot.db').as_posix()}")
    monkeypatch.setenv("JWT_SECRET", _TEST_SECRET)
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173")
    monkeypatch.setenv("SEED_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("SEED_ADMIN_PASSWORD", "test-admin-password")
    monkeypatch.setenv("MQTT_ENABLED", "true")
    monkeypatch.setenv("MQTT_HOST", "127.0.0.1")
    monkeypatch.setenv("MQTT_PORT", "1883")
    reset_settings_cache()
    reset_engine()
    reset_runtime_status()
    limiter.reset()
    monkeypatch.setattr("iot.subscriber._try_import_paho", lambda: FakeMqtt)

    app = create_app()
    with TestClient(app) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 200
        headers = _auth(test_client)
        readings = test_client.get("/api/v1/readings", headers=headers)
        assert readings.status_code == 200
        status = test_client.get("/api/v1/iot/status", headers=headers)
        assert status.status_code == 200
        body = status.json()
        assert body["status"] == "OFFLINE"
        assert body["broker_connected"] is False
        snap = status_payload()
        assert snap["status"] == "OFFLINE"

    reset_engine()
    reset_settings_cache()
    reset_runtime_status()
    limiter.reset()
    os.environ["MQTT_ENABLED"] = "false"
