from datetime import UTC, datetime

import pytest
from iot.simulator.payload import (
    PayloadError,
    decode_mqtt_payload,
    topic_for,
    validate_mqtt_payload,
)

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


def test_payload_missing_data_origin_rejected():
    body = dict(VALID)
    del body["data_origin"]
    with pytest.raises(PayloadError, match="data_origin is required"):
        validate_mqtt_payload(body)


def test_decode_missing_data_origin_dropped():
    body = dict(VALID)
    del body["data_origin"]
    assert decode_mqtt_payload(body) is None


def test_payload_historical_rejected_on_mqtt():
    body = dict(VALID)
    body["data_origin"] = "historical"
    with pytest.raises(PayloadError, match="simulated_iot or real_iot"):
        validate_mqtt_payload(body)


def test_payload_simulated_accepted():
    out = validate_mqtt_payload(VALID)
    assert out["data_origin"] == "simulated_iot"
    assert out["occupancy_source"] == "simulated"
    assert out["timestamp"] == "2026-01-01T12:00:00Z"
    assert out["ts"] == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert out["energy_kwh"] == 42.8
    assert out["temperature"] == 31.2


def test_payload_real_iot_accepted():
    body = dict(VALID)
    body["data_origin"] = "real_iot"
    body["occupancy_source"] = "measured"
    assert validate_mqtt_payload(body)["data_origin"] == "real_iot"


def test_topic_contract():
    assert topic_for("building_001") == "smartbuilding/building_001/sensors"


def test_topic_mismatch_dropped():
    assert decode_mqtt_payload(VALID, topic="smartbuilding/other/sensors") is None


def test_valid_payload_on_matching_topic_accepted():
    out = decode_mqtt_payload(VALID, topic="smartbuilding/building_001/sensors")
    assert out is not None
    assert out["building_id"] == "building_001"


def test_ts_alias_accepted():
    body = dict(VALID)
    body["ts"] = body.pop("timestamp")
    out = validate_mqtt_payload(body)
    assert out["timestamp"].endswith("Z")
