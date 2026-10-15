"""MQTT reading payload contract.

Historical BDG2 rows are ingest-only (`data_origin=historical`) and must never be published here.
MQTT payloads require data_origin of simulated_iot or real_iot. Missing data_origin is dropped.

User topic (wins over older prefixes): smartbuilding/{building_id}/sensors
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

MQTT_ORIGINS = frozenset({"simulated_iot", "real_iot"})
ALL_ORIGINS = frozenset({"historical", "simulated_iot", "real_iot"})
OCCUPANCY_SOURCES = frozenset({"simulated", "measured"})
SCHEMA_VERSION = "1"
DEFAULT_TOPIC_ROOT = "smartbuilding"
TOPIC_LEAF = "sensors"
REQUIRED_NUMERIC = ("temperature", "humidity", "occupancy", "energy_kwh")

logger = logging.getLogger("iot")


class PayloadError(ValueError):
    pass


def topic_root(root: str | None = None) -> str:
    value = (root or DEFAULT_TOPIC_ROOT).strip().strip("/")
    return value or DEFAULT_TOPIC_ROOT


def topic_for(building_id: str, *, root: str | None = None) -> str:
    bid = str(building_id).strip()
    if not bid:
        raise PayloadError("building_id is required")
    return f"{topic_root(root)}/{bid}/{TOPIC_LEAF}"


def topic_filter(root: str | None = None) -> str:
    return f"{topic_root(root)}/+/{TOPIC_LEAF}"


def building_id_from_topic(topic: str, *, root: str | None = None) -> str | None:
    parts = [p for p in str(topic).split("/") if p]
    expected_root = topic_root(root)
    if len(parts) != 3:
        return None
    if parts[0] != expected_root or parts[2] != TOPIC_LEAF:
        return None
    return parts[1]


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            ts = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise PayloadError("timestamp must be ISO-8601") from exc
    else:
        raise PayloadError("timestamp must be ISO-8601")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _require_number(payload: dict[str, Any], key: str) -> float:
    if key not in payload or payload[key] is None or payload[key] == "":
        raise PayloadError(f"{key} is required")
    try:
        return float(payload[key])
    except (TypeError, ValueError) as exc:
        raise PayloadError(f"{key} must be numeric") from exc


def validate_mqtt_payload(payload: Any) -> dict[str, Any]:
    """Return a canonical payload. Raises PayloadError if the message must be dropped."""
    if not isinstance(payload, dict):
        raise PayloadError("payload must be a JSON object")
    if "data_origin" not in payload or payload["data_origin"] in (None, ""):
        raise PayloadError("data_origin is required")
    origin = str(payload["data_origin"])
    if origin not in MQTT_ORIGINS:
        raise PayloadError("MQTT data_origin must be simulated_iot or real_iot")
    if "building_id" not in payload or not payload["building_id"]:
        raise PayloadError("building_id is required")

    stamp = payload.get("timestamp", payload.get("ts"))
    if stamp in (None, ""):
        raise PayloadError("timestamp is required")
    ts = parse_timestamp(stamp)

    for key in REQUIRED_NUMERIC:
        _require_number(payload, key)

    schema_version = payload.get("schema_version")
    if schema_version in (None, ""):
        raise PayloadError("schema_version is required")

    occupancy_source = payload.get("occupancy_source")
    if occupancy_source in (None, ""):
        raise PayloadError("occupancy_source is required")
    occupancy_source = str(occupancy_source)
    if occupancy_source not in OCCUPANCY_SOURCES:
        raise PayloadError("occupancy_source must be simulated or measured")
    if origin == "simulated_iot" and occupancy_source != "simulated":
        raise PayloadError("simulated_iot occupancy_source must be simulated")

    iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    canonical = dict(payload)
    canonical.update(
        {
            "building_id": str(payload["building_id"]),
            "timestamp": iso,
            "ts": ts,
            "temperature": float(payload["temperature"]),
            "humidity": float(payload["humidity"]),
            "occupancy": float(payload["occupancy"]),
            "energy_kwh": float(payload["energy_kwh"]),
            "data_origin": origin,
            "schema_version": str(schema_version),
            "occupancy_source": occupancy_source,
        }
    )
    return canonical


def decode_mqtt_payload(
    raw: bytes | str | dict[str, Any],
    *,
    topic: str | None = None,
    root: str | None = None,
) -> dict[str, Any] | None:
    """Validate a broker message. Returns None (and logs) when the payload must be dropped."""
    try:
        if isinstance(raw, dict):
            body: Any = raw
        else:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            body = json.loads(text)
        canonical = validate_mqtt_payload(body)
    except (PayloadError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Dropping MQTT payload: %s", exc)
        return None

    if topic:
        topic_bid = building_id_from_topic(topic, root=root)
        if topic_bid is None:
            logger.warning("Dropping MQTT payload: unexpected topic %s", topic)
            return None
        if topic_bid != canonical["building_id"]:
            logger.warning(
                "Dropping MQTT payload: topic building_id %s != payload %s",
                topic_bid,
                canonical["building_id"],
            )
            return None
    return canonical


def wire_payload(canonical: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable MQTT body (no datetime objects)."""
    ts = canonical["ts"]
    if not isinstance(ts, datetime):
        ts = parse_timestamp(ts)
    return {
        "building_id": canonical["building_id"],
        "timestamp": ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "temperature": round(float(canonical["temperature"]), 2),
        "humidity": round(float(canonical["humidity"]), 2),
        "occupancy": round(float(canonical["occupancy"]), 1),
        "energy_kwh": round(float(canonical["energy_kwh"]), 3),
        "data_origin": canonical["data_origin"],
        "schema_version": str(canonical["schema_version"]),
        "occupancy_source": canonical["occupancy_source"],
    }
