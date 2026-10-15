"""MQTT publisher for the SIMULATED building stream. Never used by historical CSV ingest."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from iot.simulator.patterns import generate_reading
from iot.simulator.payload import topic_for, validate_mqtt_payload, wire_payload

logger = logging.getLogger("iot.simulator")

SIMULATED_BANNER = """
============================================================
  SIMULATED  — PowerPredict IoT simulator
  This is NOT live physical sensor data.
  data_origin=simulated_iot   occupancy_source=simulated
============================================================
"""


def make_client(settings: dict[str, Any], *, client_id: str):
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:  # pragma: no cover - exercised when extra is missing
        raise RuntimeError("paho-mqtt is required to publish. pip install paho-mqtt") from exc

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        protocol=mqtt.MQTTv311,
    )
    username = settings.get("username")
    password = settings.get("password")
    if username:
        client.username_pw_set(username, password or "")
    return client


def connect_publisher(client: Any, settings: dict[str, Any]) -> None:
    host = settings.get("host") or "localhost"
    port = int(settings.get("port") or 1883)
    client.connect(host, port, keepalive=30)


def publish_reading(
    client: Any,
    building_id: str,
    payload: dict[str, Any],
    *,
    topic_root: str,
) -> str:
    canonical = validate_mqtt_payload(payload)
    body = wire_payload(canonical)
    topic = topic_for(building_id, root=topic_root)
    client.publish(topic, json.dumps(body), qos=1, retain=False)
    return topic


def run_loop(
    *,
    settings: dict[str, Any],
    building_ids: list[str],
    interval_s: float = 5.0,
    ticks: int | None = None,
    accelerate: bool = False,
    tick_minutes: float = 0.0,
    client_id: str = "powerpredict-simulator",
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    client: Any | None = None,
) -> int:
    """Publish simulated readings. `ticks` limits iterations (None = until interrupted).

    `--accelerate` shortens the wall interval. `tick_minutes` > 0 advances a simulated
    clock instead of wall time (still labeled simulated_iot).
    """
    logger.warning("SIMULATED IoT publisher starting for %s", ", ".join(building_ids))
    wait = 1.0 if accelerate and interval_s >= 5.0 else max(0.05, interval_s)
    root = settings.get("topic_root") or "smartbuilding"
    owned_client = client is None
    if client is None:
        client = make_client(settings, client_id=client_id)
        try:
            connect_publisher(client, settings)
        except Exception as exc:  # noqa: BLE001 — broker down is a clean CLI failure
            logger.error("Broker unreachable (%s). Historical API queries do not need MQTT.", exc)
            print(f"Broker unreachable: {exc}", flush=True)
            return 2
        client.loop_start()

    clock = (now_fn or (lambda: datetime.now(UTC)))()
    published = 0
    try:
        iteration = 0
        while ticks is None or iteration < ticks:
            for building_id in building_ids:
                payload = generate_reading(building_id, clock)
                topic = publish_reading(client, building_id, payload, topic_root=root)
                published += 1
                logger.info(
                    "SIMULATED %s %s energy_kwh=%s occupancy=%s",
                    topic,
                    payload["timestamp"],
                    payload["energy_kwh"],
                    payload["occupancy"],
                )
            iteration += 1
            if ticks is not None and iteration >= ticks:
                break
            sleep_fn(wait)
            if tick_minutes > 0:
                clock = clock + timedelta(minutes=tick_minutes)
            else:
                clock = (now_fn or (lambda: datetime.now(UTC)))()
    except KeyboardInterrupt:
        logger.warning("SIMULATED publisher stopped by user")
    finally:
        if owned_client:
            try:
                client.loop_stop()
                client.disconnect()
            except Exception:  # noqa: BLE001
                pass
    logger.warning("SIMULATED publisher finished (%s messages)", published)
    return 0
