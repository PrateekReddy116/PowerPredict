"""FastAPI MQTT consumer. Fail-soft: broker down does not take the API down."""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.core.config import Settings, get_settings
from app.db.session import get_session_factory
from app.services.iot_ingest import persist_mqtt_reading

logger = logging.getLogger("app.mqtt")

ConnectionStatus = Literal["LIVE", "SIMULATED", "OFFLINE"]


def _ensure_repo_on_path() -> Path:
    repo = Path(__file__).resolve().parents[3]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


@dataclass
class IotRuntimeStatus:
    broker_connected: bool = False
    last_message_at: datetime | None = None
    last_data_origin: str | None = None
    last_building_id: str | None = None
    last_error: str | None = None
    topic: str = "smartbuilding/+/sensors"

    def connection_status(self) -> ConnectionStatus:
        if not self.broker_connected:
            return "OFFLINE"
        if self.last_data_origin == "real_iot":
            return "LIVE"
        # Connected to the demo broker path. Simulated traffic is never labeled LIVE.
        return "SIMULATED"


_lock = threading.Lock()
_status = IotRuntimeStatus()
_subscriber: Any = None


def get_runtime_status() -> IotRuntimeStatus:
    with _lock:
        return IotRuntimeStatus(
            broker_connected=_status.broker_connected,
            last_message_at=_status.last_message_at,
            last_data_origin=_status.last_data_origin,
            last_building_id=_status.last_building_id,
            last_error=_status.last_error,
            topic=_status.topic,
        )


def status_payload(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    snap = get_runtime_status()
    note = (
        "OFFLINE means the broker is unreachable; historical queries still work. "
        "SIMULATED is the honest demo state (simulator / simulated_iot). "
        "LIVE is reserved for real_iot traffic and is never used for the simulator."
    )
    return {
        "status": snap.connection_status(),
        "broker_connected": snap.broker_connected,
        "last_message_at": snap.last_message_at,
        "last_data_origin": snap.last_data_origin,
        "last_building_id": snap.last_building_id,
        "topic": snap.topic or f"{settings.mqtt_topic_root}/+/sensors",
        "mqtt_enabled": settings.mqtt_enabled,
        "note": note,
    }


def _on_status(state: str, error: str | None) -> None:
    with _lock:
        _status.broker_connected = state == "connected"
        _status.last_error = error


def _on_payload(payload: dict[str, Any], _topic: str) -> None:
    origin = str(payload.get("data_origin") or "")
    if origin not in {"simulated_iot", "real_iot"}:
        logger.warning("Dropping MQTT message with unknown origin %r", origin)
        return
    factory = get_session_factory()
    db = factory()
    try:
        persist_mqtt_reading(db, payload)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to persist MQTT reading")
        return
    finally:
        db.close()
    with _lock:
        _status.last_message_at = datetime.now(UTC)
        _status.last_data_origin = origin
        _status.last_building_id = str(payload.get("building_id") or "")


def start_mqtt_consumer(settings: Settings | None = None) -> None:
    global _subscriber
    settings = settings or get_settings()
    with _lock:
        _status.topic = f"{settings.mqtt_topic_root}/+/sensors"
        _status.broker_connected = False
        _status.last_error = None
    if not settings.mqtt_enabled:
        logger.info("MQTT consumer disabled (MQTT_ENABLED=false); status OFFLINE.")
        return
    _ensure_repo_on_path()
    try:
        from iot.subscriber import MqttSubscriber
    except Exception as exc:  # noqa: BLE001
        logger.warning("MQTT subscriber import failed (%s); API continues.", exc)
        return

    mqtt_cfg = {
        "host": settings.mqtt_host,
        "port": settings.mqtt_port,
        "username": settings.mqtt_username,
        "password": settings.mqtt_password,
        "topic_root": settings.mqtt_topic_root,
    }
    sub = MqttSubscriber(
        mqtt_cfg,
        on_payload=_on_payload,
        on_status=_on_status,
        client_id=settings.mqtt_client_id,
    )
    _subscriber = sub
    try:
        sub.start()
    except Exception as exc:  # noqa: BLE001 — never crash FastAPI over MQTT
        logger.warning("MQTT consumer failed to start (%s); historical queries still work.", exc)
        with _lock:
            _status.broker_connected = False
            _status.last_error = str(exc)


def stop_mqtt_consumer() -> None:
    global _subscriber
    sub = _subscriber
    _subscriber = None
    if sub is None:
        return
    try:
        sub.stop()
    except Exception:  # noqa: BLE001
        pass
    with _lock:
        _status.broker_connected = False


def reset_runtime_status() -> None:
    """Test helper."""
    global _subscriber
    _subscriber = None
    with _lock:
        _status.broker_connected = False
        _status.last_message_at = None
        _status.last_data_origin = None
        _status.last_building_id = None
        _status.last_error = None
        _status.topic = "smartbuilding/+/sensors"
