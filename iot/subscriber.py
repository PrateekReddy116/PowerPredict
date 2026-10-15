"""MQTT subscriber used by the FastAPI lifespan. Does not write the database.

If the broker is down, start() logs and returns; the API keeps serving historical queries.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from iot.simulator.payload import decode_mqtt_payload, topic_filter

logger = logging.getLogger("iot.subscriber")

ValidPayloadHandler = Callable[[dict[str, Any], str], None]
StatusHandler = Callable[[str, str | None], None]


def _try_import_paho():
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        logger.warning("paho-mqtt is not installed; MQTT consumer stays OFFLINE.")
        return None
    return mqtt


def _connect_failed(reason_code: Any) -> bool:
    if hasattr(reason_code, "is_failure"):
        return bool(reason_code.is_failure)
    try:
        return int(reason_code) != 0
    except (TypeError, ValueError):
        return True


class MqttSubscriber:
    """Background paho client. Fail-soft: never raise out of start()."""

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        on_payload: ValidPayloadHandler,
        on_status: StatusHandler | None = None,
        client_id: str = "powerpredict-api",
    ) -> None:
        self.settings = settings
        self.on_payload = on_payload
        self.on_status = on_status
        self.client_id = client_id
        self.client: Any | None = None
        self._started = False

    def _emit(self, state: str, error: str | None = None) -> None:
        if self.on_status is not None:
            self.on_status(state, error)

    def start(self) -> None:
        mqtt = _try_import_paho()
        if mqtt is None:
            self._emit("offline", "paho-mqtt missing")
            return
        root = self.settings.get("topic_root") or "smartbuilding"
        filt = topic_filter(root)
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
            protocol=mqtt.MQTTv311,
        )
        username = self.settings.get("username")
        password = self.settings.get("password")
        if username:
            client.username_pw_set(username, password or "")

        def on_connect(client, _userdata, _flags, reason_code, _properties=None):
            if _connect_failed(reason_code):
                logger.warning("MQTT connect failed (%s); API continues without broker.", reason_code)
                self._emit("offline", str(reason_code))
                return
            client.subscribe(filt, qos=1)
            logger.info("MQTT subscribed to %s", filt)
            self._emit("connected", None)

        def on_disconnect(client, _userdata, _flags, reason_code, _properties=None):
            logger.warning("MQTT disconnected (%s)", reason_code)
            self._emit("offline", str(reason_code))

        def on_message(_client, _userdata, msg):
            canonical = decode_mqtt_payload(msg.payload, topic=msg.topic, root=root)
            if canonical is None:
                return
            try:
                self.on_payload(canonical, msg.topic)
            except Exception:
                logger.exception("MQTT persist handler failed")

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message
        self.client = client
        host = self.settings.get("host") or "localhost"
        port = int(self.settings.get("port") or 1883)
        try:
            client.connect_async(host, port, keepalive=30)
            client.loop_start()
            self._started = True
            logger.info("MQTT consumer connecting to %s:%s (fail-soft)", host, port)
        except Exception as exc:  # noqa: BLE001 — broker down must not take down the API
            logger.warning("MQTT broker unreachable (%s); historical queries still work.", exc)
            self._emit("offline", str(exc))

    def stop(self) -> None:
        if self.client is None:
            return
        try:
            if self._started:
                self.client.loop_stop()
                self.client.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self._started = False
        self._emit("offline", None)
