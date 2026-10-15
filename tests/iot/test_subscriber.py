"""MQTT subscriber unit tests. No live Mosquitto required."""

from iot.subscriber import MqttSubscriber


class _FakeMqtt:
    CallbackAPIVersion = type("V", (), {"VERSION2": 2})
    MQTTv311 = 4

    class Client:
        def __init__(self, *args, **kwargs):
            self.on_connect = None
            self.on_disconnect = None
            self.on_message = None
            self.connected = False

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


def test_subscriber_offline_when_connect_fails(monkeypatch):
    events: list[tuple[str, str | None]] = []
    monkeypatch.setattr("iot.subscriber._try_import_paho", lambda: _FakeMqtt)
    sub = MqttSubscriber(
        {"host": "127.0.0.1", "port": 1883, "topic_root": "smartbuilding"},
        on_payload=lambda *_: None,
        on_status=lambda state, err: events.append((state, err)),
    )
    sub.start()
    assert events
    assert events[-1][0] == "offline"
    assert "broker unreachable" in (events[-1][1] or "")
