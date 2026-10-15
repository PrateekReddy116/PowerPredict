import json

from iot.simulator.publisher import run_loop


class _FakePublisher:
    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def publish(self, topic, payload, qos=1, retain=False):
        self.messages.append((topic, payload))

    def loop_start(self):
        return None

    def loop_stop(self):
        return None

    def disconnect(self):
        return None


def test_run_loop_publishes_simulated_without_broker():
    fake = _FakePublisher()
    rc = run_loop(
        settings={"topic_root": "smartbuilding"},
        building_ids=["building_001"],
        interval_s=0.01,
        ticks=1,
        client=fake,
        sleep_fn=lambda _s: None,
    )
    assert rc == 0
    assert fake.messages
    topic, raw = fake.messages[0]
    assert topic == "smartbuilding/building_001/sensors"
    body = json.loads(raw)
    assert body["data_origin"] == "simulated_iot"
    assert body["occupancy_source"] == "simulated"
    assert "energy_kwh" in body
