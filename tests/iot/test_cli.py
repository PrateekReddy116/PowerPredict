from iot.simulator.__main__ import build_parser, main


def test_cli_help_mentions_simulated_and_interval():
    help_text = build_parser().format_help()
    assert "--interval" in help_text
    assert "--accelerate" in help_text
    assert "--dry-run" in help_text
    assert "simulated" in help_text.lower()


def test_dry_run_prints_banner_and_payload(capsys, monkeypatch):
    monkeypatch.setenv("MQTT_TOPIC_PREFIX", "smartbuilding")
    rc = main(["--dry-run", "--buildings", "building_001"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SIMULATED" in out
    assert "simulated_iot" in out
    assert "smartbuilding/building_001/sensors" in out
    assert "occupancy_source" in out
