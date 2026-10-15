from datetime import UTC, datetime, timedelta

from iot.simulator.buildings import DEMO_BUILDING_IDS, resolve_building_ids
from iot.simulator.patterns import energy_kwh, generate_reading, occupancy_percent


def test_weekday_occupancy_and_energy_higher_than_night():
    day = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)  # Monday
    night = datetime(2026, 8, 17, 2, 0, tzinfo=UTC)
    bid = "building_001"
    assert occupancy_percent(bid, day) > occupancy_percent(bid, night)
    day_e = energy_kwh(bid, day, occupancy_percent(bid, day), 24.0)
    night_e = energy_kwh(bid, night, occupancy_percent(bid, night), 18.0)
    assert day_e > night_e


def test_weekend_occupancy_lower_than_weekday():
    weekday = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)  # Monday
    weekend = weekday + timedelta(days=5)  # Saturday
    bid = "building_002"
    assert occupancy_percent(bid, weekday) > occupancy_percent(bid, weekend)


def test_generated_payload_is_simulated():
    body = generate_reading("building_001")
    assert body["data_origin"] == "simulated_iot"
    assert body["occupancy_source"] == "simulated"
    assert "energy_kwh" in body
    assert body["schema_version"] == "1"


def test_demo_ids_used_when_no_pipeline(tmp_path):
    ids = resolve_building_ids(repo_root=tmp_path)
    assert ids == list(DEMO_BUILDING_IDS)


def test_pipeline_summary_wins(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "pipeline_summary.json").write_text(
        '{"building_ids": ["Hog_education_Casandra"]}',
        encoding="utf-8",
    )
    assert resolve_building_ids(repo_root=tmp_path) == ["Hog_education_Casandra"]
