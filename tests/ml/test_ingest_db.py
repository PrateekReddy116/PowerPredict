import pandas as pd
from ml.constants import DATA_ORIGIN_HISTORICAL
from ml.ingest.db import ingest_historical


def test_ingest_historical_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "phase2.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    buildings = pd.DataFrame(
        {
            "id": ["TEST_FIXTURE_education_Alpha"],
            "site_id": ["TestSite"],
            "primary_use": ["Education"],
            "sqm": [1200.0],
            "timezone": ["UTC"],
            "selected_reason": ["TEST FIXTURE, not BDG2"],
        }
    )
    readings = pd.DataFrame(
        {
            "building_id": ["TEST_FIXTURE_education_Alpha", "TEST_FIXTURE_education_Alpha"],
            "ts": pd.to_datetime(["2017-01-01 00:00:00Z", "2017-01-01 01:00:00Z"], utc=True),
            "energy_kwh": [10.0, None],
            "data_origin": [DATA_ORIGIN_HISTORICAL, DATA_ORIGIN_HISTORICAL],
        }
    )
    dry = ingest_historical(buildings, readings, database_url=url, dry_run=True)
    assert dry["mqtt_publish"] is False
    assert dry["readings_attempted"] == 1
    payload = ingest_historical(buildings, readings, database_url=url, dry_run=False)
    assert payload["status"] == "ok"
    assert payload["readings_inserted"] == 1
    assert payload["data_origin"] == DATA_ORIGIN_HISTORICAL
