from ml.constants import DATA_ORIGIN_HISTORICAL
from ml.features.occupancy import DEFAULT_OCCUPANCY_SOURCE
from ml.ingest.fixture import (
    ALPHA_ID,
    BETA_ID,
    FIXTURE_LABEL,
    make_fixture_electricity_wide,
    make_fixture_metadata,
    make_fixture_weather,
)
from ml.ingest.schema import validate_electricity_wide, validate_metadata, validate_weather
from ml.pipeline import FEATURE_COLUMNS, run_pipeline


def test_fixture_is_labeled_not_bdg2():
    assert ALPHA_ID.startswith("TEST_FIXTURE_")
    assert BETA_ID.startswith("TEST_FIXTURE_")
    assert "not BDG2" in FIXTURE_LABEL
    assert make_fixture_metadata()["building_id"].str.startswith("TEST_FIXTURE_").all()


def test_schema_validation_on_fixture():
    validate_metadata(make_fixture_metadata())
    validate_electricity_wide(make_fixture_electricity_wide(periods=48))
    validate_weather(make_fixture_weather(periods=48))


def test_full_pipeline_on_tiny_fixture():
    result = run_pipeline(
        make_fixture_metadata(),
        make_fixture_electricity_wide(periods=400),
        make_fixture_weather(periods=400),
        source="test_fixture",
        n_buildings=2,
    )
    assert result.source == "test_fixture"
    assert set(result.buildings["id"]) == {ALPHA_ID, BETA_ID}
    assert result.readings["data_origin"].eq(DATA_ORIGIN_HISTORICAL).all()
    assert result.features["occupancy_source"].eq(DEFAULT_OCCUPANCY_SOURCE).all()
    for col in FEATURE_COLUMNS:
        assert col in result.features.columns
    assert "target_energy_kwh" in result.features.columns
    assert result.summary["records"]["aligned_hourly"] == 800  # 400 hours × 2 buildings
    assert result.summary["split"]["method"] == "global_unique_timestamp_fractions"
    assert len(result.train) + len(result.val) + len(result.test) == result.summary["records"]["features_after_dropna"]
    # dropna after lags removes the warmup window
    assert result.summary["records"]["features_after_dropna"] < result.summary["records"]["features_before_dropna"]
