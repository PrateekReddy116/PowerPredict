import pandas as pd
from ml.features.lags import add_lag_features
from ml.features.rolling import add_rolling_mean_features
from ml.features.splits import chronological_split_fractions
from ml.ingest.fixture import (
    FIXTURE_LABEL,
    make_fixture_electricity_wide,
    make_fixture_metadata,
    make_fixture_weather,
)
from ml.pipeline import add_target, run_pipeline


def test_lags_and_rolling_do_not_equal_future_target():
    ts = pd.date_range("2017-01-01", periods=20, freq="h", tz="UTC")
    energy = pd.Series(range(20), dtype=float)
    df = pd.DataFrame({"building_id": ["b1"] * 20, "ts": ts, "energy_kwh": energy})
    out = add_lag_features(df, lags=(1, 2, 3))
    out = add_rolling_mean_features(out, windows=(3,))
    out = add_target(out)
    mid = out.iloc[10]
    assert mid["lag_1"] == energy.iloc[9]
    assert mid["target_energy_kwh"] == energy.iloc[11]
    assert mid["lag_1"] != mid["target_energy_kwh"]
    assert mid["lag_2"] != mid["target_energy_kwh"]
    assert mid["rolling_mean_3"] != mid["target_energy_kwh"]
    # Window at t=10 uses energy 9,8,7 — not 10 or 11
    assert mid["rolling_mean_3"] == (energy.iloc[7] + energy.iloc[8] + energy.iloc[9]) / 3.0


def test_split_train_max_before_val_before_test():
    ts = pd.date_range("2017-01-01", periods=40, freq="h", tz="UTC")
    df = pd.DataFrame({"ts": ts, "energy_kwh": range(40)})
    train, val, test, _cuts = chronological_split_fractions(df)
    assert train["ts"].max() < val["ts"].min()
    assert val["ts"].max() < test["ts"].min()


def test_pipeline_fixture_has_no_future_in_features():
    result = run_pipeline(
        make_fixture_metadata(),
        make_fixture_electricity_wide(periods=400),
        make_fixture_weather(periods=400),
        source="test_fixture",
        n_buildings=2,
        winsorize=True,
    )
    assert FIXTURE_LABEL.split(",")[0] in "TEST FIXTURE"
    assert result.source == "test_fixture"
    train, val, test = result.train, result.val, result.test
    assert len(train) and len(val) and len(test)
    assert train["ts"].max() < val["ts"].min()
    assert val["ts"].max() < test["ts"].min()
    for col in ("lag_1", "lag_24", "rolling_mean_3"):
        if col in result.features.columns:
            both = result.features.dropna(subset=[col, "target_energy_kwh"])
            # Identical equality for all rows would indicate using y[t+1]
            if len(both) > 10:
                assert not (both[col] == both["target_energy_kwh"]).all()
