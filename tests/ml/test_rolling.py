import pandas as pd
from ml.features.rolling import ROLLING_INCLUDES_CURRENT_T, add_rolling_mean_features


def test_rolling_uses_only_allowed_past():
    assert ROLLING_INCLUDES_CURRENT_T is False
    ts = pd.date_range("2017-01-01", periods=8, freq="h", tz="UTC")
    energy = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    df = pd.DataFrame({"building_id": ["b1"] * 8, "ts": ts, "energy_kwh": energy})
    out = add_rolling_mean_features(df, windows=(3,))
    # rolling_mean_3 at t uses energy[t-1], [t-2], [t-3] after shift(1)
    assert pd.isna(out["rolling_mean_3"].iloc[2])
    expected = (energy.iloc[0] + energy.iloc[1] + energy.iloc[2]) / 3.0
    assert out["rolling_mean_3"].iloc[3] == expected
    # current t and future t+1 are not in the window
    assert out["rolling_mean_3"].iloc[3] != energy.iloc[3]
    assert out["rolling_mean_3"].iloc[3] != energy.iloc[4]


def test_rolling_does_not_include_target_hour():
    ts = pd.date_range("2017-01-01", periods=6, freq="h", tz="UTC")
    energy = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    df = pd.DataFrame({"building_id": ["b1"] * 6, "ts": ts, "energy_kwh": energy})
    out = add_rolling_mean_features(df, windows=(3,))
    target = energy.shift(-1)
    # For strictly increasing series, past-only mean cannot equal y_{t+1}
    comparable = out["rolling_mean_3"].notna()
    assert not (out.loc[comparable, "rolling_mean_3"] == target[comparable]).any()
