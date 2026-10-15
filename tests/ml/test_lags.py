import pandas as pd
from ml.features.lags import add_lag_features


def test_lags_match_shifted_energy():
    ts = pd.date_range("2017-01-01", periods=10, freq="h", tz="UTC")
    energy = pd.Series(range(10), dtype=float)
    df = pd.DataFrame({"building_id": ["b1"] * 10, "ts": ts, "energy_kwh": energy})
    out = add_lag_features(df, lags=(1, 2, 3))
    assert out["lag_1"].iloc[4] == energy.iloc[3]
    assert out["lag_2"].iloc[4] == energy.iloc[2]
    assert out["lag_3"].iloc[4] == energy.iloc[1]
    assert pd.isna(out["lag_1"].iloc[0])
    assert list(out["lag_1"].iloc[1:]) == list(energy.iloc[:-1])


def test_lags_are_per_building():
    ts = pd.date_range("2017-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "building_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "ts": list(ts) + list(ts),
            "energy_kwh": [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0],
        }
    )
    out = add_lag_features(df, lags=(1,))
    b = out.loc[out["building_id"] == "b"].reset_index(drop=True)
    assert pd.isna(b["lag_1"].iloc[0])
    assert b["lag_1"].iloc[1] == 20.0
    assert b["lag_1"].iloc[2] == 21.0
