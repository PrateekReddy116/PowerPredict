import pandas as pd
from ml.features.splits import (
    chronological_split,
    chronological_split_fractions,
    global_timestamp_cutoffs,
)


def test_chronological_split_is_monotonic():
    ts = pd.date_range("2017-06-29", periods=8, freq="h", tz="UTC")
    df = pd.DataFrame({"ts": ts, "energy_kwh": range(8)})
    train_end = pd.Timestamp("2017-06-29 02:00:00", tz="UTC")
    val_end = pd.Timestamp("2017-06-29 04:00:00", tz="UTC")
    train, val, test = chronological_split(df, train_end=train_end, val_end=val_end)

    assert train["ts"].is_monotonic_increasing
    assert val["ts"].is_monotonic_increasing
    assert test["ts"].is_monotonic_increasing
    assert train["ts"].max() <= train_end
    assert val["ts"].min() > train_end
    assert val["ts"].max() <= val_end
    assert test["ts"].min() > val_end
    assert len(train) + len(val) + len(test) == len(df)


def test_global_70_15_15_no_shuffle():
    ts = pd.date_range("2017-01-01", periods=100, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "ts": list(ts) + list(ts),
            "building_id": ["a"] * 100 + ["b"] * 100,
            "energy_kwh": list(range(100)) + list(range(100, 200)),
        }
    )
    shuffled_index = list(range(199, -1, -1))
    df = df.iloc[shuffled_index].reset_index(drop=True)
    train, val, test, cuts = chronological_split_fractions(df)
    assert abs(cuts.train_frac_actual - 0.70) < 0.02
    assert abs(cuts.val_frac_actual - 0.15) < 0.02
    assert abs(cuts.test_frac_actual - 0.15) < 0.02
    assert train["ts"].max() < val["ts"].min()
    assert val["ts"].max() < test["ts"].min()
    assert train["ts"].is_monotonic_increasing
    assert val["ts"].is_monotonic_increasing
    assert test["ts"].is_monotonic_increasing
    assert len(train) + len(val) + len(test) == len(df)
    # Same global cuts for both buildings
    assert train["ts"].max() == cuts.train_end
    assert val["ts"].max() == cuts.val_end


def test_cutoff_indices_are_monotonic():
    ts = pd.date_range("2016-01-01", periods=200, freq="h", tz="UTC")
    cuts = global_timestamp_cutoffs(pd.Series(ts))
    assert cuts.train_end < cuts.val_end
    unique = pd.DatetimeIndex(ts)
    assert unique.get_loc(cuts.train_end) < unique.get_loc(cuts.val_end)
