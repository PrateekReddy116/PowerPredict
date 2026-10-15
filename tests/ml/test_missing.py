import pandas as pd
from ml.preprocess.quality import (
    TrainIqrWinsorizer,
    TrainMedianImputer,
    clip_invalid_energy,
    interpolate_short_gaps,
)


def test_negative_energy_becomes_nan():
    df = pd.DataFrame({"energy_kwh": [1.0, -2.0, 0.0, 3.0]})
    out = clip_invalid_energy(df)
    assert out["energy_kwh"].tolist()[0] == 1.0
    assert pd.isna(out["energy_kwh"].iloc[1])
    assert out["energy_kwh"].iloc[2] == 0.0


def test_short_gap_interpolated_within_group():
    ts = pd.date_range("2017-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "building_id": ["b1"] * 5,
            "ts": ts,
            "energy_kwh": [10.0, None, None, 16.0, 18.0],
        }
    )
    out = interpolate_short_gaps(df, ["energy_kwh"], limit=3)
    assert out["energy_kwh"].iloc[1] == 12.0
    assert out["energy_kwh"].iloc[2] == 14.0


def test_median_imputer_fits_train_only():
    train = pd.DataFrame({"energy_kwh": [10.0, 20.0, None]})
    val = pd.DataFrame({"energy_kwh": [None, 100.0]})
    imputer = TrainMedianImputer.fit(train, ["energy_kwh"])
    assert imputer.medians["energy_kwh"] == 15.0
    filled = imputer.transform(val)
    assert filled["energy_kwh"].iloc[0] == 15.0


def test_iqr_winsorizer_does_not_use_test_stats():
    train = pd.DataFrame({"building_id": ["b1"] * 6, "energy_kwh": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    test = pd.DataFrame({"building_id": ["b1"], "energy_kwh": [1000.0]})
    clipper = TrainIqrWinsorizer.fit(train, k=3.0)
    out = clipper.transform(test)
    assert out["energy_kwh"].iloc[0] < 1000.0
    assert out["energy_kwh"].iloc[0] == clipper.bounds["b1"][1]
