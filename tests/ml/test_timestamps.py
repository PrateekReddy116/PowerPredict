import pandas as pd
from ml.preprocess.timestamps import (
    convert_frame_local_to_utc,
    parse_local_timestamps,
    sort_chronologically,
)


def test_parse_bdg2_naive_local_format():
    parsed = parse_local_timestamps(pd.Series(["2017-01-01 00:00:00", "2017-01-01 01:00:00"]))
    assert parsed.dt.tz is None
    assert parsed.iloc[0] == pd.Timestamp("2017-01-01 00:00:00")


def test_eastern_local_converts_to_utc():
    df = pd.DataFrame(
        {
            "timestamp": ["2017-01-01 00:00:00"],
            "timezone": ["US/Eastern"],
            "building_id": ["b1"],
        }
    )
    out = convert_frame_local_to_utc(df, group_cols=("building_id",))
    assert str(out["ts"].dt.tz) == "UTC"
    # 2017-01-01 is EST (UTC-5)
    assert out["ts"].iloc[0] == pd.Timestamp("2017-01-01 05:00:00", tz="UTC")
    assert out["ts_local"].iloc[0].hour == 0


def test_chronological_sort_is_stable():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2017-01-01 03:00:00Z", "2017-01-01 01:00:00Z", "2017-01-01 02:00:00Z"]),
            "v": [3, 1, 2],
        }
    )
    out = sort_chronologically(df)
    assert out["v"].tolist() == [1, 2, 3]
    assert out["ts"].is_monotonic_increasing
