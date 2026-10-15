from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ml.constants import LAGS, ROLLING_WINDOWS


def make_fixture_frame(n: int = 400, *, seed: int = 42) -> pd.DataFrame:
    """Synthetic hourly frame using Phase 2 column names. Not BDG2; occupancy is simulated."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2017-01-01", periods=n, freq="h", tz="UTC")
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    month = ts.month.to_numpy()
    day = ts.day.to_numpy()
    diurnal = 40.0 + 15.0 * np.sin(2 * np.pi * hour / 24.0)
    weekly = 5.0 * (dow < 5).astype(float)
    energy = np.clip(diurnal + weekly + rng.normal(0.0, 1.5, n), 8.0, None)
    occupancy = np.where(
        (dow < 5) & (hour >= 8) & (hour <= 18),
        0.6 + 0.2 * rng.random(n),
        0.05 + 0.05 * rng.random(n),
    )
    air = 12.0 + 6.0 * np.sin(2 * np.pi * hour / 24.0) + rng.normal(0, 0.5, n)
    dew = air - 4.0
    df = pd.DataFrame(
        {
            "ts": ts,
            "building_id": "fixture_office_1",
            "energy_kwh": energy,
            "sqm": 800.0,
            "hour": hour.astype(float),
            "day": day.astype(float),
            "day_of_week": dow.astype(float),
            "month": month.astype(float),
            "weekend": (dow >= 5).astype(float),
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "air_temperature": air,
            "dew_temperature": dew,
            "humidity": np.clip(60.0 + rng.normal(0, 3, n), 0.0, 100.0),
            "wind_speed": np.clip(3.0 + rng.normal(0, 0.8, n), 0.0, None),
            "precip_depth_1hr": 0.0,
            "occupancy": occupancy,
            "occupancy_source": "simulated",
            "data_origin": "historical",
        }
    )
    for lag in LAGS:
        df[f"lag_{lag}"] = df["energy_kwh"].shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = df["energy_kwh"].shift(1).rolling(window, min_periods=window).mean()
    df["rolling_std_24"] = df["energy_kwh"].shift(1).rolling(24, min_periods=24).std()
    df["target_energy_kwh"] = df["energy_kwh"].shift(-1)
    return df


@pytest.fixture
def fixture_frame() -> pd.DataFrame:
    return make_fixture_frame()
