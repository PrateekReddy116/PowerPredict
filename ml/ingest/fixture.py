"""Tiny synthetic series labeled TEST FIXTURE — not BDG2. For tests and download fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd

FIXTURE_LABEL = "TEST FIXTURE, not BDG2"
ALPHA_ID = "TEST_FIXTURE_education_Alpha"
BETA_ID = "TEST_FIXTURE_office_Beta"
SITE_ID = "TestSite"


def make_fixture_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "building_id": [ALPHA_ID, BETA_ID],
            "site_id": [SITE_ID, SITE_ID],
            "primaryspaceusage": ["Education", "Office"],
            "sqm": [1200.0, 800.0],
            "timezone": ["UTC", "UTC"],
            "electricity": ["Yes", "Yes"],
            "yearbuilt": [2001.0, 1998.0],
            "numberoffloors": [3, 2],
        }
    )


def make_fixture_electricity_wide(*, periods: int = 400, seed: int = 0) -> pd.DataFrame:
    """Hourly synthetic energy with a few NaNs, one negative, one spike — not BDG2."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2017-01-01", periods=periods, freq="h")
    hours = np.arange(periods)
    diurnal = 10 + 4 * np.sin(2 * np.pi * (hours % 24) / 24.0)
    alpha = diurnal + 2.0 + 0.15 * rng.normal(size=periods)
    beta = diurnal + 8.0 + 0.20 * rng.normal(size=periods)
    if periods > 20:
        alpha[20] = np.nan
    if periods > 40:
        beta[40] = np.nan
    if periods > 50:
        alpha[50] = -3.0
    if periods > 80:
        beta[80] = 500.0
    return pd.DataFrame(
        {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            ALPHA_ID: alpha,
            BETA_ID: beta,
        }
    )


def make_fixture_weather(*, periods: int = 400) -> pd.DataFrame:
    ts = pd.date_range("2017-01-01", periods=periods, freq="h")
    hours = np.arange(periods)
    air = 10 + 8 * np.sin(2 * np.pi * (hours % 24) / 24.0)
    dew = air - 4.0
    weather = pd.DataFrame(
        {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "site_id": SITE_ID,
            "airTemperature": air,
            "dewTemperature": dew,
            "windSpeed": 2.0 + 0.5 * np.sin(hours / 10.0),
            "precipDepth1HR": 0.0,
        }
    )
    weather.loc[15, "airTemperature"] = np.nan
    return weather
