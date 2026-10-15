"""Weather alignment and humidity from dew point (BDG2 has no RH column)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.constants import MAGNUS_A, MAGNUS_B

WEATHER_RENAME = {
    "airTemperature": "air_temperature",
    "dewTemperature": "dew_temperature",
    "windSpeed": "wind_speed",
    "precipDepth1HR": "precip_depth_1hr",
    "precipDepth6HR": "precip_depth_6hr",
    "seaLvlPressure": "sea_level_pressure",
    "cloudCoverage": "cloud_coverage",
    "windDirection": "wind_direction",
}

WEATHER_FEATURE_COLUMNS = (
    "air_temperature",
    "dew_temperature",
    "humidity",
    "wind_speed",
    "precip_depth_1hr",
)


def relative_humidity_percent(air_temperature_c: pd.Series, dew_temperature_c: pd.Series) -> pd.Series:
    """Magnus approximation of relative humidity (%) from T and Td in °C.

    RH = 100 * exp(γ(Td)) / exp(γ(T)) with γ(t) = a*t / (b+t), a=17.625, b=243.04.
    """
    air = pd.to_numeric(air_temperature_c, errors="coerce")
    dew = pd.to_numeric(dew_temperature_c, errors="coerce")
    gamma_dew = (MAGNUS_A * dew) / (MAGNUS_B + dew)
    gamma_air = (MAGNUS_A * air) / (MAGNUS_B + air)
    rh = 100.0 * np.exp(gamma_dew) / np.exp(gamma_air)
    return rh.clip(lower=0.0, upper=100.0)


def normalize_weather_columns(weather: pd.DataFrame) -> pd.DataFrame:
    out = weather.copy()
    out.columns = [str(col).strip() for col in out.columns]
    return out.rename(columns=WEATHER_RENAME)


def add_humidity(weather: pd.DataFrame) -> pd.DataFrame:
    out = weather.copy()
    if "air_temperature" in out.columns and "dew_temperature" in out.columns:
        out["humidity"] = relative_humidity_percent(out["air_temperature"], out["dew_temperature"])
    else:
        out["humidity"] = np.nan
    return out


def align_weather(
    energy: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    on: tuple[str, str] = ("site_id", "ts"),
) -> pd.DataFrame:
    """Left-join weather onto energy rows. Weather must already share UTC ``ts``."""
    weather_cols = [col for col in weather.columns if col not in energy.columns or col in on]
    keep = list(dict.fromkeys([*on, *weather_cols]))
    right = weather.loc[:, [col for col in keep if col in weather.columns]].drop_duplicates(list(on))
    merged = energy.merge(right, on=list(on), how="left", suffixes=("", "_weather"))
    return merged
