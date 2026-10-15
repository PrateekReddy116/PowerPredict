"""Schema validation for BDG2-shaped tables and the internal long format."""

from __future__ import annotations

import pandas as pd

METADATA_REQUIRED = ("building_id", "site_id", "primaryspaceusage", "timezone", "electricity")
WEATHER_REQUIRED = ("timestamp", "site_id")
WEATHER_OPTIONAL = (
    "airTemperature",
    "dewTemperature",
    "windSpeed",
    "precipDepth1HR",
    "precipDepth6HR",
    "seaLvlPressure",
    "cloudCoverage",
    "windDirection",
)
ELECTRICITY_WIDE_TIME = "timestamp"
LONG_REQUIRED = ("building_id", "timestamp", "energy_kwh")


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], *, name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("metadata table is empty")
    _require_columns(df, METADATA_REQUIRED, name="metadata")
    if df["building_id"].duplicated().any():
        raise ValueError("metadata has duplicate building_id values")
    return df


def validate_weather(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("weather table is empty")
    _require_columns(df, WEATHER_REQUIRED, name="weather")
    return df


def validate_electricity_wide(df: pd.DataFrame) -> pd.DataFrame:
    if ELECTRICITY_WIDE_TIME not in df.columns:
        raise ValueError("electricity wide table must have a 'timestamp' column")
    if df.shape[1] < 2:
        raise ValueError("electricity wide table has no building columns")
    if df[ELECTRICITY_WIDE_TIME].isna().all():
        raise ValueError("electricity timestamps are all missing")
    return df


def validate_energy_long(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, LONG_REQUIRED, name="energy long table")
    if df.empty:
        raise ValueError("energy long table is empty")
    return df
