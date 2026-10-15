"""Load BDG2-shaped CSVs (or TEST FIXTURE files) into pandas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.ingest.schema import (
    validate_electricity_wide,
    validate_energy_long,
    validate_metadata,
    validate_weather,
)

METADATA_KEEP = (
    "building_id",
    "site_id",
    "primaryspaceusage",
    "primary_use",
    "sqm",
    "timezone",
    "electricity",
    "yearbuilt",
    "numberoffloors",
)


def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Accept raw BDG2 column names or the internal `primary_use` rename."""
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    if "primary_use" not in out.columns and "primaryspaceusage" in out.columns:
        out = out.rename(columns={"primaryspaceusage": "primary_use"})
    if "building_id" in out.columns:
        out["building_id"] = out["building_id"].astype(str).str.strip()
    if "site_id" in out.columns:
        out["site_id"] = out["site_id"].astype(str).str.strip()
    return out


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    validate_metadata(df)
    keep = [col for col in METADATA_KEEP if col in df.columns]
    out = normalize_metadata(df.loc[:, keep])
    return out


def load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    validate_weather(df)
    df["site_id"] = df["site_id"].astype(str).str.strip()
    return df


def load_electricity_wide(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=usecols)
    df.columns = [str(col).strip() for col in df.columns]
    validate_electricity_wide(df)
    return df


def melt_electricity(wide: pd.DataFrame) -> pd.DataFrame:
    """Wide BDG2 meters (timestamp + one column per building) → long table."""
    validate_electricity_wide(wide)
    building_cols = [col for col in wide.columns if col != "timestamp"]
    long = wide.melt(id_vars=["timestamp"], value_vars=building_cols, var_name="building_id", value_name="energy_kwh")
    long["building_id"] = long["building_id"].astype(str).str.strip()
    long["energy_kwh"] = pd.to_numeric(long["energy_kwh"], errors="coerce")
    return validate_energy_long(long)


def electricity_completeness(wide: pd.DataFrame) -> pd.Series:
    """Fraction of non-null hourly readings per building column."""
    validate_electricity_wide(wide)
    building_cols = [col for col in wide.columns if col != "timestamp"]
    n = len(wide)
    if n == 0:
        return pd.Series(dtype=float)
    return wide[building_cols].notna().sum(axis=0) / n
