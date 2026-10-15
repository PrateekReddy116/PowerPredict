"""Isolation Forest feature matrix. Missing-value rule is documented and crash-free."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.anomaly.config import AnomalySettings

REQUIRED_COLUMNS = ("energy_kwh", "hour")
TEMPERATURE_ALIASES = ("air_temperature", "air_temperature_c")
ALLOWED_ORIGINS = ("historical", "simulated_iot", "real_iot")


def resolve_temperature_column(columns: pd.Index | list[str]) -> str | None:
    names = set(columns)
    for alias in TEMPERATURE_ALIASES:
        if alias in names:
            return alias
    return None


def resolve_if_columns(columns: pd.Index | list[str], settings: AnomalySettings) -> list[str]:
    """Map configured IF names onto columns that actually exist (temperature aliases)."""
    available = set(columns)
    resolved: list[str] = []
    temp_col = resolve_temperature_column(columns)
    for name in settings.feature_names:
        if name == "air_temperature":
            if temp_col and temp_col not in resolved:
                resolved.append(temp_col)
            continue
        if name == "residual":
            if "residual" in available and "residual" not in resolved:
                resolved.append("residual")
            continue
        if name in available and name not in resolved:
            resolved.append(name)
    return resolved


def ensure_meta(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "ts" not in out.columns:
        raise ValueError("feature frame must include ts")
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    if "building_id" not in out.columns:
        out["building_id"] = "_all"
    out["building_id"] = out["building_id"].astype(str)
    if "data_origin" not in out.columns:
        out["data_origin"] = "historical"
    origin = out["data_origin"].astype(str)
    unknown = ~origin.isin(ALLOWED_ORIGINS)
    if unknown.any():
        origin = origin.where(~unknown, "historical")
    out["data_origin"] = origin
    if "split" not in out.columns:
        out["split"] = ""
    if "hour" not in out.columns:
        out["hour"] = out["ts"].dt.hour.astype(float)
    else:
        out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    return out


def drop_required_missing(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows missing energy_kwh or hour. Returns (kept, n_dropped)."""
    needed = [col for col in REQUIRED_COLUMNS if col in frame.columns]
    if not needed:
        return frame.iloc[0:0].copy(), len(frame)
    mask = frame[needed].notna().all(axis=1)
    dropped = int((~mask).sum())
    return frame.loc[mask].copy(), dropped


def train_medians(train: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for col in columns:
        if col not in train.columns:
            continue
        series = pd.to_numeric(train[col], errors="coerce")
        if series.notna().any():
            medians[col] = float(series.median())
        else:
            medians[col] = 0.0
    return medians


def impute_optional(frame: pd.DataFrame, columns: list[str], medians: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        fill = medians.get(col, 0.0)
        out[col] = out[col].fillna(fill)
    return out


def group_keys(frame: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = frame.loc[:, ["building_id", "data_origin"]].drop_duplicates()
    return [(str(row.building_id), str(row.data_origin)) for row in pairs.itertuples(index=False)]


def mask_group(frame: pd.DataFrame, building_id: str, origin: str) -> pd.Series:
    return (frame["building_id"].astype(str) == str(building_id)) & (
        frame["data_origin"].astype(str) == str(origin)
    )


def as_float_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=np.float64)
    return frame.loc[:, columns].to_numpy(dtype=np.float64)


def missing_report(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for col in columns:
        if col not in frame.columns:
            report[col] = {"present": False, "n_missing": None}
            continue
        n_missing = int(pd.to_numeric(frame[col], errors="coerce").isna().sum())
        report[col] = {"present": True, "n_missing": n_missing}
    return report
