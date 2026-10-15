"""Calendar features from **site-local** time (schedules are local, storage is UTC)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame, *, local_col: str = "ts_local") -> pd.DataFrame:
    """hour, day, day_of_week, month, weekend, plus optional cyclic encodings."""
    if local_col not in df.columns:
        raise ValueError(f"{local_col} is required for local-time temporal features")
    out = df.copy()
    local = pd.to_datetime(out[local_col])
    out["hour"] = local.dt.hour.astype("int16")
    out["day"] = local.dt.day.astype("int16")
    out["day_of_week"] = local.dt.dayofweek.astype("int16")
    out["month"] = local.dt.month.astype("int16")
    out["weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    return out
