"""Lag features — past observations only."""

from __future__ import annotations

import pandas as pd

from ml.constants import LAGS


def add_lag_features(
    df: pd.DataFrame,
    *,
    energy_col: str = "energy_kwh",
    group_col: str = "building_id",
    time_col: str = "ts",
    lags: tuple[int, ...] = LAGS,
) -> pd.DataFrame:
    """``lag_k`` at time t is energy at t-k (same building). Never uses t+1 or later."""
    out = df.sort_values([group_col, time_col], kind="mergesort").copy()
    grouped = out.groupby(group_col, sort=False)[energy_col]
    for lag in lags:
        if lag < 1:
            raise ValueError("lags must be positive (past only)")
        out[f"lag_{lag}"] = grouped.shift(lag)
    return out
