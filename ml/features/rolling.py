"""Rolling means — shift before rolling so the window cannot include the future."""

from __future__ import annotations

import pandas as pd

from ml.constants import ROLLING_WINDOWS

# Current timestamp t is NOT included. rolling_mean_w at t uses energy[t-1] ... energy[t-w].
ROLLING_INCLUDES_CURRENT_T = False


def add_rolling_mean_features(
    df: pd.DataFrame,
    *,
    energy_col: str = "energy_kwh",
    group_col: str = "building_id",
    time_col: str = "ts",
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Past-only rolling means.

    Implementation: ``shift(1)`` then ``rolling(w)`` per building.
    Window at t therefore uses only t-1 .. t-w (current t excluded, t+1 never included).
    """
    out = df.sort_values([group_col, time_col], kind="mergesort").copy()
    if ROLLING_INCLUDES_CURRENT_T:
        raise RuntimeError("rolling must not include current t in this pipeline")

    for window in windows:
        if window < 1:
            raise ValueError("rolling windows must be positive")
        out[f"rolling_mean_{window}"] = out.groupby(group_col, sort=False)[energy_col].transform(
            lambda series, w=window: series.shift(1).rolling(window=w, min_periods=w).mean()
        )
    # Extra: 24h rolling std (past-only) for the training-matrix contract.
    out["rolling_std_24"] = out.groupby(group_col, sort=False)[energy_col].transform(
        lambda series: series.shift(1).rolling(window=24, min_periods=24).std()
    )
    return out
