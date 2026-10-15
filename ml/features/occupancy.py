"""Occupancy is not in BDG2. Default source is simulated — never present it as measured."""

from __future__ import annotations

import pandas as pd

from ml.constants import OCCUPANCY_SOURCE_SIMULATED

DEFAULT_OCCUPANCY_SOURCE = OCCUPANCY_SOURCE_SIMULATED


def occupancy_source_label(source: str | None = None) -> str:
    if source is None or source.strip() == "":
        return DEFAULT_OCCUPANCY_SOURCE
    return source.strip()


def occupancy_profile(primary_use: str | None, hour: int, is_weekend: bool) -> float:
    """Deterministic 0–1 occupancy from primary_use + hour (office/education schedules)."""
    use = (primary_use or "").strip().lower()
    if is_weekend:
        return 0.08 if "education" in use else 0.04
    if "office" in use:
        if 8 <= hour < 18:
            return 0.90
        if hour in (7, 18, 19):
            return 0.35
        return 0.05
    if "education" in use:
        if 8 <= hour < 17:
            return 0.85
        if hour in (7, 17, 18):
            return 0.40
        return 0.08
    if 8 <= hour < 18:
        return 0.70
    return 0.10


def add_simulated_occupancy(
    df: pd.DataFrame,
    *,
    primary_use_col: str = "primary_use",
    hour_col: str = "hour",
    weekend_col: str = "weekend",
) -> pd.DataFrame:
    """Add occupancy in [0, 1] and occupancy_source='simulated' on every row."""
    out = df.copy()
    hours = out[hour_col].astype(int)
    weekends = out[weekend_col].astype(bool)
    uses = out[primary_use_col] if primary_use_col in out.columns else pd.Series([None] * len(out))
    out["occupancy"] = [
        occupancy_profile(use, int(hour), bool(weekend))
        for use, hour, weekend in zip(uses, hours, weekends, strict=True)
    ]
    out["occupancy_source"] = DEFAULT_OCCUPANCY_SOURCE
    return out
