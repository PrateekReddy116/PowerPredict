"""Parse BDG2 naive local timestamps and convert to UTC."""

from __future__ import annotations

import pandas as pd


def parse_local_timestamps(values: pd.Series) -> pd.Series:
    """Parse BDG2 `YYYY-MM-DD HH:MM:SS` local timestamps (naive)."""
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().all():
        raise ValueError("timestamp parsing produced all-NaT values")
    return parsed


def localize_to_utc(naive_local: pd.Series, timezone_name: str) -> tuple[pd.Series, pd.Series]:
    """Localize naive BDG2 local time, then convert to UTC.

    BDG2 meter and weather timestamps are local to the site (no offset in the CSV).
    Ambiguous DST hours use pandas ``ambiguous='infer'`` when possible; otherwise
    ``ambiguous=True``. Nonexistent times are shifted forward.
    """
    naive = parse_local_timestamps(naive_local)
    try:
        local = naive.dt.tz_localize(
            timezone_name,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    except (ValueError, TypeError):
        local = naive.dt.tz_localize(
            timezone_name,
            ambiguous=True,
            nonexistent="shift_forward",
        )
    utc = local.dt.tz_convert("UTC")
    return utc, local


def sort_chronologically(df: pd.DataFrame, *, time_col: str = "ts") -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"time column {time_col!r} is missing")
    return df.sort_values(time_col, kind="mergesort").reset_index(drop=True)


def convert_frame_local_to_utc(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    timezone_col: str = "timezone",
    group_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Add ``ts`` (UTC) and ``ts_local`` (site-local, tz-aware). Sort by group then time."""
    if timezone_col not in df.columns:
        raise ValueError("timezone column is required for BDG2 local → UTC conversion")
    frames: list[pd.DataFrame] = []
    for tz_name, group in df.groupby(timezone_col, sort=False, dropna=False):
        if pd.isna(tz_name) or str(tz_name).strip() == "":
            raise ValueError("a row is missing timezone; cannot convert BDG2 local timestamps")
        work = group.copy()
        utc, local = localize_to_utc(work[timestamp_col], str(tz_name))
        work["ts"] = utc
        work["ts_local"] = local
        frames.append(work)
    out = pd.concat(frames, ignore_index=True)
    sort_cols = [*(group_cols or ()), "ts"]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
