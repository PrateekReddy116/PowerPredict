"""Chronological train / val / test. Never shuffle time-series rows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.constants import TRAIN_FRAC, VAL_FRAC

DEFAULT_TRAIN_END = pd.Timestamp("2017-06-30 23:00:00", tz="UTC")
DEFAULT_VAL_END = pd.Timestamp("2017-09-30 23:00:00", tz="UTC")


def _as_utc(ts: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def chronological_split(
    df: pd.DataFrame,
    *,
    time_col: str = "ts",
    train_end: pd.Timestamp | str = DEFAULT_TRAIN_END,
    val_end: pd.Timestamp | str = DEFAULT_VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on time order only. train_end / val_end match the project default BDG2 windows."""
    if time_col not in df.columns:
        raise ValueError(f"time column {time_col!r} is missing")
    if df.empty:
        empty = df.copy()
        return empty, empty, empty

    work = df.sort_values(time_col, kind="mergesort")
    ts = pd.to_datetime(work[time_col], utc=True)
    train_cut = _as_utc(pd.Timestamp(train_end))
    val_cut = _as_utc(pd.Timestamp(val_end))
    if val_cut <= train_cut:
        raise ValueError("val_end must be after train_end")

    train = work.loc[ts <= train_cut]
    val = work.loc[(ts > train_cut) & (ts <= val_cut)]
    test = work.loc[ts > val_cut]
    return train, val, test


@dataclass(frozen=True)
class GlobalCutoffs:
    """Same cut timestamps for every building (unique-time 70% / 85%)."""

    train_end: pd.Timestamp
    val_end: pd.Timestamp
    n_unique_timestamps: int
    train_frac_actual: float
    val_frac_actual: float
    test_frac_actual: float


def global_timestamp_cutoffs(
    timestamps: pd.Series,
    *,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> GlobalCutoffs:
    """Cutoffs from sorted unique timestamps at 70% and 85% (integer rounding)."""
    unique = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).unique().sort_values()
    n = len(unique)
    if n < 3:
        raise ValueError("need at least 3 unique timestamps for a 70/15/15 split")
    n_train = max(int(round(n * train_frac)), 1)
    n_train_val = max(int(round(n * (train_frac + val_frac))), n_train + 1)
    n_train_val = min(n_train_val, n - 1)
    train_end = pd.Timestamp(unique[n_train - 1])
    val_end = pd.Timestamp(unique[n_train_val - 1])
    n_val = n_train_val - n_train
    n_test = n - n_train_val
    return GlobalCutoffs(
        train_end=train_end,
        val_end=val_end,
        n_unique_timestamps=n,
        train_frac_actual=n_train / n,
        val_frac_actual=n_val / n,
        test_frac_actual=n_test / n,
    )


def chronological_split_fractions(
    df: pd.DataFrame,
    *,
    time_col: str = "ts",
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, GlobalCutoffs]:
    """Global chronological 70/15/15 split. No shuffle. Same cuts for all buildings."""
    cuts = global_timestamp_cutoffs(df[time_col], train_frac=train_frac, val_frac=val_frac)
    train, val, test = chronological_split(
        df,
        time_col=time_col,
        train_end=cuts.train_end,
        val_end=cuts.val_end,
    )
    return train, val, test, cuts
