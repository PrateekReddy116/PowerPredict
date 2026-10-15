"""Consume a Phase 2 feature matrix. Do not rebuild lags/calendar/weather here.

Phase 2 writes `target_energy_kwh` (y_{t+1}) and `FEATURE_COLUMNS` from `ml.pipeline`.
If some intended names are absent, remaining numeric columns (minus meta/target) are used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.features.splits import (
    DEFAULT_TRAIN_END,
    DEFAULT_VAL_END,
    chronological_split,
    chronological_split_fractions,
)
from ml.pipeline import FEATURE_COLUMNS, MODEL_DROPNA_COLUMNS, add_target

LOGGER = logging.getLogger(__name__)

TARGET_COL = "target_energy_kwh"
META_COLUMNS = (
    "ts",
    "ts_local",
    "building_id",
    "site_id",
    "primary_use",
    "timezone",
    "occupancy_source",
    "data_origin",
    "meter_type",
    "split",
    "y_next",
    TARGET_COL,
)


@dataclass
class WindowMeta:
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    val_start: pd.Timestamp | None
    val_end: pd.Timestamp | None
    test_start: pd.Timestamp | None
    test_end: pd.Timestamp | None
    split_mode: str
    n_train: int
    n_val: int
    n_test: int

    def as_dict(self) -> dict[str, Any]:
        def _iso(ts: pd.Timestamp | None) -> str | None:
            if ts is None or pd.isna(ts):
                return None
            stamp = pd.Timestamp(ts)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            else:
                stamp = stamp.tz_convert("UTC")
            return stamp.isoformat()

        return {
            "train_start": _iso(self.train_start),
            "train_end": _iso(self.train_end),
            "val_start": _iso(self.val_start),
            "val_end": _iso(self.val_end),
            "test_start": _iso(self.test_start),
            "test_end": _iso(self.test_end),
            "split_mode": self.split_mode,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
        }


@dataclass
class PreparedSplits:
    """Chronological arrays. X is unscaled; scaler is fit on train only."""

    feature_names: list[str]
    scaler: StandardScaler
    X_train: np.ndarray
    y_train: np.ndarray
    ts_train: np.ndarray
    groups_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    ts_val: np.ndarray
    groups_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    ts_test: np.ndarray
    groups_test: np.ndarray
    windows: WindowMeta
    building_id: str | None


def ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL in df.columns:
        return df
    if "y_next" in df.columns:
        out = df.copy()
        out[TARGET_COL] = out["y_next"]
        return out
    return add_target(df)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    intended = [col for col in FEATURE_COLUMNS if col in df.columns]
    if intended:
        return intended
    skip = set(META_COLUMNS)
    numeric = [
        col
        for col in df.columns
        if col not in skip and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric:
        raise ValueError("no numeric feature columns found in frame")
    return numeric


def drop_incomplete(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    needed = list(dict.fromkeys([*feature_cols, TARGET_COL, *MODEL_DROPNA_COLUMNS]))
    cols = [col for col in needed if col in df.columns]
    sort_cols = [col for col in ("building_id", "ts") if col in df.columns]
    work = df.dropna(subset=cols)
    if sort_cols:
        work = work.sort_values(sort_cols, kind="mergesort")
    return work.reset_index(drop=True)


def chronological_ratio_split(
    df: pd.DataFrame,
    *,
    time_col: str = "ts",
    ratio_train: float = 0.70,
    ratio_val: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phase 2 unique-timestamp 70/15/15. Wrapper so training code does not reimplement splits."""
    train, val, test, _cuts = chronological_split_fractions(
        df, time_col=time_col, train_frac=ratio_train, val_frac=ratio_val
    )
    return train, val, test


def _window_meta(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, *, split_mode: str
) -> WindowMeta:
    def _span(part: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if part.empty:
            return None, None
        ts = pd.to_datetime(part["ts"], utc=True)
        return ts.min(), ts.max()

    tr_s, tr_e = _span(train)
    va_s, va_e = _span(val)
    te_s, te_e = _span(test)
    return WindowMeta(
        train_start=tr_s,
        train_end=tr_e,
        val_start=va_s,
        val_end=va_e,
        test_start=te_s,
        test_end=te_e,
        split_mode=split_mode,
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
    )


def _from_split_column(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if "split" not in df.columns:
        return None
    labels = set(df["split"].astype(str).unique())
    if not {"train", "val", "test"}.issubset(labels):
        return None
    train = df.loc[df["split"].astype(str) == "train"]
    val = df.loc[df["split"].astype(str) == "val"]
    test = df.loc[df["split"].astype(str) == "test"]
    if train.empty or val.empty or test.empty:
        return None
    return train, val, test


def split_feature_frame(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    force_ratio: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, WindowMeta]:
    """Prefer Phase 2 precomputed split labels, else unique-timestamp 70/15/15."""
    split_cfg = config.get("split") or {}
    if not force_ratio:
        pre = _from_split_column(df)
        if pre is not None:
            train, val, test = pre
            return train, val, test, _window_meta(train, val, test, split_mode="precomputed")

        method = str(split_cfg.get("method") or "global_fractions")
        if method == "date":
            train, val, test = chronological_split(
                df,
                train_end=split_cfg.get("train_end") or DEFAULT_TRAIN_END,
                val_end=split_cfg.get("val_end") or DEFAULT_VAL_END,
            )
            if len(train) and len(val) and len(test):
                return train, val, test, _window_meta(train, val, test, split_mode="date")
            LOGGER.warning(
                "Date split produced empty train/val/test (n=%s/%s/%s); using chronological fractions",
                len(train),
                len(val),
                len(test),
            )

    train, val, test, _cuts = chronological_split_fractions(
        df,
        train_frac=float(split_cfg.get("train_frac", split_cfg.get("ratio_train", 0.70))),
        val_frac=float(split_cfg.get("val_frac", split_cfg.get("ratio_val", 0.15))),
    )
    return train, val, test, _window_meta(train, val, test, split_mode="ratio")


def _xy(
    part: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = part[feature_cols].to_numpy(dtype=np.float64)
    y = part[TARGET_COL].to_numpy(dtype=np.float64).reshape(-1)
    ts = pd.to_datetime(part["ts"], utc=True).to_numpy()
    if "building_id" in part.columns:
        groups = part["building_id"].astype(str).to_numpy()
    else:
        groups = np.array(["_all"] * len(part), dtype=object)
    return x, y, ts, groups


def _prepared_from_parts(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    windows: WindowMeta,
) -> PreparedSplits:
    x_train, y_train, ts_train, g_train = _xy(train, feature_cols)
    x_val, y_val, ts_val, g_val = _xy(val, feature_cols)
    x_test, y_test, ts_test, g_test = _xy(test, feature_cols)
    scaler = StandardScaler()
    scaler.fit(x_train)
    building = None
    if "building_id" in train.columns:
        ids = pd.concat([train, val, test], ignore_index=True)["building_id"].astype(str).unique()
        if len(ids) == 1:
            building = str(ids[0])
    return PreparedSplits(
        feature_names=feature_cols,
        scaler=scaler,
        X_train=x_train,
        y_train=y_train,
        ts_train=ts_train,
        groups_train=g_train,
        X_val=x_val,
        y_val=y_val,
        ts_val=ts_val,
        groups_val=g_val,
        X_test=x_test,
        y_test=y_test,
        ts_test=ts_test,
        groups_test=g_test,
        windows=windows,
        building_id=building,
    )


def prepare_splits(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    force_ratio: bool = False,
) -> PreparedSplits:
    work = ensure_target(df)
    if "ts" not in work.columns:
        raise ValueError("feature matrix must include ts")
    work["ts"] = pd.to_datetime(work["ts"], utc=True)
    sort_cols = [col for col in ("building_id", "ts") if col in work.columns]
    work = work.sort_values(sort_cols, kind="mergesort")
    feature_cols = select_feature_columns(work)
    work = drop_incomplete(work, feature_cols)
    if work.empty:
        raise ValueError("no rows remain after dropping NaN features/target")
    train, val, test, windows = split_feature_frame(work, config, force_ratio=force_ratio)
    return _prepared_from_parts(train, val, test, feature_cols, windows)


def prepare_from_pre_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> PreparedSplits:
    """Use Phase 2 `features_train` / `_val` / `_test` as-is (already chronological)."""
    train_e = drop_incomplete(ensure_target(train), select_feature_columns(ensure_target(train)))
    feature_cols = select_feature_columns(train_e)
    val_e = drop_incomplete(ensure_target(val), feature_cols)
    test_e = drop_incomplete(ensure_target(test), feature_cols)
    windows = _window_meta(train_e, val_e, test_e, split_mode="precomputed_files")
    return _prepared_from_parts(train_e, val_e, test_e, feature_cols, windows)


def scale_features(prepared: PreparedSplits) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the train-fitted scaler. Never refit on val/test."""
    return (
        prepared.scaler.transform(prepared.X_train),
        prepared.scaler.transform(prepared.X_val),
        prepared.scaler.transform(prepared.X_test),
    )


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Windows of length seq_len ending at t to predict y_{t+1} stored at that row."""
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if len(X) < seq_len:
        raise ValueError(f"need at least seq_len={seq_len} rows, got {len(X)}")
    xs = np.stack([X[i - seq_len + 1 : i + 1] for i in range(seq_len - 1, len(X))])
    ys = y[seq_len - 1 :]
    return xs, ys


def make_sequences_with_history(
    X_hist: np.ndarray,
    y_hist: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every row in X/y using prior-split rows as warm-up. No future leakage."""
    n_features = X.shape[1] if len(X) else (X_hist.shape[1] if len(X_hist) else 0)
    if len(X) == 0:
        return np.empty((0, seq_len, n_features)), np.empty((0,))
    x_all = np.concatenate([X_hist, X], axis=0) if len(X_hist) else X
    y_all = np.concatenate([y_hist, y], axis=0) if len(y_hist) else y
    start = len(X_hist)
    xs = []
    ys = []
    for i in range(start, start + len(X)):
        left = i - seq_len + 1
        if left < 0:
            continue
        xs.append(x_all[left : i + 1])
        ys.append(y_all[i])
    if not xs:
        return np.empty((0, seq_len, n_features)), np.empty((0,))
    return np.stack(xs), np.asarray(ys, dtype=np.float64)


def _unique_groups(groups: np.ndarray) -> np.ndarray:
    return pd.unique(groups)


def make_sequences_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seq_len: int,
    ts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-building windows so sequences never cross building boundaries.

    Returns (X_seq, y_seq, ts_seq, groups_seq). groups_seq aligns with each window
    so LSTM/CNN-LSTM predictions can be persisted with building_id.
    """
    n_features = X.shape[1]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    t_out: list[np.ndarray] = []
    g_out: list[np.ndarray] = []
    empty_g = np.array([], dtype=object)
    empty_t = np.array([], dtype="datetime64[ns]")
    for grp in _unique_groups(groups):
        mask = groups == grp
        xg, yg = X[mask], y[mask]
        if len(xg) < seq_len:
            continue
        xa, ya = make_sequences(xg, yg, seq_len)
        xs.append(xa)
        ys.append(ya)
        g_out.append(np.full(len(ya), grp, dtype=object))
        if ts is not None:
            t_out.append(ts[mask][seq_len - 1 :])
    if not xs:
        return np.empty((0, seq_len, n_features)), np.empty((0,)), empty_t, empty_g
    ts_arr = np.concatenate(t_out) if t_out else empty_t
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), ts_arr, np.concatenate(g_out)


def make_sequences_with_history_grouped(
    X_hist: np.ndarray,
    y_hist: np.ndarray,
    groups_hist: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seq_len: int,
    ts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_features = X.shape[1] if len(X) else X_hist.shape[1]
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    t_out: list[np.ndarray] = []
    g_out: list[np.ndarray] = []
    empty_g = np.array([], dtype=object)
    empty_t = np.array([], dtype="datetime64[ns]")
    for grp in _unique_groups(groups):
        hist_mask = groups_hist == grp
        cur_mask = groups == grp
        xa, ya = make_sequences_with_history(
            X_hist[hist_mask], y_hist[hist_mask], X[cur_mask], y[cur_mask], seq_len
        )
        if len(ya) == 0:
            continue
        xs.append(xa)
        ys.append(ya)
        g_out.append(np.full(len(ya), grp, dtype=object))
        if ts is not None:
            t_cur = ts[cur_mask]
            t_out.append(t_cur[-len(ya) :])
    if not xs:
        return np.empty((0, seq_len, n_features)), np.empty((0,)), empty_t, empty_g
    ts_arr = np.concatenate(t_out) if t_out else empty_t
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), ts_arr, np.concatenate(g_out)


def _existing_table(processed_dir: Path, stem: str) -> Path | None:
    for suffix in (".parquet", ".csv"):
        path = processed_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def load_feature_matrix(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported feature matrix type: {path}")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_processed_splits(
    processed_dir: Path, *, building_id: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    train_p = _existing_table(processed_dir, "features_train")
    val_p = _existing_table(processed_dir, "features_val")
    test_p = _existing_table(processed_dir, "features_test")
    if not (train_p and val_p and test_p):
        return None

    def _filter(frame: pd.DataFrame) -> pd.DataFrame:
        if building_id is None or "building_id" not in frame.columns:
            return frame
        out = frame.loc[frame["building_id"].astype(str) == str(building_id)].copy()
        if out.empty:
            raise ValueError(f"no rows for building_id={building_id!r}")
        return out

    return (
        _filter(load_feature_matrix(train_p)),
        _filter(load_feature_matrix(val_p)),
        _filter(load_feature_matrix(test_p)),
    )


def load_processed_frame(processed_dir: Path, *, building_id: str | None = None) -> pd.DataFrame:
    feat = _existing_table(processed_dir, "features")
    if feat is None:
        raise FileNotFoundError(
            f"no Phase 2 feature matrix under {processed_dir} "
            "(expected features.parquet/csv or features_train/val/test). "
            "Run `python -m ml prepare` first."
        )
    df = load_feature_matrix(feat)
    if building_id is not None:
        if "building_id" not in df.columns:
            raise ValueError("building_id filter requested but column is missing")
        df = df.loc[df["building_id"].astype(str) == str(building_id)].copy()
        if df.empty:
            raise ValueError(f"no rows for building_id={building_id!r}")
    sort_cols = [col for col in ("building_id", "ts") if col in df.columns]
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
