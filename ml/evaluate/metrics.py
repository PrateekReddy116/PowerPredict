"""Regression metrics. MAPE uses a documented |y| floor; never fabricate scores."""

from __future__ import annotations

from typing import Any

import numpy as np

# Default floor in kWh. Below this, relative error is unstable (cleaned zeros / near-zeros).
DEFAULT_MAPE_FLOOR_KWH = 1.0


def _as_1d(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).reshape(-1)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred length mismatch")
    if yt.size == 0:
        raise ValueError("cannot score empty arrays")
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred length mismatch")
    if yt.size == 0:
        raise ValueError("cannot score empty arrays")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred length mismatch")
    if yt.size == 0:
        raise ValueError("cannot score empty arrays")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def mape_with_floor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    floor: float = DEFAULT_MAPE_FLOOR_KWH,
) -> tuple[float | None, int, int]:
    """MAPE in percent on rows where |y_true| >= floor.

    Returns (mape_or_none, n_used, n_excluded). None when no rows pass the floor.
    """
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred length mismatch")
    mask = np.abs(yt) >= float(floor)
    n_used = int(mask.sum())
    n_excluded = int(yt.size - n_used)
    if n_used == 0:
        return None, n_used, n_excluded
    pct = np.abs((yt[mask] - yp[mask]) / yt[mask]) * 100.0
    return float(np.mean(pct)), n_used, n_excluded


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mape_floor: float = DEFAULT_MAPE_FLOOR_KWH,
) -> dict[str, Any]:
    mape, n_used, n_excl = mape_with_floor(y_true, y_pred, floor=mape_floor)
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mape,
        "mape_floor_kwh": float(mape_floor),
        "mape_n_used": n_used,
        "mape_n_excluded": n_excl,
        "n": int(_as_1d(y_true).size),
    }
