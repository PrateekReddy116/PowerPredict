"""Evaluation plots from *computed* predictions / metrics JSON — never hardcoded scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_actual_vs_predicted(
    ts: np.ndarray | pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    *,
    title: str,
) -> Path:
    _ensure_dir(path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, y_true, label="actual", linewidth=1.2)
    ax.plot(ts, y_pred, label="predicted", linewidth=1.2, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("energy (kWh)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_residuals(
    ts: np.ndarray | pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    *,
    title: str,
) -> Path:
    _ensure_dir(path)
    resid = np.asarray(y_true, dtype=np.float64).reshape(-1) - np.asarray(
        y_pred, dtype=np.float64
    ).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ts, resid, linewidth=1.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("residual vs time")
    axes[0].set_xlabel("timestamp")
    axes[0].set_ylabel("actual − predicted (kWh)")
    axes[1].hist(resid, bins=min(30, max(5, resid.size // 3)))
    axes[1].set_title("residual histogram")
    axes[1].set_xlabel("residual (kWh)")
    fig.suptitle(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_comparison_table(rows: list[dict[str, Any]], path: Path) -> Path:
    """Table image built from metrics JSON rows, not from hardcoded academic scores."""
    _ensure_dir(path)
    if not rows:
        raise ValueError("no metric rows to plot")
    columns = ["model_name", "mae", "rmse", "r2", "mape", "training_time_sec"]

    def _fmt(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    cell_text = [[_fmt(row.get(col)) for col in columns] for row in rows]
    fig_h = 1.2 + 0.45 * len(rows)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    ax.set_title("Test-set comparison (from computed metrics.json)")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    _ensure_dir(path)
    frame = pd.DataFrame(rows)
    keep = [
        c
        for c in (
            "model_name",
            "mae",
            "rmse",
            "r2",
            "mape",
            "training_time_sec",
            "n",
            "artifact_path",
        )
        if c in frame.columns
    ]
    frame[keep].to_csv(path, index=False)
    return path
