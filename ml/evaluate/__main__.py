"""Evaluation CLI: `python -m ml.evaluate`.

Reads artifacts/metrics.json and per-model test_predictions.csv (computed, not invented)
and writes plots under artifacts/plots/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from ml.config import load_config
from ml.evaluate.metrics import regression_metrics
from ml.evaluate.persist import load_metrics_json
from ml.evaluate.plots import (
    plot_actual_vs_predicted,
    plot_comparison_table,
    plot_residuals,
    write_comparison_csv,
)
from ml.paths import DEFAULT_ARTIFACTS_DIR, DEFAULT_CONFIG, REPO_ROOT

LOGGER = logging.getLogger("ml.evaluate")


def _resolve(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def generate_plots(artifacts_dir: Path, mape_floor: float | None = None) -> list[Path]:
    metrics_path = artifacts_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} is missing. Run `python -m ml.train` first; do not invent metrics."
        )
    payload = load_metrics_json(metrics_path)
    rows = payload.get("models") or []
    if not rows:
        raise ValueError("metrics.json contains no model rows")
    floor = float(mape_floor if mape_floor is not None else payload.get("mape_floor_kwh", 1.0))
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for row in rows:
        name = row["model_name"]
        pred_path = artifacts_dir / name / "test_predictions.csv"
        if not pred_path.exists():
            LOGGER.warning("skip plots for %s: %s missing", name, pred_path)
            continue
        frame = pd.read_csv(pred_path, parse_dates=["ts"])
        y_true = frame["y_true"].to_numpy()
        y_pred = frame["y_pred"].to_numpy()
        recomputed = regression_metrics(y_true, y_pred, mape_floor=floor)
        LOGGER.info(
            "%s recomputed test MAE=%s RMSE=%s R2=%s MAPE=%s (from predictions file)",
            name,
            recomputed["mae"],
            recomputed["rmse"],
            recomputed["r2"],
            recomputed["mape"],
        )
        ts = frame["ts"]
        written.append(
            plot_actual_vs_predicted(
                ts,
                y_true,
                y_pred,
                plots_dir / f"actual_vs_pred_{name}.png",
                title=f"{name}: actual vs predicted (test)",
            )
        )
        written.append(
            plot_residuals(
                ts,
                y_true,
                y_pred,
                plots_dir / f"residuals_{name}.png",
                title=f"{name}: residuals (test)",
            )
        )
    written.append(plot_comparison_table(rows, plots_dir / "comparison_table.png"))
    written.append(write_comparison_csv(rows, plots_dir / "comparison.csv"))
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Plot PowerPredict evaluation artifacts")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    paths = config.get("paths") or {}
    artifacts_dir = _resolve(
        args.artifacts_dir or paths.get("artifacts_dir") or DEFAULT_ARTIFACTS_DIR
    )
    try:
        written = generate_plots(artifacts_dir)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2
    for path in written:
        LOGGER.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
