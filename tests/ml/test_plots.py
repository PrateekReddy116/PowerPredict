from pathlib import Path

import numpy as np
import pandas as pd
from ml.evaluate.plots import plot_actual_vs_predicted, plot_comparison_table, plot_residuals


def test_plot_functions_on_fixture_predictions(tmp_path: Path):
    ts = pd.date_range("2017-10-01", periods=12, freq="h", tz="UTC")
    y_true = np.linspace(10.0, 21.0, 12)
    y_pred = y_true + np.array([0.2, -0.1, 0.0, 0.4, -0.3, 0.1, 0.0, -0.2, 0.3, 0.0, -0.4, 0.2])
    actual = plot_actual_vs_predicted(
        ts, y_true, y_pred, tmp_path / "actual.png", title="fixture actual vs pred"
    )
    resid = plot_residuals(ts, y_true, y_pred, tmp_path / "resid.png", title="fixture residuals")
    table = plot_comparison_table(
        [
            {
                "model_name": "fixture_only",
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "r2": 0.99,
                "mape": 1.2,
                "training_time_sec": 0.01,
            }
        ],
        tmp_path / "table.png",
    )
    assert actual.exists() and actual.stat().st_size > 0
    assert resid.exists() and resid.stat().st_size > 0
    assert table.exists() and table.stat().st_size > 0
