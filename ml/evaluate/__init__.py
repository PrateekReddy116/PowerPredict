"""Metrics helpers. Persist real MAE/RMSE/R² from a training run — never hardcode scores."""

from ml.evaluate.metrics import (
    DEFAULT_MAPE_FLOOR_KWH,
    mae,
    mape_with_floor,
    r2_score,
    regression_metrics,
    rmse,
)
from ml.evaluate.persist import (
    load_metrics_json,
    save_metrics_json,
    try_insert_forecasts,
    try_insert_model_metrics,
)
from ml.evaluate.plots import (
    plot_actual_vs_predicted,
    plot_comparison_table,
    plot_residuals,
    write_comparison_csv,
)

__all__ = [
    "DEFAULT_MAPE_FLOOR_KWH",
    "load_metrics_json",
    "mae",
    "mape_with_floor",
    "plot_actual_vs_predicted",
    "plot_comparison_table",
    "plot_residuals",
    "r2_score",
    "regression_metrics",
    "rmse",
    "save_metrics_json",
    "try_insert_forecasts",
    "try_insert_model_metrics",
    "write_comparison_csv",
]
