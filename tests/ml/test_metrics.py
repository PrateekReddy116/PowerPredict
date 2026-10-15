import numpy as np
import pytest
from ml.evaluate.metrics import mape_with_floor, regression_metrics


def test_regression_metrics_match_known_tiny_vector():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])
    scores = regression_metrics(y_true, y_pred, mape_floor=0.5)
    assert scores["mae"] == pytest.approx(0.25)
    assert scores["rmse"] == pytest.approx(0.5)
    assert scores["r2"] == pytest.approx(0.8)
    # MAPE = mean(|e|/y)*100 on all four points: (0+0+0+0.25)/4 * 100 = 6.25
    assert scores["mape"] == pytest.approx(6.25)
    assert scores["mape_n_used"] == 4
    assert scores["mape_n_excluded"] == 0
    assert scores["n"] == 4


def test_mape_null_when_all_below_floor():
    y_true = np.array([0.1, 0.2, 0.0])
    y_pred = np.array([0.2, 0.2, 0.1])
    mape, n_used, n_excl = mape_with_floor(y_true, y_pred, floor=1.0)
    assert mape is None
    assert n_used == 0
    assert n_excl == 3


def test_mape_excludes_rows_below_floor():
    y_true = np.array([0.5, 10.0])
    y_pred = np.array([1.0, 12.0])
    mape, n_used, n_excl = mape_with_floor(y_true, y_pred, floor=1.0)
    assert n_used == 1
    assert n_excl == 1
    assert mape == pytest.approx(20.0)
