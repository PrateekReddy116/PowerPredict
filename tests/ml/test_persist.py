from pathlib import Path

from ml.config import apply_smoke_overrides, load_config
from ml.evaluate.persist import load_metrics_json, save_metrics_json, try_insert_model_metrics
from ml.train import run_training


def test_metrics_json_has_db_compatible_fields(fixture_frame, tmp_path: Path):
    config = apply_smoke_overrides(load_config())
    payload = run_training(
        frame=fixture_frame,
        config=config,
        artifacts_dir=tmp_path,
        model_names=["linear_regression"],
        smoke=True,
        force_ratio=True,
        data_source="synthetic_fixture",
        persist_db=False,
    )
    row = payload["models"][0]
    required = {
        "model_name",
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
        "mae",
        "rmse",
        "r2",
        "mape",
        "artifact_path",
        "config_json",
        "random_seed",
        "training_time_sec",
    }
    assert required.issubset(row.keys())
    assert row["model_name"] == "linear_regression"
    assert isinstance(row["mae"], float)
    assert isinstance(row["training_time_sec"], float)
    assert Path(row["artifact_path"]).exists()
    saved = load_metrics_json(tmp_path / "metrics.json")
    assert saved["data_source"] == "synthetic_fixture"
    assert saved["models"][0]["mae"] == row["mae"]


def test_db_insert_returns_false_when_url_missing(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    ok = try_insert_model_metrics(
        {
            "model_name": "linear_regression",
            "mae": 1.0,
            "rmse": 1.0,
            "r2": 0.0,
            "mape": None,
            "artifact_path": "artifacts/linear_regression/model.joblib",
            "config_json": {},
            "random_seed": 42,
            "training_time_sec": 0.1,
        }
    )
    assert ok is False


def test_save_metrics_json_roundtrip(tmp_path: Path):
    path = tmp_path / "metrics.json"
    save_metrics_json({"models": [{"model_name": "x", "mae": 1.25}]}, path)
    loaded = load_metrics_json(path)
    assert loaded["models"][0]["mae"] == 1.25


def test_evaluate_cli_writes_plots_from_real_predictions(fixture_frame, tmp_path: Path):
    from ml.evaluate.__main__ import generate_plots as cli_plots

    config = apply_smoke_overrides(load_config())
    run_training(
        frame=fixture_frame,
        config=config,
        artifacts_dir=tmp_path,
        model_names=["linear_regression"],
        smoke=True,
        force_ratio=True,
        data_source="synthetic_fixture",
        persist_db=False,
    )
    written = cli_plots(tmp_path)
    names = {p.name for p in written}
    assert "actual_vs_pred_linear_regression.png" in names
    assert "residuals_linear_regression.png" in names
    assert "comparison_table.png" in names
    assert "comparison.csv" in names
