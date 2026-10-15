from pathlib import Path

import pandas as pd
from ml.config import apply_smoke_overrides, load_config
from ml.models.dataset import chronological_ratio_split, prepare_splits
from ml.train import run_training


def test_ratio_split_is_chronological_no_overlap(fixture_frame):
    work = fixture_frame.dropna().sort_values("ts", kind="mergesort")
    train, val, test = chronological_ratio_split(work)
    assert train["ts"].is_monotonic_increasing
    assert val["ts"].is_monotonic_increasing
    assert test["ts"].is_monotonic_increasing
    assert train["ts"].max() < val["ts"].min()
    assert val["ts"].max() < test["ts"].min()
    assert len(train) + len(val) + len(test) == len(work)


def test_prepare_splits_force_ratio_is_time_ordered(fixture_frame):
    config = load_config()
    prepared = prepare_splits(fixture_frame, config, force_ratio=True)
    assert prepared.windows.split_mode == "ratio"
    assert prepared.ts_train.max() < prepared.ts_val.min()
    assert prepared.ts_val.max() < prepared.ts_test.min()


def test_training_does_not_shuffle_and_shares_test_window(fixture_frame, tmp_path: Path):
    config = apply_smoke_overrides(load_config())
    payload = run_training(
        frame=fixture_frame,
        config=config,
        artifacts_dir=tmp_path,
        model_names=["linear_regression", "random_forest"],
        smoke=True,
        force_ratio=True,
        data_source="synthetic_fixture",
        persist_db=False,
    )
    rows = payload["models"]
    assert rows[0]["test_start"] == rows[1]["test_start"]
    assert rows[0]["test_end"] == rows[1]["test_end"]
    assert payload["split"]["split_mode"] == "ratio"
    # Train window ends before test window begins (chronological, no overlap).
    assert payload["split"]["train_end"] < payload["split"]["val_start"]
    assert payload["split"]["val_end"] < payload["split"]["test_start"]


def test_all_five_models_smoke_on_fixture(fixture_frame, tmp_path: Path):
    config = apply_smoke_overrides(load_config())
    payload = run_training(
        frame=fixture_frame,
        config=config,
        artifacts_dir=tmp_path,
        model_names=["linear_regression", "random_forest", "xgboost", "lstm", "cnn_lstm"],
        smoke=True,
        force_ratio=True,
        data_source="synthetic_fixture",
        persist_db=False,
    )
    names = [row["model_name"] for row in payload["models"]]
    assert names == ["linear_regression", "random_forest", "xgboost", "lstm", "cnn_lstm"]
    test_end = payload["models"][0]["test_end"]
    for row in payload["models"]:
        assert row["test_end"] == test_end
        assert row["mae"] is not None
        pred_path = tmp_path / row["model_name"] / "test_predictions.csv"
        assert pred_path.exists()
        preds = pd.read_csv(pred_path)
        assert "building_id" in preds.columns
        assert preds["building_id"].notna().all()
        assert (preds["building_id"].astype(str) != "_all").all()


def test_train_cli_exits_when_processed_missing(tmp_path: Path):
    from ml.train import main

    empty = tmp_path / "processed"
    empty.mkdir()
    out = tmp_path / "art"
    assert main([
        "--processed-dir",
        str(empty),
        "--artifacts-dir",
        str(out),
        "--smoke",
        "--no-db",
    ]) == 2
    assert not (out / "metrics.json").exists()


def test_ml_sources_do_not_enable_shuffle():
    root = Path("ml")
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "shuffle=True" in text or "shuffle = True" in text:
            offenders.append(path.as_posix())
        if "train_test_split" in text:
            offenders.append(path.as_posix())
    assert offenders == []
