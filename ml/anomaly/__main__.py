"""Offline Isolation Forest CLI: `python -m ml.anomaly`.

Fits on chronological train only. Writes artifacts/anomaly/. Best-effort DB insert.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ml.anomaly.config import load_anomaly_settings
from ml.anomaly.detect import run_anomaly_detection
from ml.anomaly.persist import try_insert_anomalies, write_anomaly_artifacts
from ml.evaluate.persist import save_metrics_json
from ml.config import load_config
from ml.paths import DEFAULT_ARTIFACTS_DIR, DEFAULT_CONFIG, DEFAULT_PROCESSED_DIR, REPO_ROOT

LOGGER = logging.getLogger("ml.anomaly")


def _resolve(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _processed_source_label(processed: Path) -> str:
    summary = processed / "pipeline_summary.json"
    if summary.exists():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            src = payload.get("source")
            if src:
                return f"{src}:{processed.as_posix()}"
        except (OSError, json.JSONDecodeError):
            pass
    return processed.as_posix()


def _load_buildings(processed: Path):
    import pandas as pd

    for stem in ("buildings",):
        for suffix in (".parquet", ".csv"):
            path = processed / f"{stem}{suffix}"
            if path.exists():
                if suffix == ".parquet":
                    return pd.read_parquet(path)
                return pd.read_csv(path)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Isolation Forest energy anomalies (train-only fit)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--building-id", default=None)
    parser.add_argument("--fixture", action="store_true", help="Score the labeled TEST FIXTURE (not BDG2)")
    parser.add_argument("--score-train", action="store_true", help="Also score the train split (contamination-rate flags)")
    parser.add_argument("--no-db", action="store_true", help="do not attempt anomalies table insert")
    parser.add_argument("--no-saved-forecaster", action="store_true", help="always fit the train-only energy baseline")
    return parser


def _load_splits(processed: Path, building_id: str | None):
    from ml.models.dataset import load_processed_frame, load_processed_splits

    splits = load_processed_splits(processed, building_id=building_id)
    if splits is not None:
        return splits
    frame = load_processed_frame(processed, building_id=building_id)
    train = frame.loc[frame["split"].astype(str) == "train"] if "split" in frame.columns else frame
    val = frame.loc[frame["split"].astype(str) == "val"] if "split" in frame.columns else frame.iloc[0:0]
    test = frame.loc[frame["split"].astype(str) == "test"] if "split" in frame.columns else frame.iloc[0:0]
    return train, val, test


def _fixture_splits():
    from ml.ingest.fixture import (
        make_fixture_electricity_wide,
        make_fixture_metadata,
        make_fixture_weather,
    )
    from ml.pipeline import run_pipeline

    result = run_pipeline(
        make_fixture_metadata(),
        make_fixture_electricity_wide(),
        make_fixture_weather(),
        source="test_fixture",
        n_buildings=2,
    )
    return result.train, result.val, result.test, "test_fixture", result.buildings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from ml.ingest.db import load_env_file

    load_env_file()
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    settings = load_anomaly_settings(config=config)
    paths = config.get("paths") or {}
    artifacts_root = _resolve(args.artifacts_dir or paths.get("artifacts_dir") or DEFAULT_ARTIFACTS_DIR)
    anomaly_dir = artifacts_root / "anomaly"
    forecast_artifacts = artifacts_root

    buildings = None
    if args.fixture:
        train, val, test, source, buildings = _fixture_splits()
    else:
        processed = _resolve(args.processed_dir or paths.get("processed_dir") or DEFAULT_PROCESSED_DIR)
        try:
            train, val, test = _load_splits(processed, args.building_id)
        except FileNotFoundError as exc:
            LOGGER.error("%s", exc)
            LOGGER.error("Processed features are missing. Use --fixture for the TEST FIXTURE, not BDG2.")
            return 2
        source = _processed_source_label(processed)
        buildings = _load_buildings(processed)

    result = run_anomaly_detection(
        train,
        val,
        test,
        settings=settings,
        artifacts_dir=None if args.no_saved_forecaster else forecast_artifacts,
        processed_source=source,
        score_train=bool(args.score_train),
        prefer_saved_forecaster=not args.no_saved_forecaster,
    )
    written = write_anomaly_artifacts(result.scores, result.summary, anomaly_dir, detectors=result.detectors)
    LOGGER.info("wrote %s", written.get("summary"))
    LOGGER.info(
        "scored=%s flagged=%s features=%s contamination=%s seed=%s expected=%s",
        result.summary.get("n_scored"),
        result.summary.get("n_flagged"),
        result.summary.get("if_features"),
        result.summary.get("contamination"),
        result.summary.get("random_seed"),
        (result.summary.get("residual") or {}).get("expected_source"),
    )
    if not args.no_db:
        db_status = try_insert_anomalies(result.scores, buildings=buildings)
        result.summary["db"] = db_status
        save_metrics_json(result.summary, anomaly_dir / "summary.json")
        LOGGER.info("db insert: %s", db_status)
    print(json.dumps({k: result.summary[k] for k in (
        "data_source",
        "n_scored",
        "n_flagged",
        "by_severity",
        "by_building",
        "if_features",
        "contamination",
        "random_seed",
        "residual",
    ) if k in result.summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
