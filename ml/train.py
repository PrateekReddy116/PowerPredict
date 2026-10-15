"""Offline training CLI: `python -m ml.train`.

Fits the five forecasting models on a Phase 2 feature matrix (or a test fixture).
Writes artifacts and metrics.json. Does not invent scores. Does not shuffle splits.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.config import apply_smoke_overrides, load_config
from ml.evaluate.metrics import DEFAULT_MAPE_FLOOR_KWH, regression_metrics
from ml.evaluate.persist import save_metrics_json, try_insert_forecasts, try_insert_model_metrics
from ml.models import ALL_MODEL_NAMES, FORECASTER_CLASSES, is_sequence_model
from ml.models.base import library_versions, set_random_seeds
from ml.models.dataset import (
    make_sequences_grouped,
    make_sequences_with_history_grouped,
    prepare_from_pre_split,
    prepare_splits,
    scale_features,
)
from ml.paths import DEFAULT_ARTIFACTS_DIR, DEFAULT_CONFIG, REPO_ROOT

LOGGER = logging.getLogger("ml.train")


def _model_dir(artifacts_dir: Path, name: str) -> Path:
    path = artifacts_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_filename(name: str) -> str:
    return "model.pt" if is_sequence_model(name) else "model.joblib"


def _write_predictions(
    path: Path,
    ts: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    building_ids: np.ndarray | None = None,
) -> None:
    data: dict[str, Any] = {
        "ts": pd.to_datetime(ts, utc=True),
        "y_true": np.asarray(y_true, dtype=np.float64).reshape(-1),
        "y_pred": np.asarray(y_pred, dtype=np.float64).reshape(-1),
    }
    preds = data["y_pred"]
    if building_ids is not None and len(building_ids) == len(preds):
        data["building_id"] = np.asarray(building_ids).reshape(-1)
    pd.DataFrame(data).to_csv(path, index=False)


def train_one(
    name: str,
    prepared,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    *,
    seq_len: int,
    model_cfg: dict[str, Any],
    random_seed: int,
    artifacts_dir: Path,
    mape_floor: float,
    extra_config: dict[str, Any],
) -> dict[str, Any]:
    cls = FORECASTER_CLASSES[name]
    cfg = deepcopy(model_cfg)
    cfg["seq_len"] = seq_len
    cfg["device"] = extra_config.get("device", "cpu")
    model = cls(config=cfg, random_seed=random_seed)

    if is_sequence_model(name):
        x_tr, y_tr, _ts_tr, _g_tr = make_sequences_grouped(
            x_train, prepared.y_train, prepared.groups_train, seq_len
        )
        x_va, y_va, _ts_va, _g_va = make_sequences_with_history_grouped(
            x_train,
            prepared.y_train,
            prepared.groups_train,
            x_val,
            prepared.y_val,
            prepared.groups_val,
            seq_len,
        )
        x_hist = np.concatenate([x_train, x_val], axis=0)
        y_hist = np.concatenate([prepared.y_train, prepared.y_val], axis=0)
        g_hist = np.concatenate([prepared.groups_train, prepared.groups_val], axis=0)
        x_te, y_te, ts_te, groups_te = make_sequences_with_history_grouped(
            x_hist,
            y_hist,
            g_hist,
            x_test,
            prepared.y_test,
            prepared.groups_test,
            seq_len,
            ts=prepared.ts_test,
        )
        if len(x_tr) == 0:
            raise ValueError(f"{name}: training sequences empty (need seq_len={seq_len})")
        t0 = time.perf_counter()
        model.fit(x_tr, y_tr, x_va if len(x_va) else None, y_va if len(y_va) else None)
        training_time_sec = time.perf_counter() - t0
        y_pred = model.predict(x_te) if len(x_te) else np.empty((0,))
        y_true = y_te
    else:
        t0 = time.perf_counter()
        model.fit(
            x_train,
            prepared.y_train,
            x_val,
            prepared.y_val,
            feature_names=prepared.feature_names,
        )
        training_time_sec = time.perf_counter() - t0
        y_pred = model.predict(x_test)
        y_true = prepared.y_test
        ts_te = prepared.ts_test

    metrics = regression_metrics(y_true, y_pred, mape_floor=mape_floor)
    dest = _model_dir(artifacts_dir, name)
    artifact_path = dest / _artifact_filename(name)
    model.save(artifact_path)
    groups_out = groups_te if is_sequence_model(name) else prepared.groups_test
    _write_predictions(dest / "test_predictions.csv", ts_te, y_true, y_pred, building_ids=groups_out)

    windows = prepared.windows.as_dict()
    config_json = {
        "model": json_safe_cfg(cfg),
        "seq_len": seq_len,
        "feature_names": prepared.feature_names,
        "library_versions": library_versions(),
        "split_mode": windows["split_mode"],
        "training_time_sec": training_time_sec,
        "epochs_trained": getattr(model, "epochs_trained", None),
        "best_val_rmse": getattr(model, "best_val_rmse", None),
    }
    row = {
        "model_name": name,
        "building_id": prepared.building_id,
        **{k: windows[k] for k in (
            "train_start",
            "train_end",
            "val_start",
            "val_end",
            "test_start",
            "test_end",
        )},
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "mape": metrics["mape"],
        "mape_floor_kwh": metrics["mape_floor_kwh"],
        "mape_n_used": metrics["mape_n_used"],
        "mape_n_excluded": metrics["mape_n_excluded"],
        "n": metrics["n"],
        "artifact_path": artifact_path.as_posix(),
        "config_json": config_json,
        "random_seed": random_seed,
        "training_time_sec": training_time_sec,
    }
    save_metrics_json(row, dest / "metrics.json")
    LOGGER.info(
        "%s test MAE=%s RMSE=%s R2=%s MAPE=%s time=%.3fs",
        name,
        row["mae"],
        row["rmse"],
        row["r2"],
        row["mape"],
        training_time_sec,
    )
    return row


def json_safe_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def run_training(
    *,
    frame: pd.DataFrame | None = None,
    train: pd.DataFrame | None = None,
    val: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    config: dict[str, Any],
    artifacts_dir: Path,
    model_names: list[str],
    smoke: bool = False,
    force_ratio: bool = False,
    data_source: str = "unknown",
    persist_db: bool = True,
) -> dict[str, Any]:
    cfg = apply_smoke_overrides(config) if smoke else deepcopy(config)
    seed = int(cfg.get("random_seed", 42))
    set_random_seeds(seed)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if train is not None and val is not None and test is not None:
        prepared = prepare_from_pre_split(train, val, test)
    elif frame is not None:
        prepared = prepare_splits(frame, cfg, force_ratio=force_ratio)
    else:
        raise ValueError("provide frame or train/val/test feature matrices")
    x_train, x_val, x_test = scale_features(prepared)
    joblib.dump(
        {
            "scaler": prepared.scaler,
            "feature_names": prepared.feature_names,
            "seq_len": int(cfg.get("seq_len", 24)),
        },
        artifacts_dir / "feature_scaler.joblib",
    )

    seq_len = int(cfg.get("seq_len", 24))
    mape_floor = float(cfg.get("mape_floor_kwh", DEFAULT_MAPE_FLOOR_KWH))
    models_cfg = cfg.get("models") or {}
    rows: list[dict[str, Any]] = []
    for name in model_names:
        if name not in FORECASTER_CLASSES:
            raise ValueError(f"unknown model {name!r}; choose from {ALL_MODEL_NAMES}")
        row = train_one(
            name,
            prepared,
            x_train,
            x_val,
            x_test,
            seq_len=seq_len,
            model_cfg=dict(models_cfg.get(name) or {}),
            random_seed=seed,
            artifacts_dir=artifacts_dir,
            mape_floor=mape_floor,
            extra_config=cfg,
        )
        rows.append(row)
        if persist_db:
            metric_id = try_insert_model_metrics(row)
            inserted = try_insert_forecasts(
                artifacts_dir / name / "test_predictions.csv",
                model_metrics_id=metric_id,
                split="test",
            )
            if not inserted.get("ok"):
                LOGGER.warning("forecasts not persisted for %s: %s", name, inserted.get("reason"))

    payload = {
        "data_source": data_source,
        "random_seed": seed,
        "seq_len": seq_len,
        "mape_floor_kwh": mape_floor,
        "smoke": bool(smoke),
        "feature_names": prepared.feature_names,
        "building_id": prepared.building_id,
        "split": prepared.windows.as_dict(),
        "library_versions": library_versions(),
        "models": rows,
    }
    save_metrics_json(payload, artifacts_dir / "metrics.json")
    return payload


def _resolve_under_root(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PowerPredict energy forecasting models")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--models",
        default=",".join(ALL_MODEL_NAMES),
        help="comma-separated model names",
    )
    parser.add_argument("--smoke", action="store_true", help="1 epoch / tiny trees for a sanity run")
    parser.add_argument("--data", type=Path, default=None, help="parquet/csv feature matrix")
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--building-id", default=None)
    parser.add_argument("--force-ratio-split", action="store_true")
    parser.add_argument("--no-db", action="store_true", help="do not attempt model_metrics insert")
    return parser


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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from ml.ingest.db import load_env_file

    load_env_file()
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    paths = config.get("paths") or {}
    artifacts_dir = _resolve_under_root(
        args.artifacts_dir or paths.get("artifacts_dir") or DEFAULT_ARTIFACTS_DIR
    )
    names = [n.strip() for n in str(args.models).split(",") if n.strip()]

    if args.data is not None:
        from ml.models.dataset import load_feature_matrix

        data_path = _resolve_under_root(args.data)
        frame = load_feature_matrix(data_path)
        run_training(
            frame=frame,
            config=config,
            artifacts_dir=artifacts_dir,
            model_names=names,
            smoke=bool(args.smoke),
            force_ratio=bool(args.force_ratio_split),
            data_source=data_path.as_posix(),
            persist_db=not args.no_db,
        )
    else:
        from ml.models.dataset import load_processed_frame, load_processed_splits

        processed = _resolve_under_root(
            args.processed_dir or paths.get("processed_dir") or "data/processed"
        )
        try:
            splits = load_processed_splits(processed, building_id=args.building_id)
            if splits is not None:
                run_training(
                    train=splits[0],
                    val=splits[1],
                    test=splits[2],
                    config=config,
                    artifacts_dir=artifacts_dir,
                    model_names=names,
                    smoke=bool(args.smoke),
                    force_ratio=False,
                    data_source=_processed_source_label(processed),
                    persist_db=not args.no_db,
                )
            else:
                frame = load_processed_frame(processed, building_id=args.building_id)
                run_training(
                    frame=frame,
                    config=config,
                    artifacts_dir=artifacts_dir,
                    model_names=names,
                    smoke=bool(args.smoke),
                    force_ratio=bool(args.force_ratio_split),
                    data_source=_processed_source_label(processed),
                    persist_db=not args.no_db,
                )
        except FileNotFoundError as exc:
            LOGGER.error("%s", exc)
            LOGGER.error("Processed BDG2 is not present. Tests train on fixtures; do not invent metrics.")
            return 2
    LOGGER.info("wrote %s", artifacts_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
