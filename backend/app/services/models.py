"""Stored training runs and test metrics. Never fabricate MAE/RMSE/R²/MAPE."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models.model_metric import ModelMetric


def list_model_runs(db: Session) -> list[ModelMetric]:
    return db.query(ModelMetric).order_by(ModelMetric.created_at.desc()).all()


def _parse_ts(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _row_from_metric(row: ModelMetric) -> dict[str, Any]:
    return {
        "id": row.id,
        "model_name": row.model_name,
        "created_at": row.created_at,
        "train_start": row.train_start,
        "train_end": row.train_end,
        "val_start": row.val_start,
        "val_end": row.val_end,
        "test_start": row.test_start,
        "test_end": row.test_end,
        "mae": row.mae,
        "rmse": row.rmse,
        "r2": row.r2,
        "mape": row.mape,
        "artifact_path": row.artifact_path,
        "random_seed": row.random_seed,
        "training_time_sec": row.training_time_sec,
        "source": "database",
        "data_source": None,
        "smoke": None,
        "n": None,
    }


def _rows_from_metrics_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    data_source = payload.get("data_source")
    smoke = payload.get("smoke") if isinstance(payload.get("smoke"), bool) else None
    rows: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict) or not item.get("model_name"):
            continue
        rows.append(
            {
                "id": None,
                "model_name": str(item["model_name"]),
                "created_at": None,
                "train_start": _parse_ts(item.get("train_start")),
                "train_end": _parse_ts(item.get("train_end")),
                "val_start": _parse_ts(item.get("val_start")),
                "val_end": _parse_ts(item.get("val_end")),
                "test_start": _parse_ts(item.get("test_start")),
                "test_end": _parse_ts(item.get("test_end")),
                "mae": item.get("mae"),
                "rmse": item.get("rmse"),
                "r2": item.get("r2"),
                "mape": item.get("mape"),
                "artifact_path": item.get("artifact_path"),
                "random_seed": item.get("random_seed"),
                "training_time_sec": item.get("training_time_sec"),
                "source": "artifacts",
                "data_source": item.get("data_source") or data_source,
                "smoke": smoke,
                "n": item.get("n"),
            }
        )
    return rows


def list_model_performance(
    db: Session,
    artifacts_dir: Path,
) -> list[dict[str, Any]]:
    """DB rows first. If none, read artifacts/metrics.json only — never invent scores."""
    db_rows = [_row_from_metric(row) for row in list_model_runs(db)]
    if db_rows:
        return db_rows
    return _rows_from_metrics_json(Path(artifacts_dir) / "metrics.json")
