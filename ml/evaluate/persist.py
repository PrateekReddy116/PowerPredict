"""Write metrics JSON (DB-compatible fields). Optional Postgres upsert — never fail the CLI."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from ml.models.base import json_safe

LOGGER = logging.getLogger(__name__)


def _parse_ts(value: str | None) -> datetime | None:
    if value is None or value == "":
        return None
    from pandas import Timestamp

    stamp = Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def save_metrics_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = json_safe(payload)
    path.write_text(json.dumps(serial, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load_metrics_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sqlite_connect_args(url: str) -> dict[str, bool]:
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def try_insert_model_metrics(row: dict[str, Any], *, database_url: str | None = None) -> UUID | bool:
    """Insert one training run. Returns the row id, or False if DATABASE_URL is missing or the DB is down."""
    url = database_url if database_url is not None else os.environ.get("DATABASE_URL")
    if not url:
        LOGGER.info("DATABASE_URL unset; skipping model_metrics insert")
        return False
    try:
        from sqlalchemy import MetaData, Table, create_engine
    except ImportError:
        LOGGER.warning("sqlalchemy not installed; skipping model_metrics insert")
        return False

    metric_id = uuid4()
    cfg = row.get("config_json")
    if url.startswith("sqlite") and isinstance(cfg, (dict, list)):
        cfg = json.dumps(cfg)
    values = {
        "id": str(metric_id),
        "model_name": row["model_name"],
        "created_at": datetime.now(UTC),
        "train_start": _parse_ts(row.get("train_start")),
        "train_end": _parse_ts(row.get("train_end")),
        "val_start": _parse_ts(row.get("val_start")),
        "val_end": _parse_ts(row.get("val_end")),
        "test_start": _parse_ts(row.get("test_start")),
        "test_end": _parse_ts(row.get("test_end")),
        "mae": row.get("mae"),
        "rmse": row.get("rmse"),
        "r2": row.get("r2"),
        "mape": row.get("mape"),
        "artifact_path": row.get("artifact_path"),
        "config_json": cfg,
        "random_seed": row.get("random_seed"),
        "training_time_sec": row.get("training_time_sec"),
    }
    try:
        engine = create_engine(url, connect_args=_sqlite_connect_args(url))
        meta = MetaData()
        with engine.begin() as conn:
            table = Table("model_metrics", meta, autoload_with=conn)
            cols = {c.name for c in table.columns}
            filtered = {k: v for k, v in values.items() if k in cols}
            conn.execute(table.insert().values(**filtered))
        return metric_id
    except Exception as exc:
        LOGGER.warning("model_metrics insert skipped: %s", exc)
        return False


def try_insert_forecasts(
    predictions_path: Path | str,
    *,
    model_metrics_id: UUID | bool | None = None,
    split: str = "test",
    database_url: str | None = None,
) -> dict[str, Any]:
    """Insert test/val predictions into `forecasts`. Never fail the training CLI."""
    import pandas as pd

    payload: dict[str, Any] = {"attempted": 0, "inserted": 0, "ok": False}
    url = database_url if database_url is not None else os.environ.get("DATABASE_URL")
    if not url:
        LOGGER.info("DATABASE_URL unset; skipping forecasts insert")
        payload["reason"] = "DATABASE_URL unset"
        return payload
    path = Path(predictions_path)
    if not path.is_file():
        payload["reason"] = f"missing {path}"
        return payload
    try:
        from sqlalchemy import MetaData, Table, create_engine
    except ImportError:
        LOGGER.warning("sqlalchemy not installed; skipping forecasts insert")
        payload["reason"] = "sqlalchemy missing"
        return payload

    frame = pd.read_csv(path)
    if "building_id" not in frame.columns or "ts" not in frame.columns or "y_pred" not in frame.columns:
        payload["reason"] = "predictions file missing building_id/ts/y_pred"
        return payload
    frame = frame.loc[frame["building_id"].astype(str) != "_all"].copy()
    frame = frame.loc[frame["y_pred"].notna()].copy()
    payload["attempted"] = int(len(frame))
    if frame.empty:
        payload["ok"] = True
        payload["reason"] = "no rows"
        return payload

    metric_fk = model_metrics_id if isinstance(model_metrics_id, UUID) else None
    try:
        from ml.ingest.db import REPO_ROOT as _repo

        import sys

        backend = str(_repo / "backend")
        if backend not in sys.path:
            sys.path.insert(0, backend)
        from app.db.base import Base
        from app.models import load_all_models

        load_all_models()
        engine = create_engine(url, connect_args=_sqlite_connect_args(url), future=True)
        Base.metadata.create_all(bind=engine)
        meta = MetaData()
        with engine.begin() as conn:
            buildings_table = Table("buildings", meta, autoload_with=conn)
            for building_id in sorted({str(v) for v in frame["building_id"].astype(str).unique()}):
                existing = conn.execute(
                    buildings_table.select().where(buildings_table.c.id == building_id)
                ).first()
                if existing is None:
                    conn.execute(
                        buildings_table.insert().values(
                            id=building_id,
                            site_id="unknown",
                            timezone="UTC",
                        )
                    )
            table = Table("forecasts", meta, autoload_with=conn)
            cols = {c.name for c in table.columns}
            if metric_fk is not None:
                conn.execute(table.delete().where(table.c.model_metrics_id == str(metric_fk)))
            batch: list[dict[str, Any]] = []
            inserted = 0
            for rec in frame.to_dict(orient="records"):
                y_true = rec.get("y_true")
                row = {
                    "id": str(uuid4()),
                    "building_id": str(rec["building_id"]),
                    "model_metrics_id": str(metric_fk) if metric_fk is not None else None,
                    "target_ts": _parse_ts(str(rec["ts"])),
                    "y_true": None if y_true is None or (isinstance(y_true, float) and pd.isna(y_true)) else float(y_true),
                    "y_pred": float(rec["y_pred"]),
                    "split": split,
                }
                batch.append({k: v for k, v in row.items() if k in cols})
                if len(batch) >= 1000:
                    conn.execute(table.insert(), batch)
                    inserted += len(batch)
                    batch = []
            if batch:
                conn.execute(table.insert(), batch)
                inserted += len(batch)
            payload["inserted"] = inserted
            payload["ok"] = True
        engine.dispose()
        return payload
    except Exception as exc:
        LOGGER.warning("forecasts insert skipped: %s", exc)
        payload["reason"] = str(exc)
        return payload
