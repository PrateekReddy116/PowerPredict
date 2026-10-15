"""Write anomaly artifacts. Best-effort Postgres/SQLite insert — never fail the CLI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from ml.evaluate.persist import save_metrics_json
from ml.models.base import json_safe

LOGGER = logging.getLogger("ml.anomaly.persist")

SCORE_COLUMNS = [
    "building_id",
    "ts",
    "split",
    "data_origin",
    "score",
    "is_anomaly",
    "if_outlier",
    "deviation_pct",
    "severity",
    "actual_kwh",
    "expected_kwh",
    "residual",
    "energy_kwh",
]


def write_anomaly_artifacts(
    scores: pd.DataFrame,
    summary: dict[str, Any],
    dest: Path,
    *,
    detectors: dict | None = None,
) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    keep = [col for col in SCORE_COLUMNS if col in scores.columns]
    table = scores.loc[:, keep].copy() if keep else scores.copy()
    parquet_path = dest / "scores.parquet"
    csv_path = dest / "scores.csv"
    try:
        table.to_parquet(parquet_path, index=False)
        written["scores"] = parquet_path.as_posix()
    except Exception:
        table.to_csv(csv_path, index=False)
        written["scores"] = csv_path.as_posix()
        parquet_path = csv_path
    json_path = dest / "summary.json"
    save_metrics_json(json_safe(summary), json_path)
    written["summary"] = json_path.as_posix()
    if detectors:
        try:
            import joblib

            payload = {
                key: {
                    "building_id": det.building_id,
                    "data_origin": det.data_origin,
                    "scaler": det.scaler,
                    "forest": det.forest,
                    "feature_names": det.feature_names,
                    "medians": det.medians,
                    "n_train": det.n_train,
                }
                for key, det in detectors.items()
            }
            model_path = dest / "isolation_forest.joblib"
            joblib.dump(payload, model_path)
            written["model"] = model_path.as_posix()
        except Exception as exc:
            LOGGER.warning("could not pickle Isolation Forest: %s", exc)
    return written


def _parse_ts(value: object):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def try_insert_anomalies(
    scores: pd.DataFrame,
    *,
    database_url: str | None = None,
    buildings: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Replace anomaly rows for scored buildings. Returns status; never raises to callers."""
    url = database_url if database_url is not None else os.environ.get("DATABASE_URL")
    payload: dict[str, Any] = {"attempted": int(len(scores)), "inserted": 0, "ok": False}
    if not url:
        LOGGER.info("DATABASE_URL unset; skipping anomalies insert")
        payload["reason"] = "DATABASE_URL unset"
        return payload
    if scores.empty:
        payload["ok"] = True
        payload["reason"] = "no rows"
        return payload
    try:
        from sqlalchemy import MetaData, Table, create_engine, delete
    except ImportError:
        LOGGER.warning("sqlalchemy not installed; skipping anomalies insert")
        payload["reason"] = "sqlalchemy missing"
        return payload

    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    try:
        from ml.ingest.db import REPO_ROOT as _repo

        import sys

        backend = str(_repo / "backend")
        if backend not in sys.path:
            sys.path.insert(0, backend)
        from app.db.base import Base
        from app.models import load_all_models

        load_all_models()
        engine = create_engine(url, connect_args=connect_args, future=True)
        Base.metadata.create_all(bind=engine)
        meta = MetaData()
        with engine.begin() as conn:
            if buildings is not None and len(buildings):
                try:
                    buildings_table = Table("buildings", meta, autoload_with=conn)
                    bcols = {c.name for c in buildings_table.columns}
                    for rec in buildings.to_dict(orient="records"):
                        row = {
                            "id": str(rec.get("id") or rec.get("building_id")),
                            "site_id": str(rec.get("site_id") or "unknown"),
                            "primary_use": rec.get("primary_use"),
                            "sqm": rec.get("sqm"),
                            "timezone": str(rec.get("timezone") or "UTC"),
                            "selected_reason": rec.get("selected_reason"),
                        }
                        row = {k: v for k, v in row.items() if k in bcols}
                        existing = conn.execute(
                            buildings_table.select().where(buildings_table.c.id == row["id"])
                        ).first()
                        if existing is None:
                            conn.execute(buildings_table.insert().values(**row))
                except Exception as exc:
                    LOGGER.warning("building upsert skipped: %s", exc)

            table = Table("anomalies", meta, autoload_with=conn)
            cols = {c.name for c in table.columns}
            building_ids = sorted({str(v) for v in scores["building_id"].astype(str).unique()})
            conn.execute(delete(table).where(table.c.building_id.in_(building_ids)))
            inserted = 0
            batch: list[dict[str, Any]] = []
            for rec in scores.to_dict(orient="records"):
                row = {
                    "id": str(uuid4()),
                    "building_id": str(rec["building_id"]),
                    "ts": _parse_ts(rec["ts"]),
                    "score": float(rec["score"]),
                    "is_anomaly": bool(rec["is_anomaly"]),
                    "model_metrics_id": rec.get("model_metrics_id"),
                    "data_origin": str(rec.get("data_origin") or "historical"),
                    "deviation_pct": None if rec.get("deviation_pct") is None else float(rec["deviation_pct"]),
                    "severity": rec.get("severity"),
                    "actual_kwh": None if rec.get("actual_kwh") is None else float(rec["actual_kwh"]),
                    "expected_kwh": None if rec.get("expected_kwh") is None else float(rec["expected_kwh"]),
                }
                filtered = {k: v for k, v in row.items() if k in cols}
                batch.append(filtered)
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
        LOGGER.warning("anomalies insert skipped: %s", exc)
        payload["reason"] = str(exc)
        return payload
