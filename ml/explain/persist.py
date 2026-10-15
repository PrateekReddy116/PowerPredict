"""Write explanation JSON under artifacts/<run>/explanations/. Optional DB insert."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ml.models.base import json_safe

LOGGER = logging.getLogger(__name__)

GLOBAL_FILENAME = "global.json"
LOCAL_FILENAME = "local.json"
MANIFEST_FILENAME = "manifest.json"


def explanations_dir(artifacts_dir: Path, model_name: str) -> Path:
    path = Path(artifacts_dir) / "explanations" / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict[str, Any] | list[Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_model_explanations(
    artifacts_dir: Path,
    model_name: str,
    *,
    global_payload: dict[str, Any],
    local_payloads: list[dict[str, Any]],
) -> dict[str, Path]:
    dest = explanations_dir(artifacts_dir, model_name)
    written = {
        "global": save_json(global_payload, dest / GLOBAL_FILENAME),
        "local": save_json(local_payloads, dest / LOCAL_FILENAME),
    }
    return written


def write_manifest(artifacts_dir: Path, payload: dict[str, Any]) -> Path:
    dest = Path(artifacts_dir) / "explanations"
    dest.mkdir(parents=True, exist_ok=True)
    return save_json(payload, dest / MANIFEST_FILENAME)


def try_insert_explanation(
    *,
    method: str,
    payload: dict[str, Any],
    database_url: str | None = None,
) -> bool:
    """Insert one explanations row. Returns False if DATABASE_URL is missing or the DB is down."""
    url = database_url if database_url is not None else os.environ.get("DATABASE_URL")
    if not url:
        LOGGER.info("DATABASE_URL unset; skipping explanations insert")
        return False
    try:
        from sqlalchemy import MetaData, Table, create_engine
    except ImportError:
        LOGGER.warning("sqlalchemy not installed; skipping explanations insert")
        return False
    values = {
        "id": str(uuid4()),
        "model_metrics_id": None,
        "forecast_id": None,
        "method": method,
        "payload_json": json.dumps(json_safe(payload)),
        "created_at": datetime.now(UTC),
    }
    try:
        engine = create_engine(url)
        meta = MetaData()
        with engine.begin() as conn:
            table = Table("explanations", meta, autoload_with=conn)
            cols = {c.name for c in table.columns}
            filtered = {k: v for k, v in values.items() if k in cols}
            conn.execute(table.insert().values(**filtered))
        return True
    except Exception as exc:
        LOGGER.warning("explanations insert skipped: %s", exc)
        return False
