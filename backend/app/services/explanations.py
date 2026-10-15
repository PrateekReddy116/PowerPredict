"""Load stored SHAP JSON. The API never computes KernelSHAP and never returns dummy arrays."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pandas import Timestamp


def explanations_root(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / "explanations"


def global_path(artifacts_dir: Path, model_name: str) -> Path:
    return explanations_root(artifacts_dir) / model_name / "global.json"


def local_path(artifacts_dir: Path, model_name: str) -> Path:
    return explanations_root(artifacts_dir) / model_name / "local.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_global(artifacts_dir: Path, model_name: str) -> dict[str, Any] | None:
    path = global_path(artifacts_dir, model_name)
    if not path.exists():
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None
    if "ranking" not in payload or "explainer" not in payload:
        return None
    return payload


def load_locals(artifacts_dir: Path, model_name: str) -> list[dict[str, Any]] | None:
    path = local_path(artifacts_dir, model_name)
    if not path.exists():
        return None
    payload = load_json(path)
    if not isinstance(payload, list):
        return None
    return payload


def _parse_ts(value: str | datetime | None) -> Timestamp | None:
    if value is None or value == "":
        return None
    stamp = Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


def find_local(
    items: list[dict[str, Any]],
    *,
    building_id: str | None = None,
    ts: str | None = None,
    row_index: int | None = None,
) -> dict[str, Any] | None:
    if not items:
        return None
    if row_index is not None:
        for item in items:
            if item.get("row_index") == row_index:
                return item
        if 0 <= row_index < len(items):
            return items[row_index]
        return None

    ts_q = _parse_ts(ts) if ts else None
    matched: list[dict[str, Any]] = []
    for item in items:
        if building_id is not None and str(item.get("building_id") or "") != str(building_id):
            continue
        if ts_q is not None:
            item_ts = _parse_ts(item.get("ts"))
            if item_ts is None or item_ts != ts_q:
                continue
        matched.append(item)
    if building_id is None and ts_q is None:
        return items[0]
    return matched[0] if matched else None
