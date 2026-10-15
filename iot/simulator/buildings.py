"""Resolve which buildings the simulator publishes for.

Preference: CLI ids → pipeline_summary.json → download_manifest.json → documented demo ids.
"""

from __future__ import annotations

import json
from pathlib import Path

from iot.env import REPO_ROOT

# Documented fallbacks when the Phase 2 pipeline has not been run.
DEMO_BUILDING_IDS = ("building_001", "building_002", "building_003")


def _ids_from_pipeline_summary(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    ids = data.get("building_ids") or []
    return [str(item) for item in ids if item]


def _ids_from_manifest(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    ids = data.get("selected_building_ids") or []
    return [str(item) for item in ids if item]


def resolve_building_ids(
    explicit: list[str] | None = None,
    *,
    repo_root: Path | None = None,
) -> list[str]:
    if explicit:
        return [str(item).strip() for item in explicit if str(item).strip()]
    root = repo_root or REPO_ROOT
    from_pipeline = _ids_from_pipeline_summary(root / "data" / "processed" / "pipeline_summary.json")
    if from_pipeline:
        return from_pipeline
    from_manifest = _ids_from_manifest(root / "data" / "raw" / "download_manifest.json")
    if from_manifest:
        return from_manifest
    return list(DEMO_BUILDING_IDS)
