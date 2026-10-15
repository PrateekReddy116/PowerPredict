"""Load ml/configs YAML. Training hyperparameters live here — never metrics."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ml.paths import DEFAULT_CONFIG


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config at {cfg_path} must be a mapping")
    return loaded


def apply_smoke_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Tiny CPU settings for tests and `--smoke`. Does not invent metrics."""
    out = deepcopy(config)
    smoke = out.get("smoke") or {}
    out["seq_len"] = int(smoke.get("seq_len", 8))
    models = out.setdefault("models", {})
    rf = models.setdefault("random_forest", {})
    rf["n_estimators"] = int(smoke.get("n_estimators", 8))
    rf["n_jobs"] = 1
    xgb = models.setdefault("xgboost", {})
    xgb["n_estimators"] = int(smoke.get("n_estimators", 8))
    xgb["n_jobs"] = 1
    xgb["early_stopping_rounds"] = 1
    for name in ("lstm", "cnn_lstm"):
        block = models.setdefault(name, {})
        block["epochs"] = int(smoke.get("epochs", 1))
        block["hidden_size"] = int(smoke.get("hidden_size", 16))
        block["batch_size"] = int(smoke.get("batch_size", 8))
        block["patience"] = int(smoke.get("patience", 1))
        block["num_layers"] = 1
    return out
