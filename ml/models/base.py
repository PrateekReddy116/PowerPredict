"""Shared forecaster contract, seeding, and artifact helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Forecaster(Protocol):
    """Every Phase 3 model implements fit / predict / save / load."""

    name: str

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> Forecaster: ...


def set_random_seeds(seed: int) -> None:
    """Python, NumPy, and Torch (including CUDA if present)."""
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "sklearn", "xgboost", "torch", "shap"):
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not-installed"
    return versions


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def as_1d(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64).reshape(-1)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
