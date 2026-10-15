"""Repository data locations."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
ML_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"
DEFAULT_CONFIG = ML_CONFIG
DEFAULT_PROCESSED_DIR = DATA_PROCESSED
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts"

RAW_METADATA = DATA_RAW / "metadata.csv"
RAW_WEATHER = DATA_RAW / "weather.csv"
RAW_ELECTRICITY = DATA_RAW / "electricity.csv"
RAW_ELECTRICITY_SUBSET = DATA_RAW / "electricity_selected.csv"


def ensure_data_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
