"""Callable scoring helpers. FastAPI must not train; it loads stored anomaly rows.

Use `run_anomaly_detection` from the offline CLI. The API reuses `classify_severity`
so severity bands stay configurable (env/YAML), never hardcoded in the UI.
"""

from ml.anomaly.config import AnomalySettings, classify_severity, load_anomaly_settings
from ml.anomaly.detect import run_anomaly_detection

__all__ = [
    "AnomalySettings",
    "classify_severity",
    "load_anomaly_settings",
    "run_anomaly_detection",
]
