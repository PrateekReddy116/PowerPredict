"""Isolation Forest and severity settings from YAML plus environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.config import load_config
from ml.paths import DEFAULT_CONFIG

DEFAULT_IF_FEATURES: tuple[str, ...] = (
    "energy_kwh",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "lag_168",
    "hour",
    "air_temperature",
    "residual",
)

SEVERITY_NORMAL = "normal"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"
SEVERITY_VALUES = (SEVERITY_NORMAL, SEVERITY_WARNING, SEVERITY_CRITICAL)


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return float(raw)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return int(raw)


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AnomalySettings:
    contamination: float = 0.01
    n_estimators: int = 100
    max_samples: int | str = "auto"
    random_seed: int = 42
    min_train_rows: int = 24
    expected_floor_kwh: float = 1.0
    flag_if_outlier: bool = True
    warning_deviation_pct: float = 25.0
    critical_deviation_pct: float = 50.0
    warning_score_cutoff: float | None = None
    critical_score_cutoff: float | None = None
    feature_names: tuple[str, ...] = DEFAULT_IF_FEATURES
    missing_rule: str = "drop_required_impute_optional"

    def as_dict(self) -> dict[str, Any]:
        return {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "random_seed": self.random_seed,
            "min_train_rows": self.min_train_rows,
            "expected_floor_kwh": self.expected_floor_kwh,
            "flag_if_outlier": self.flag_if_outlier,
            "warning_deviation_pct": self.warning_deviation_pct,
            "critical_deviation_pct": self.critical_deviation_pct,
            "warning_score_cutoff": self.warning_score_cutoff,
            "critical_score_cutoff": self.critical_score_cutoff,
            "feature_names": list(self.feature_names),
            "missing_rule": self.missing_rule,
        }


def _block(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(key) or {}
    return cur if isinstance(cur, dict) else {}


def load_anomaly_settings(
    path: Path | str | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> AnomalySettings:
    """YAML `anomaly:` block, then env overrides. Never hardcodes dashboard flags."""
    cfg = config if config is not None else load_config(path or DEFAULT_CONFIG)
    block = cfg.get("anomaly") if isinstance(cfg.get("anomaly"), dict) else {}
    flag = _block(block, "flag")
    severity = _block(block, "severity")
    features = block.get("features")
    if isinstance(features, list) and features:
        feature_names = tuple(str(item) for item in features)
    else:
        feature_names = DEFAULT_IF_FEATURES

    max_samples_raw = block.get("max_samples", "auto")
    if isinstance(max_samples_raw, (int, float)) and not isinstance(max_samples_raw, bool):
        max_samples: int | str = int(max_samples_raw)
    else:
        max_samples = "auto"

    warning_dev = float(
        severity.get("warning_deviation_pct", flag.get("warning_deviation_pct", 25.0))
    )
    critical_dev = float(severity.get("critical_deviation_pct", 50.0))
    warning_score = severity.get("warning_score_cutoff")
    critical_score = severity.get("critical_score_cutoff")
    seed = int(block.get("random_seed") or cfg.get("random_seed") or 42)

    settings = AnomalySettings(
        contamination=float(block.get("contamination", 0.01)),
        n_estimators=int(block.get("n_estimators", 100)),
        max_samples=max_samples,
        random_seed=seed,
        min_train_rows=int(block.get("min_train_rows", 24)),
        expected_floor_kwh=float(block.get("expected_floor_kwh", 1.0)),
        flag_if_outlier=bool(flag.get("isolation_forest_outlier", True)),
        warning_deviation_pct=warning_dev,
        critical_deviation_pct=critical_dev,
        warning_score_cutoff=None if warning_score is None else float(warning_score),
        critical_score_cutoff=None if critical_score is None else float(critical_score),
        feature_names=feature_names,
        missing_rule=str(block.get("missing", "drop_required_impute_optional")),
    )
    return apply_env_overrides(settings)


def apply_env_overrides(settings: AnomalySettings) -> AnomalySettings:
    contamination = _env_float("ANOMALY_CONTAMINATION")
    n_estimators = _env_int("ANOMALY_N_ESTIMATORS")
    seed = _env_int("ANOMALY_RANDOM_SEED")
    min_rows = _env_int("ANOMALY_MIN_TRAIN_ROWS")
    floor = _env_float("ANOMALY_EXPECTED_FLOOR_KWH")
    flag_if = _env_bool("ANOMALY_FLAG_IF_OUTLIER")
    warn_dev = _env_float("ANOMALY_WARNING_DEVIATION_PCT")
    crit_dev = _env_float("ANOMALY_CRITICAL_DEVIATION_PCT")
    warn_score = _env_float("ANOMALY_WARNING_SCORE_CUTOFF")
    crit_score = _env_float("ANOMALY_CRITICAL_SCORE_CUTOFF")
    return AnomalySettings(
        contamination=settings.contamination if contamination is None else contamination,
        n_estimators=settings.n_estimators if n_estimators is None else n_estimators,
        max_samples=settings.max_samples,
        random_seed=settings.random_seed if seed is None else seed,
        min_train_rows=settings.min_train_rows if min_rows is None else min_rows,
        expected_floor_kwh=settings.expected_floor_kwh if floor is None else floor,
        flag_if_outlier=settings.flag_if_outlier if flag_if is None else flag_if,
        warning_deviation_pct=settings.warning_deviation_pct if warn_dev is None else warn_dev,
        critical_deviation_pct=settings.critical_deviation_pct if crit_dev is None else crit_dev,
        warning_score_cutoff=settings.warning_score_cutoff if warn_score is None else warn_score,
        critical_score_cutoff=settings.critical_score_cutoff if crit_score is None else crit_score,
        feature_names=settings.feature_names,
        missing_rule=settings.missing_rule,
    )


def deviation_percentage(actual: float, expected: float, *, floor: float) -> float:
    """(actual - expected) / safe_denom * 100. Denom is max(|expected|, floor)."""
    denom = max(abs(float(expected)), float(floor))
    if denom <= 0.0:
        denom = 1.0
    return 100.0 * (float(actual) - float(expected)) / denom


def classify_severity(
    *,
    if_outlier: bool,
    deviation_pct: float | None,
    score: float,
    settings: AnomalySettings,
) -> tuple[bool, str]:
    """Return (is_anomaly, severity). Bands come from settings, not UI constants."""
    abs_dev = 0.0
    if deviation_pct is not None:
        try:
            if deviation_pct == deviation_pct:  # not NaN
                abs_dev = abs(float(deviation_pct))
        except (TypeError, ValueError):
            abs_dev = 0.0

    flagged = bool(settings.flag_if_outlier and if_outlier) or abs_dev > settings.warning_deviation_pct
    if settings.warning_score_cutoff is not None and score <= settings.warning_score_cutoff:
        flagged = True

    critical = abs_dev >= settings.critical_deviation_pct
    if settings.critical_score_cutoff is not None and score <= settings.critical_score_cutoff:
        critical = True

    if critical:
        return True, SEVERITY_CRITICAL
    if flagged:
        return True, SEVERITY_WARNING
    return False, SEVERITY_NORMAL
