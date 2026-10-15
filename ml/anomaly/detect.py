"""Fit Isolation Forest on chronological train only; score other splits.

This layer is independent of BDG2 S-H-ESD / long-zero cleaning: those already
replaced some outliers with NaN before Phase 2. Isolation Forest may still flag
unusual remaining patterns or forecast residuals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ml.anomaly.config import AnomalySettings, classify_severity, deviation_percentage, load_anomaly_settings
from ml.anomaly.features import (
    as_float_matrix,
    drop_required_missing,
    ensure_meta,
    group_keys,
    impute_optional,
    mask_group,
    missing_report,
    resolve_if_columns,
    train_medians,
)
from ml.anomaly.residuals import attach_expected
from ml.models.base import json_safe, set_random_seeds

LOGGER = logging.getLogger("ml.anomaly")

BDG2_CLEANING_CAVEAT = (
    "BDG2 cleaned electricity already replaced S-H-ESD outliers, zero runs longer "
    "than 24h, and electricity zeros with NaN. Isolation Forest is an independent "
    "second layer on the remaining series (plus forecast residuals); it does not "
    "re-run BDG2 cleaning and may flag unusual-but-real occupancy or weather patterns."
)


@dataclass
class GroupDetector:
    building_id: str
    data_origin: str
    scaler: StandardScaler
    forest: IsolationForest
    feature_names: list[str]
    medians: dict[str, float]
    n_train: int


@dataclass
class AnomalyRunResult:
    scores: pd.DataFrame
    summary: dict[str, Any]
    detectors: dict[tuple[str, str], GroupDetector] = field(default_factory=dict)
    settings: AnomalySettings | None = None


def _prepare_split(frame: pd.DataFrame, split_label: str) -> pd.DataFrame:
    out = ensure_meta(frame)
    if "split" not in frame.columns or out["split"].astype(str).eq("").all():
        out["split"] = split_label
    return out


def _optional_impute_cols(feature_cols: list[str]) -> list[str]:
    required = {"energy_kwh", "hour"}
    return [col for col in feature_cols if col not in required]


def _fit_group(
    train_g: pd.DataFrame,
    feature_cols: list[str],
    settings: AnomalySettings,
) -> GroupDetector | None:
    if len(train_g) < settings.min_train_rows:
        LOGGER.warning(
            "skip IF fit for building=%s origin=%s: n_train=%s < min_train_rows=%s",
            train_g["building_id"].iloc[0] if len(train_g) else "?",
            train_g["data_origin"].iloc[0] if len(train_g) else "?",
            len(train_g),
            settings.min_train_rows,
        )
        return None
    medians = train_medians(train_g, feature_cols)
    filled = impute_optional(train_g, _optional_impute_cols(feature_cols), medians)
    x = as_float_matrix(filled, feature_cols)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    forest = IsolationForest(
        n_estimators=settings.n_estimators,
        contamination=settings.contamination,
        max_samples=settings.max_samples,
        random_state=settings.random_seed,
        n_jobs=1,
    )
    forest.fit(x_scaled)
    return GroupDetector(
        building_id=str(train_g["building_id"].iloc[0]),
        data_origin=str(train_g["data_origin"].iloc[0]),
        scaler=scaler,
        forest=forest,
        feature_names=list(feature_cols),
        medians=medians,
        n_train=len(train_g),
    )


def _score_group(part: pd.DataFrame, detector: GroupDetector, settings: AnomalySettings) -> pd.DataFrame:
    filled = impute_optional(part, _optional_impute_cols(detector.feature_names), detector.medians)
    x = as_float_matrix(filled, detector.feature_names)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_scaled = detector.scaler.transform(x)
    decision = np.asarray(detector.forest.decision_function(x_scaled), dtype=np.float64).reshape(-1)
    pred = np.asarray(detector.forest.predict(x_scaled), dtype=np.int32).reshape(-1)
    rows = filled.copy()
    rows["score"] = decision
    rows["if_outlier"] = pred == -1
    floor = settings.expected_floor_kwh
    actual = rows["actual_kwh"].to_numpy(dtype=np.float64)
    expected = rows["expected_kwh"].to_numpy(dtype=np.float64)
    dev = np.array(
        [deviation_percentage(a, e, floor=floor) for a, e in zip(actual, expected, strict=True)],
        dtype=np.float64,
    )
    rows["deviation_pct"] = dev
    flags: list[bool] = []
    severities: list[str] = []
    for outlier, pct, sc in zip(rows["if_outlier"].tolist(), dev.tolist(), decision.tolist(), strict=True):
        is_anom, sev = classify_severity(
            if_outlier=bool(outlier),
            deviation_pct=float(pct),
            score=float(sc),
            settings=settings,
        )
        flags.append(is_anom)
        severities.append(sev)
    rows["is_anomaly"] = flags
    rows["severity"] = severities
    return rows


def run_anomaly_detection(
    train: pd.DataFrame,
    val: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    *,
    settings: AnomalySettings | None = None,
    artifacts_dir: Path | None = None,
    processed_source: str = "unknown",
    score_train: bool = False,
    prefer_saved_forecaster: bool = True,
) -> AnomalyRunResult:
    """Fit IF on train feature space only; score val/test (and train if requested)."""
    cfg = settings or load_anomaly_settings()
    set_random_seeds(cfg.random_seed)

    train_m = _prepare_split(train, "train")
    parts = [("train", train_m)]
    if val is not None and len(val):
        parts.append(("val", _prepare_split(val, "val")))
    if test is not None and len(test):
        parts.append(("test", _prepare_split(test, "test")))

    combined = pd.concat([frame for _label, frame in parts], ignore_index=True)
    combined, n_required_dropped = drop_required_missing(combined)
    train_kept = combined.loc[combined["split"].astype(str) == "train"].copy()
    if train_kept.empty:
        raise ValueError("no train rows remain after dropping missing energy_kwh/hour")

    combined, residual_meta = attach_expected(
        train_kept,
        combined,
        artifacts_dir=artifacts_dir,
        processed_source=processed_source,
        prefer_saved_forecaster=prefer_saved_forecaster,
    )
    train_kept = combined.loc[combined["split"].astype(str) == "train"].copy()
    feature_cols = resolve_if_columns(combined.columns, cfg)
    if "residual" in cfg.feature_names and "residual" in combined.columns and "residual" not in feature_cols:
        feature_cols.append("residual")
    feature_cols = [
        col
        for col in feature_cols
        if col == "residual" or (col in train_kept.columns and train_kept[col].notna().any())
    ]
    if not feature_cols:
        raise ValueError("no Isolation Forest features available")

    optional_cols = _optional_impute_cols(feature_cols)
    detectors: dict[tuple[str, str], GroupDetector] = {}
    for building_id, origin in group_keys(train_kept):
        mask = mask_group(train_kept, building_id, origin)
        detector = _fit_group(train_kept.loc[mask], feature_cols, cfg)
        if detector is not None:
            detectors[(building_id, origin)] = detector

    if not detectors:
        raise ValueError("Isolation Forest was not fit; not enough train rows per building/origin")

    historical_fallback = {key[0]: det for key, det in detectors.items() if key[1] == "historical"}

    scored_parts: list[pd.DataFrame] = []
    skipped_groups: list[dict[str, Any]] = []
    score_labels = {label for label, _frame in parts}
    if not score_train:
        score_labels.discard("train")

    to_score = combined.loc[combined["split"].astype(str).isin(score_labels)].copy()
    for building_id, origin in group_keys(to_score):
        mask = mask_group(to_score, building_id, origin)
        part = to_score.loc[mask]
        detector = detectors.get((building_id, origin))
        if detector is None and origin != "historical":
            detector = historical_fallback.get(building_id)
            if detector is not None:
                LOGGER.info(
                    "scoring origin=%s for %s with historical IF (no train rows for that origin)",
                    origin,
                    building_id,
                )
        if detector is None:
            skipped_groups.append(
                {"building_id": building_id, "data_origin": origin, "n": int(len(part)), "reason": "no_fitted_if"}
            )
            continue
        scored_parts.append(_score_group(part, detector, cfg))

    if scored_parts:
        scores = pd.concat(scored_parts, ignore_index=True)
        scores = scores.sort_values(["building_id", "ts"], kind="mergesort").reset_index(drop=True)
    else:
        scores = pd.DataFrame()

    n_scored = int(len(scores))
    n_flagged = int(scores["is_anomaly"].sum()) if n_scored else 0
    by_severity: dict[str, int] = {}
    by_origin: dict[str, int] = {}
    by_building: dict[str, dict[str, int]] = {}
    if n_scored:
        by_severity = {str(k): int(v) for k, v in scores["severity"].value_counts().items()}
        by_origin = {str(k): int(v) for k, v in scores["data_origin"].value_counts().items()}
        for bid, grp in scores.groupby("building_id"):
            by_building[str(bid)] = {
                "n_scored": int(len(grp)),
                "n_flagged": int(grp["is_anomaly"].sum()),
            }

    summary = {
        "data_source": processed_source,
        "method": "isolation_forest",
        "fit": "train_only_per_building_origin",
        "score_splits": sorted(score_labels),
        "bdg2_cleaning_caveat": BDG2_CLEANING_CAVEAT,
        "missing_rule": cfg.missing_rule,
        "n_dropped_missing_required": n_required_dropped,
        "if_features": feature_cols,
        "contamination": cfg.contamination,
        "random_seed": cfg.random_seed,
        "n_estimators": cfg.n_estimators,
        "n_groups_fitted": len(detectors),
        "n_train_fit": {f"{k[0]}|{k[1]}": v.n_train for k, v in detectors.items()},
        "n_scored": n_scored,
        "n_flagged": n_flagged,
        "by_severity": by_severity,
        "by_origin": by_origin,
        "by_building": by_building,
        "skipped_groups": skipped_groups,
        "missing_before_impute": missing_report(combined, feature_cols),
        "optional_impute_columns": optional_cols,
        "severity_config": {
            "warning_deviation_pct": cfg.warning_deviation_pct,
            "critical_deviation_pct": cfg.critical_deviation_pct,
            "warning_score_cutoff": cfg.warning_score_cutoff,
            "critical_score_cutoff": cfg.critical_score_cutoff,
            "flag_if_outlier": cfg.flag_if_outlier,
            "expected_floor_kwh": cfg.expected_floor_kwh,
        },
        "residual": json_safe(residual_meta),
        "settings": cfg.as_dict(),
    }
    return AnomalyRunResult(scores=scores, summary=summary, detectors=detectors, settings=cfg)
