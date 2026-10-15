"""Expected consumption and residuals — never invented.

Prefer a compatible saved tabular forecaster (same data source). Otherwise fit a
per-building LinearRegression on TRAIN only to predict energy_kwh from non-leaky
features (lags, hour, temperature, …). Isolation Forest is still fit on train only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ml.models.dataset import META_COLUMNS, select_feature_columns
from ml.pipeline import FEATURE_COLUMNS

LOGGER = logging.getLogger("ml.anomaly.residuals")

# Do not use current energy or the next-hour target as baseline inputs.
BASELINE_EXCLUDE = {
    "energy_kwh",
    "target_energy_kwh",
    "y_next",
    "residual",
    "expected_kwh",
    "actual_kwh",
    "deviation_pct",
}


def _finite_enough(frame: pd.DataFrame, col: str, *, min_frac: float = 0.3) -> bool:
    series = pd.to_numeric(frame[col], errors="coerce")
    if len(series) == 0:
        return False
    return float(series.notna().mean()) >= min_frac


def baseline_feature_columns(frame: pd.DataFrame) -> list[str]:
    intended = [col for col in FEATURE_COLUMNS if col in frame.columns and col not in BASELINE_EXCLUDE]
    if not intended:
        skip = set(META_COLUMNS) | BASELINE_EXCLUDE
        intended = [
            col
            for col in frame.columns
            if col not in skip and pd.api.types.is_numeric_dtype(frame[col])
        ]
    return [col for col in intended if _finite_enough(frame, col)]


def _metrics_source(artifacts_dir: Path) -> str:
    path = artifacts_dir / "metrics.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("data_source") or "")


def artifacts_compatible(artifacts_dir: Path | None, processed_source: str) -> bool:
    if artifacts_dir is None or not artifacts_dir.exists():
        return False
    art = _metrics_source(artifacts_dir).lower()
    src = processed_source.lower()
    if not art:
        return False
    fixture_art = "fixture" in art or "synthetic" in art
    fixture_src = "fixture" in src or "synthetic" in src
    if fixture_art != fixture_src:
        return False
    bdg2_art = "bdg2" in art
    bdg2_src = "bdg2" in src
    if bdg2_art and not bdg2_src:
        return False
    if bdg2_src and fixture_art:
        return False
    return True


def _prefer_model_dir(artifacts_dir: Path) -> Path | None:
    for name in ("xgboost", "random_forest", "linear_regression"):
        path = artifacts_dir / name / "model.joblib"
        if path.exists():
            return artifacts_dir / name
    return None


def try_predict_with_saved_forecaster(
    train: pd.DataFrame,
    score_frame: pd.DataFrame,
    artifacts_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Predict y_{t+1} with a Phase 3 tabular artifact. None if incompatible or load fails."""
    from ml.models import FORECASTER_CLASSES, is_sequence_model

    model_dir = _prefer_model_dir(artifacts_dir)
    scaler_path = artifacts_dir / "feature_scaler.joblib"
    if model_dir is None or not scaler_path.exists():
        return None
    name = model_dir.name
    if is_sequence_model(name):
        return None
    try:
        import joblib

        bundle = joblib.load(scaler_path)
        scaler = bundle["scaler"] if isinstance(bundle, dict) else bundle
        feature_names = list(bundle.get("feature_names") or []) if isinstance(bundle, dict) else []
        if not feature_names:
            feature_names = select_feature_columns(train)
        missing = [col for col in feature_names if col not in score_frame.columns]
        if missing:
            LOGGER.info("saved forecaster skipped; missing columns: %s", missing[:8])
            return None
        cls = FORECASTER_CLASSES[name]
        model = cls.load(model_dir / "model.joblib")
        x = score_frame.loc[:, feature_names].to_numpy(dtype=np.float64)
        x_scaled = scaler.transform(x)
        pred = np.asarray(model.predict(x_scaled), dtype=np.float64).reshape(-1)
        if pred.shape[0] != len(score_frame):
            return None
        meta = {
            "expected_source": f"saved_forecaster:{name}",
            "artifact_path": (model_dir / "model.joblib").as_posix(),
            "predicts": "target_energy_kwh",
        }
        return pred, meta
    except Exception as exc:  # noqa: BLE001 — best-effort artifact load
        LOGGER.info("saved forecaster unused: %s", exc)
        return None


def fit_energy_baseline(
    train: pd.DataFrame,
    score_frame: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-(building, origin) LinearRegression on train → energy_kwh. No test leakage."""
    cols = feature_cols or baseline_feature_columns(train)
    if not cols:
        raise ValueError("no numeric baseline features available to estimate expected kWh")
    expected = np.full(len(score_frame), np.nan, dtype=np.float64)
    models_fitted = 0
    for (building_id, origin), train_g in train.groupby(["building_id", "data_origin"], sort=False):
        usable = train_g.dropna(subset=[*cols, "energy_kwh"])
        if len(usable) < 2:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(usable.loc[:, cols].to_numpy(dtype=np.float64))
        y_train = usable["energy_kwh"].to_numpy(dtype=np.float64)
        model = LinearRegression()
        model.fit(x_train, y_train)
        models_fitted += 1
        mask = (score_frame["building_id"].astype(str) == str(building_id)) & (
            score_frame["data_origin"].astype(str) == str(origin)
        )
        part = score_frame.loc[mask]
        if part.empty:
            continue
        x = part.loc[:, cols].to_numpy(dtype=np.float64)
        # Train-median fill for any leftover NaNs so predict does not crash.
        col_means = np.nanmean(usable.loc[:, cols].to_numpy(dtype=np.float64), axis=0)
        nan_mask = np.isnan(x)
        if nan_mask.any():
            x = x.copy()
            x[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        pred = model.predict(scaler.transform(x))
        expected[mask.to_numpy()] = pred
    meta = {
        "expected_source": "train_only_linear_regression_per_building",
        "baseline_features": cols,
        "n_baselines_fitted": models_fitted,
        "predicts": "energy_kwh",
    }
    return expected, meta


def attach_expected(
    train: pd.DataFrame,
    score_frame: pd.DataFrame,
    *,
    artifacts_dir: Path | None = None,
    processed_source: str = "unknown",
    prefer_saved_forecaster: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add actual_kwh, expected_kwh, residual. Drops rows with no expected (never invents)."""
    out = score_frame.copy()
    used_saved = False
    meta: dict[str, Any] = {}
    if prefer_saved_forecaster and artifacts_dir is not None and artifacts_compatible(artifacts_dir, processed_source):
        loaded = try_predict_with_saved_forecaster(train, out, artifacts_dir)
        if loaded is not None:
            pred, meta = loaded
            if "target_energy_kwh" in out.columns:
                actual = pd.to_numeric(out["target_energy_kwh"], errors="coerce").to_numpy(dtype=np.float64)
            else:
                actual = pd.to_numeric(out["energy_kwh"], errors="coerce").to_numpy(dtype=np.float64)
            out["actual_kwh"] = actual
            out["expected_kwh"] = pred
            used_saved = True
    if not used_saved:
        pred, meta = fit_energy_baseline(train, out)
        out["actual_kwh"] = pd.to_numeric(out["energy_kwh"], errors="coerce").to_numpy(dtype=np.float64)
        out["expected_kwh"] = pred
    out["residual"] = out["actual_kwh"] - out["expected_kwh"]
    keep = out["expected_kwh"].notna() & out["actual_kwh"].notna()
    n_dropped = int((~keep).sum())
    out = out.loc[keep].copy()
    meta["n_rows_without_expected"] = n_dropped
    meta["used_saved_forecaster"] = used_saved
    return out, meta
