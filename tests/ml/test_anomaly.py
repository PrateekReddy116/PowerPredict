"""Isolation Forest: train-only fit, no invented residuals, missing values must not crash."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.anomaly.config import AnomalySettings, classify_severity, deviation_percentage
from ml.anomaly.detect import run_anomaly_detection
from ml.anomaly.persist import try_insert_anomalies, write_anomaly_artifacts
from tests.ml.conftest import make_fixture_frame


def _settings(**overrides) -> AnomalySettings:
    base = AnomalySettings(
        contamination=0.02,
        n_estimators=32,
        random_seed=42,
        min_train_rows=16,
        expected_floor_kwh=1.0,
        warning_deviation_pct=25.0,
        critical_deviation_pct=50.0,
        flag_if_outlier=True,
    )
    return AnomalySettings(
        contamination=overrides.get("contamination", base.contamination),
        n_estimators=overrides.get("n_estimators", base.n_estimators),
        max_samples=base.max_samples,
        random_seed=overrides.get("random_seed", base.random_seed),
        min_train_rows=overrides.get("min_train_rows", base.min_train_rows),
        expected_floor_kwh=overrides.get("expected_floor_kwh", base.expected_floor_kwh),
        flag_if_outlier=overrides.get("flag_if_outlier", base.flag_if_outlier),
        warning_deviation_pct=overrides.get("warning_deviation_pct", base.warning_deviation_pct),
        critical_deviation_pct=overrides.get("critical_deviation_pct", base.critical_deviation_pct),
        warning_score_cutoff=overrides.get("warning_score_cutoff", None),
        critical_score_cutoff=overrides.get("critical_score_cutoff", None),
        feature_names=overrides.get("feature_names", base.feature_names),
        missing_rule=base.missing_rule,
    )


def _split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)
    train = frame.iloc[:i_train].copy()
    val = frame.iloc[i_train:i_val].copy()
    test = frame.iloc[i_val:].copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return train, val, test


def test_deviation_safe_denom():
    assert deviation_percentage(20.0, 0.0, floor=1.0) == pytest.approx(2000.0)
    assert deviation_percentage(110.0, 100.0, floor=1.0) == pytest.approx(10.0)


def test_severity_uses_config_not_hardcoded_bands():
    settings = _settings(warning_deviation_pct=10.0, critical_deviation_pct=30.0, flag_if_outlier=False)
    flagged, sev = classify_severity(if_outlier=False, deviation_pct=12.0, score=0.2, settings=settings)
    assert flagged and sev == "warning"
    flagged, sev = classify_severity(if_outlier=False, deviation_pct=40.0, score=0.2, settings=settings)
    assert flagged and sev == "critical"
    flagged, sev = classify_severity(if_outlier=True, deviation_pct=1.0, score=-0.2, settings=settings)
    assert not flagged and sev == "normal"


def test_in_distribution_readings_not_flagged():
    """Tight diurnal series (no weekly jump): in-distribution hours must not be flagged."""
    n = 336
    ts = pd.date_range("2017-01-02", periods=n, freq="h", tz="UTC")
    hour = ts.hour.to_numpy().astype(float)
    energy = 40.0 + 8.0 * np.sin(2 * np.pi * hour / 24.0)
    frame = pd.DataFrame(
        {
            "ts": ts,
            "building_id": "fixture_office_1",
            "energy_kwh": energy,
            "hour": hour,
            "air_temperature": 12.0 + 3.0 * np.sin(2 * np.pi * hour / 24.0),
            "lag_1": pd.Series(energy).shift(1),
            "lag_24": pd.Series(energy).shift(24),
            "data_origin": "historical",
        }
    )
    train, val, test = _split_frame(frame.dropna(subset=["lag_1", "lag_24"]).reset_index(drop=True))
    result = run_anomaly_detection(
        train,
        val,
        test,
        settings=_settings(
            contamination=0.05,
            n_estimators=40,
            flag_if_outlier=False,
            warning_deviation_pct=25.0,
            feature_names=("energy_kwh", "lag_1", "lag_24", "hour", "air_temperature", "residual"),
        ),
        processed_source="synthetic_fixture",
        prefer_saved_forecaster=False,
        score_train=False,
    )
    assert len(result.scores) > 0
    abs_dev = result.scores["deviation_pct"].abs()
    assert float(abs_dev.median()) < 5.0
    assert float(np.nanmax(abs_dev.to_numpy())) < 15.0
    # In-distribution residual is inside the warning band, so nothing is flagged.
    # (IF contamination can still mark a few envelope points if flag_if_outlier is on.)
    assert int(result.scores["is_anomaly"].sum()) == 0


def test_extreme_readings_are_flagged():
    frame = make_fixture_frame(n=360, seed=1)
    train, val, test = _split_frame(frame)
    test = test.copy()
    spike_idx = test.index[-3:]
    test.loc[spike_idx, "energy_kwh"] = test["energy_kwh"].median() * 12.0
    result = run_anomaly_detection(
        train,
        val,
        test,
        settings=_settings(),
        processed_source="synthetic_fixture",
        prefer_saved_forecaster=False,
        score_train=False,
    )
    spiked = result.scores.loc[result.scores["ts"].isin(test.loc[spike_idx, "ts"])]
    assert len(spiked) == 3
    assert spiked["is_anomaly"].all()
    assert (spiked["severity"].isin(["warning", "critical"])).all()
    assert (spiked["deviation_pct"].abs() > 25.0).all()


def test_missing_values_do_not_crash():
    frame = make_fixture_frame(n=280, seed=2)
    train, val, test = _split_frame(frame)
    train = train.copy()
    val = val.copy()
    test = test.copy()
    train.loc[train.index[5], "air_temperature"] = np.nan
    val.loc[val.index[0], "air_temperature"] = np.nan
    test.loc[test.index[1], "energy_kwh"] = np.nan
    test.loc[test.index[2], "hour"] = np.nan
    result = run_anomaly_detection(
        train,
        val,
        test,
        settings=_settings(),
        processed_source="synthetic_fixture",
        prefer_saved_forecaster=False,
        score_train=False,
    )
    assert result.summary["n_dropped_missing_required"] >= 2
    assert result.summary["n_scored"] == len(result.scores)
    assert result.scores["score"].notna().all()


def test_isolation_forest_fit_on_train_only():
    frame = make_fixture_frame(n=300, seed=3)
    train, val, test = _split_frame(frame)
    test = test.copy()
    test["energy_kwh"] = test["energy_kwh"] + 500.0
    result = run_anomaly_detection(
        train,
        val,
        test,
        settings=_settings(),
        processed_source="synthetic_fixture",
        prefer_saved_forecaster=False,
        score_train=False,
    )
    detector = next(iter(result.detectors.values()))
    energy_idx = detector.feature_names.index("energy_kwh")
    np.testing.assert_allclose(
        detector.scaler.mean_[energy_idx],
        train["energy_kwh"].mean(),
        rtol=0.05,
        atol=1.0,
    )
    full_mean = pd.concat([train, val, test], ignore_index=True)["energy_kwh"].mean()
    assert abs(detector.scaler.mean_[energy_idx] - full_mean) > 20.0
    assert "train" not in set(result.scores["split"].astype(str))


def test_write_artifacts_and_sqlite_insert(tmp_path, monkeypatch):
    frame = make_fixture_frame(n=240, seed=4)
    train, val, test = _split_frame(frame)
    result = run_anomaly_detection(
        train,
        val,
        test,
        settings=_settings(),
        processed_source="synthetic_fixture",
        prefer_saved_forecaster=False,
    )
    dest = tmp_path / "anomaly"
    written = write_anomaly_artifacts(result.scores, result.summary, dest, detectors=result.detectors)
    assert (dest / "summary.json").exists()
    assert "scores" in written
    db_path = tmp_path / "anom.db"
    url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    buildings = pd.DataFrame(
        {
            "id": ["fixture_office_1"],
            "site_id": ["TestSite"],
            "primary_use": ["Office"],
            "sqm": [800.0],
            "timezone": ["UTC"],
            "selected_reason": ["TEST FIXTURE, not BDG2"],
        }
    )
    payload = try_insert_anomalies(result.scores, database_url=url, buildings=buildings)
    assert payload["ok"] is True
    assert payload["inserted"] == len(result.scores)
    from app.core.config import reset_settings_cache
    from app.db.session import reset_engine

    reset_engine()
    reset_settings_cache()


def test_db_insert_false_when_url_missing(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    payload = try_insert_anomalies(pd.DataFrame())
    assert payload["ok"] is False or payload.get("reason") == "DATABASE_URL unset"
