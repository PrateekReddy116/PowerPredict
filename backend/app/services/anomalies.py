"""Query stored Isolation Forest rows. The API does not fit models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.anomaly import Anomaly


def classify_severity_api(
    *,
    if_outlier: bool,
    deviation_pct: float | None,
    score: float,
    settings: Settings,
) -> tuple[bool, str]:
    """Severity from env/Settings — never hardcoded in the UI."""
    abs_dev = 0.0
    if deviation_pct is not None:
        try:
            if deviation_pct == deviation_pct:
                abs_dev = abs(float(deviation_pct))
        except (TypeError, ValueError):
            abs_dev = 0.0
    flagged = bool(settings.anomaly_flag_if_outlier and if_outlier) or abs_dev > settings.anomaly_warning_deviation_pct
    if settings.anomaly_warning_score_cutoff is not None and score <= settings.anomaly_warning_score_cutoff:
        flagged = True
    critical = abs_dev >= settings.anomaly_critical_deviation_pct
    if settings.anomaly_critical_score_cutoff is not None and score <= settings.anomaly_critical_score_cutoff:
        critical = True
    if critical:
        return True, "critical"
    if flagged:
        return True, "warning"
    return False, "normal"


def anomaly_settings_from_app(settings: Settings) -> Settings:
    return settings


def query_anomalies(
    db: Session,
    building_id: str | None = None,
    *,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
    origin: str | None = None,
    limit: int | None = None,
) -> list[Anomaly]:
    q = db.query(Anomaly)
    if building_id is not None:
        q = q.filter(Anomaly.building_id == building_id)
    if from_ts is not None:
        q = q.filter(Anomaly.ts >= from_ts)
    if to_ts is not None:
        q = q.filter(Anomaly.ts <= to_ts)
    if origin is not None:
        q = q.filter(Anomaly.data_origin == origin)
    q = q.order_by(Anomaly.ts.asc())
    if limit is not None:
        q = q.limit(limit)
    return q.all()


def apply_live_classification(row: Anomaly, settings: Settings) -> tuple[bool, str]:
    if_outlier = float(row.score) < 0.0
    return classify_severity_api(
        if_outlier=if_outlier,
        deviation_pct=row.deviation_pct,
        score=float(row.score),
        settings=settings,
    )


def serialize_row(row: Anomaly, settings: Settings) -> dict:
    is_anomaly, severity = apply_live_classification(row, settings)
    return {
        "id": row.id,
        "building_id": row.building_id,
        "ts": row.ts,
        "score": row.score,
        "is_anomaly": is_anomaly,
        "model_run_id": row.model_metrics_id,
        "model_metrics_id": row.model_metrics_id,
        "data_origin": row.data_origin,
        "deviation_pct": row.deviation_pct,
        "severity": severity,
        "actual_kwh": row.actual_kwh,
        "expected_kwh": row.expected_kwh,
    }


def summary_payload(
    building_id: str,
    rows: list[Anomaly],
    settings: Settings,
) -> dict:
    by_severity: dict[str, int] = {"normal": 0, "warning": 0, "critical": 0}
    by_origin: dict[str, int] = {}
    n_flagged = 0
    for row in rows:
        is_anomaly, severity = apply_live_classification(row, settings)
        by_severity[severity] = by_severity.get(severity, 0) + 1
        origin = str(row.data_origin)
        by_origin[origin] = by_origin.get(origin, 0) + 1
        if is_anomaly:
            n_flagged += 1
    ts_min = min((row.ts for row in rows), default=None)
    ts_max = max((row.ts for row in rows), default=None)
    return {
        "building_id": building_id,
        "n_scored": len(rows),
        "n_flagged": n_flagged,
        "by_severity": by_severity,
        "by_origin": by_origin,
        "ts_min": ts_min,
        "ts_max": ts_max,
        "severity_config": {
            "warning_deviation_pct": settings.anomaly_warning_deviation_pct,
            "critical_deviation_pct": settings.anomaly_critical_deviation_pct,
            "warning_score_cutoff": settings.anomaly_warning_score_cutoff,
            "critical_score_cutoff": settings.anomaly_critical_score_cutoff,
            "flag_if_outlier": settings.anomaly_flag_if_outlier,
            "expected_floor_kwh": settings.anomaly_expected_floor_kwh,
        },
    }
