from datetime import UTC, datetime

from app.db.session import get_session_factory
from app.models.anomaly import Anomaly
from app.models.building import Building


def _token(client) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "test-admin-password"},
    )
    return response.json()["access_token"]


def test_building_anomalies_unauthorized(client):
    response = client.get("/api/v1/buildings/b1/anomalies")
    assert response.status_code == 401


def test_unknown_building_is_404(client):
    headers = {"Authorization": f"Bearer {_token(client)}"}
    response = client.get("/api/v1/buildings/does-not-exist/anomalies", headers=headers)
    assert response.status_code == 404


def test_empty_anomalies_when_none_scored(client):
    db = get_session_factory()()
    try:
        db.add(
            Building(
                id="empty_edu_1",
                site_id="Hog",
                primary_use="Education",
                timezone="UTC",
            )
        )
        db.commit()
    finally:
        db.close()
    headers = {"Authorization": f"Bearer {_token(client)}"}
    listed = client.get("/api/v1/buildings/empty_edu_1/anomalies", headers=headers)
    assert listed.status_code == 200
    assert listed.json() == []
    timeline = client.get("/api/v1/buildings/empty_edu_1/anomalies/timeline", headers=headers)
    assert timeline.status_code == 200
    assert timeline.json()["points"] == []
    summary = client.get("/api/v1/buildings/empty_edu_1/anomalies/summary", headers=headers)
    assert summary.status_code == 200
    body = summary.json()
    assert body["n_scored"] == 0
    assert body["n_flagged"] == 0
    assert "warning_deviation_pct" in body["severity_config"]


def test_scored_anomalies_list_timeline_summary(client):
    db = get_session_factory()()
    try:
        db.add(
            Building(
                id="hog_edu_1",
                site_id="Hog",
                primary_use="Education",
                timezone="US/Central",
            )
        )
        db.add_all(
            [
                Anomaly(
                    building_id="hog_edu_1",
                    ts=datetime(2017, 11, 1, 12, tzinfo=UTC),
                    score=0.15,
                    is_anomaly=False,
                    data_origin="historical",
                    deviation_pct=3.0,
                    severity="normal",
                    actual_kwh=80.0,
                    expected_kwh=77.6,
                ),
                Anomaly(
                    building_id="hog_edu_1",
                    ts=datetime(2017, 11, 1, 13, tzinfo=UTC),
                    score=-0.4,
                    is_anomaly=True,
                    data_origin="historical",
                    deviation_pct=62.0,
                    severity="critical",
                    actual_kwh=160.0,
                    expected_kwh=99.0,
                ),
            ]
        )
        db.commit()
    finally:
        db.close()

    headers = {"Authorization": f"Bearer {_token(client)}"}
    listed = client.get("/api/v1/buildings/hog_edu_1/anomalies", headers=headers)
    assert listed.status_code == 200
    rows = listed.json()
    assert len(rows) == 2
    assert {row["severity"] for row in rows} == {"normal", "critical"}
    assert all(row["data_origin"] == "historical" for row in rows)
    flagged = [row for row in rows if row["is_anomaly"]]
    assert len(flagged) == 1
    assert flagged[0]["actual_kwh"] == 160.0
    assert flagged[0]["expected_kwh"] == 99.0

    timeline = client.get("/api/v1/buildings/hog_edu_1/anomalies/timeline", headers=headers)
    assert timeline.status_code == 200
    points = timeline.json()["points"]
    assert len(points) == 2
    assert points[0]["actual_kwh"] is not None

    summary = client.get("/api/v1/buildings/hog_edu_1/anomalies/summary", headers=headers)
    body = summary.json()
    assert body["n_scored"] == 2
    assert body["n_flagged"] == 1
    assert body["by_severity"]["critical"] == 1
    assert body["by_origin"]["historical"] == 2
    assert body["severity_config"]["critical_deviation_pct"] == 50.0
