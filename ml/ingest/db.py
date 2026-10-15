"""Upsert selected buildings and historical sensor_readings. Never publishes MQTT."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, delete
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ml.constants import DATA_ORIGIN_HISTORICAL

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env_file(path: Path | None = None) -> None:
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _ensure_backend_on_path() -> None:
    import sys

    backend = str(REPO_ROOT / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)


def make_engine(database_url: str | None = None) -> Engine:
    load_env_file()
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, connect_args=connect_args, future=True)


def _chunks(frame: pd.DataFrame, size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size]


def ingest_historical(
    buildings: pd.DataFrame,
    readings: pd.DataFrame,
    *,
    database_url: str | None = None,
    dry_run: bool = False,
    batch_size: int = 2000,
) -> dict[str, int | str | bool]:
    """Replace historical electricity for the selected buildings; upsert building rows.

    Skips readings with null energy (DB column is NOT NULL). data_origin is always historical.
    """
    building_ids = [str(item) for item in buildings["id"].tolist()]
    energy = readings.loc[readings["energy_kwh"].notna()].copy()
    if "data_origin" in energy.columns:
        energy = energy.loc[energy["data_origin"].astype(str) == DATA_ORIGIN_HISTORICAL]
    # DST spring-forward can map two site-local hours onto one UTC ts.
    dup_cols = [c for c in ("building_id", "ts") if c in energy.columns]
    if dup_cols:
        energy = energy.drop_duplicates(subset=dup_cols, keep="first")
    payload = {
        "dry_run": dry_run,
        "buildings": len(building_ids),
        "readings_attempted": int(len(energy)),
        "data_origin": DATA_ORIGIN_HISTORICAL,
        "mqtt_publish": False,
    }
    if dry_run:
        payload["status"] = "dry-run"
        return payload

    _ensure_backend_on_path()
    from app.db.base import Base
    from app.models import load_all_models
    from app.models.building import Building
    from app.models.reading import SensorReading

    load_all_models()
    engine = make_engine(database_url)
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
    session = factory()
    try:
        for _, row in buildings.iterrows():
            existing = session.get(Building, str(row["id"]))
            if existing is None:
                existing = Building(id=str(row["id"]))
                session.add(existing)
            existing.site_id = str(row["site_id"])
            existing.primary_use = None if pd.isna(row.get("primary_use")) else str(row["primary_use"])
            existing.sqm = None if pd.isna(row.get("sqm")) else float(row["sqm"])
            existing.timezone = str(row.get("timezone") or "UTC")
            existing.selected_reason = (
                None if pd.isna(row.get("selected_reason")) else str(row["selected_reason"])
            )
        session.commit()

        session.execute(
            delete(SensorReading).where(
                SensorReading.building_id.in_(building_ids),
                SensorReading.data_origin == DATA_ORIGIN_HISTORICAL,
                SensorReading.meter_type == "electricity",
            )
        )
        session.commit()

        inserted = 0
        for chunk in _chunks(energy, batch_size):
            rows = []
            for rec in chunk.to_dict(orient="records"):
                ts = pd.Timestamp(rec["ts"])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
                temp = rec.get("air_temperature", rec.get("temperature_c"))
                hum = rec.get("humidity", rec.get("humidity_pct"))
                occ = rec.get("occupancy")
                occ_src = rec.get("occupancy_source")
                rows.append(
                    SensorReading(
                        building_id=str(rec["building_id"]),
                        ts=ts.to_pydatetime(),
                        energy_kwh=float(rec["energy_kwh"]),
                        data_origin=DATA_ORIGIN_HISTORICAL,
                        meter_type="electricity",
                        temperature_c=None if temp is None or pd.isna(temp) else float(temp),
                        humidity_pct=None if hum is None or pd.isna(hum) else float(hum),
                        occupancy=None if occ is None or pd.isna(occ) else float(occ),
                        occupancy_source=(
                            None
                            if occ_src is None or (isinstance(occ_src, float) and pd.isna(occ_src))
                            else str(occ_src)
                        ),
                    )
                )
            session.add_all(rows)
            session.commit()
            inserted += len(rows)
        payload["readings_inserted"] = inserted
        payload["status"] = "ok"
        return payload
    finally:
        session.close()
        engine.dispose()


def database_reachable(database_url: str | None = None) -> tuple[bool, str]:
    try:
        engine = make_engine(database_url)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        engine.dispose()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 — report any connectivity failure
        return False, str(exc)
