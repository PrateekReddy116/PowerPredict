"""Map validated MQTT payloads onto sensor_readings, weather, and occupancy.

Unknown / missing data_origin is rejected by iot.simulator.payload before this runs.
Historical CSV ingest never calls this module.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.building import Building
from app.models.reading import SensorReading
from app.models.stubs import OccupancyObservation, WeatherObservation

logger = logging.getLogger("app.iot")

MQTT_ORIGINS = frozenset({"simulated_iot", "real_iot"})


def _guess_site_id(building_id: str) -> str:
    if "_" in building_id:
        return building_id.split("_", 1)[0]
    return "simulated"


def _guess_primary_use(building_id: str) -> str | None:
    lowered = building_id.lower()
    if "office" in lowered:
        return "Office"
    if "edu" in lowered:
        return "Education"
    return None


def ensure_building(db: Session, building_id: str, *, data_origin: str) -> Building:
    row = db.get(Building, building_id)
    if row is not None:
        return row
    row = Building(
        id=building_id,
        site_id=_guess_site_id(building_id),
        primary_use=_guess_primary_use(building_id),
        timezone="UTC",
        selected_reason=f"Auto-created from MQTT ({data_origin}); not a BDG2 ingest row.",
    )
    db.add(row)
    db.flush()
    return row


def persist_mqtt_reading(db: Session, payload: dict) -> SensorReading:
    """Upsert energy (required) plus temp/humidity/occupancy when present.

    `payload` must already have passed validate_mqtt_payload.
    """
    origin = str(payload.get("data_origin") or "")
    if origin not in MQTT_ORIGINS:
        logger.warning("Dropping MQTT persist: unknown origin %r", origin)
        raise ValueError(f"unknown MQTT origin: {origin}")

    building_id = str(payload["building_id"])
    ts = payload["ts"]
    if not isinstance(ts, datetime):
        raise ValueError("payload ts must be a datetime")

    building = ensure_building(db, building_id, data_origin=origin)
    occupancy_source = str(payload.get("occupancy_source") or "simulated")
    schema_version = str(payload.get("schema_version") or "1")
    energy = float(payload["energy_kwh"])
    temperature = float(payload["temperature"])
    humidity = float(payload["humidity"])
    occupancy = float(payload["occupancy"])

    reading = (
        db.query(SensorReading)
        .filter(
            SensorReading.building_id == building_id,
            SensorReading.ts == ts,
            SensorReading.data_origin == origin,
            SensorReading.meter_type == "electricity",
        )
        .one_or_none()
    )
    if reading is None:
        reading = SensorReading(
            building_id=building_id,
            ts=ts,
            energy_kwh=energy,
            data_origin=origin,
            meter_type="electricity",
        )
        db.add(reading)
    reading.energy_kwh = energy
    reading.temperature_c = temperature
    reading.humidity_pct = humidity
    reading.occupancy = occupancy
    reading.occupancy_source = occupancy_source
    reading.schema_version = schema_version

    weather = (
        db.query(WeatherObservation)
        .filter(
            WeatherObservation.site_id == building.site_id,
            WeatherObservation.ts == ts,
            WeatherObservation.data_origin == origin,
        )
        .first()
    )
    if weather is None:
        weather = WeatherObservation(
            site_id=building.site_id,
            ts=ts,
            data_origin=origin,
        )
        db.add(weather)
    weather.air_temperature_c = temperature
    weather.humidity_pct = humidity

    occ_row = (
        db.query(OccupancyObservation)
        .filter(
            OccupancyObservation.building_id == building_id,
            OccupancyObservation.ts == ts,
        )
        .first()
    )
    if occ_row is None:
        occ_row = OccupancyObservation(
            building_id=building_id,
            ts=ts,
            occupancy=occupancy,
            occupancy_source=occupancy_source,
        )
        db.add(occ_row)
    else:
        occ_row.occupancy = occupancy
        occ_row.occupancy_source = occupancy_source

    db.flush()
    return reading
