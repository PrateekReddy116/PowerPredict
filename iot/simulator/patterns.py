"""SIMULATED occupancy, weather, and energy patterns. Not physical sensors."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from iot.simulator.payload import SCHEMA_VERSION, validate_mqtt_payload, wire_payload

DATA_ORIGIN = "simulated_iot"
OCCUPANCY_SOURCE = "simulated"


def _primary_use(building_id: str) -> str:
    lowered = building_id.lower()
    if "office" in lowered:
        return "office"
    if "edu" in lowered:
        return "education"
    # Documented demo ids: 001 education, 002 office, 003 education.
    if building_id.endswith("_002") or building_id.endswith("002"):
        return "office"
    return "education"


def _base_energy(building_id: str) -> float:
    seed = sum(ord(ch) for ch in building_id) % 40
    return 28.0 + seed


def occupancy_percent(building_id: str, when: datetime) -> float:
    """Simulated occupancy as percent of typical capacity (0–100)."""
    local = when.astimezone(UTC)
    hour = local.hour
    weekday = local.weekday()  # 0=Mon
    weekend = weekday >= 5
    use = _primary_use(building_id)
    if weekend:
        base = 12.0 if use == "education" else 6.0
    elif use == "office":
        if 8 <= hour < 18:
            base = 88.0
        elif hour in (7, 18, 19):
            base = 38.0
        else:
            base = 8.0
    else:
        if 8 <= hour < 17:
            base = 82.0
        elif hour in (7, 17, 18):
            base = 42.0
        else:
            base = 10.0
    wobble = 3.0 * math.sin(hour / 24.0 * 2.0 * math.pi + (weekday / 7.0))
    return max(0.0, min(100.0, base + wobble))


def temperature_c(building_id: str, when: datetime) -> float:
    local = when.astimezone(UTC)
    hour = local.hour
    # Diurnal: cooler at night, peak mid-afternoon. Offset per building.
    offset = (sum(ord(ch) for ch in building_id) % 5) - 2.0
    diurnal = 24.5 + 6.5 * math.sin((hour - 9) / 24.0 * 2.0 * math.pi)
    return diurnal + offset


def humidity_pct(building_id: str, when: datetime) -> float:
    temp = temperature_c(building_id, when)
    # Inverse-ish of temperature, still in a realistic indoor/outdoor band.
    humidity = 78.0 - (temp - 18.0) * 1.6
    return max(30.0, min(90.0, humidity))


def energy_kwh(building_id: str, when: datetime, occupancy: float, temperature: float) -> float:
    """Correlate energy with occupancy, working hours, and a mild HVAC term."""
    local = when.astimezone(UTC)
    hour = local.hour
    weekend = local.weekday() >= 5
    night = hour < 6 or hour >= 22
    base = _base_energy(building_id)
    occ_term = base * (0.28 + 0.72 * (occupancy / 100.0))
    hvac = 0.35 * abs(temperature - 22.0)
    if weekend:
        occ_term *= 0.55
    if night:
        occ_term *= 0.45
    return max(0.5, occ_term + hvac)


def generate_reading(building_id: str, when: datetime | None = None) -> dict:
    ts = when or datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    ts = ts.astimezone(UTC)
    occ = occupancy_percent(building_id, ts)
    temp = temperature_c(building_id, ts)
    humidity = humidity_pct(building_id, ts)
    energy = energy_kwh(building_id, ts, occ, temp)
    canonical = validate_mqtt_payload(
        {
            "building_id": building_id,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temperature": temp,
            "humidity": humidity,
            "occupancy": occ,
            "energy_kwh": energy,
            "data_origin": DATA_ORIGIN,
            "schema_version": SCHEMA_VERSION,
            "occupancy_source": OCCUPANCY_SOURCE,
        }
    )
    return wire_payload(canonical)
