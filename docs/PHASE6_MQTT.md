# Phase 6 — MQTT simulator and consumer

This phase adds a **labeled simulated** Mosquitto publisher and a FastAPI consumer that persists readings. It does **not** add dashboard IoT UI, does not treat the stream as live meters, and is not the project definition of done.

Simulator traffic is always `data_origin=simulated_iot` and `occupancy_source=simulated`.

## Broker

Compose already defines Eclipse Mosquitto 2 (`allow_anonymous false`). Credentials come from `.env`, never from source.

```bash
copy .env.example .env
# set MQTT_USERNAME / MQTT_PASSWORD (and Postgres / JWT secrets)
py -3.11 iot/mosquitto/write_passwd.py
docker compose up -d
```

`write_passwd.py` hashes `MQTT_USERNAME` + `MQTT_PASSWORD` into `iot/mosquitto/config/passwd`. Do not commit real passwords.

If the broker is down, **the API still serves** `/health`, JWT login, and historical `sensor_readings` queries. MQTT is fail-soft.

## Topic

User contract (wins over the older `powerpredict` prefix):

```
smartbuilding/{building_id}/sensors
```

The API subscribes to `smartbuilding/+/sensors`. Override the root with `MQTT_TOPIC_PREFIX` (default `smartbuilding`).

## Payload contract

Required JSON fields. Missing `data_origin` is **dropped** (never persisted).

```json
{
  "building_id": "building_001",
  "timestamp": "2026-08-23T10:00:00Z",
  "temperature": 31.2,
  "humidity": 67.4,
  "occupancy": 82,
  "energy_kwh": 42.8,
  "data_origin": "simulated_iot",
  "schema_version": "1",
  "occupancy_source": "simulated"
}
```

| Field | Rule |
| --- | --- |
| `data_origin` | MQTT allows `simulated_iot` or `real_iot` only. `historical` is ingest-only and must never be published here. |
| `timestamp` | ISO-8601 UTC. Alias `ts` is accepted, then normalized to `timestamp`. |
| `occupancy_source` | `simulated` for simulator traffic. `measured` is reserved for a future real_iot path. |
| `schema_version` | Required string (`1` from this phase). |

`temperature` is °C, `humidity` is %, `occupancy` is simulated percent of typical capacity (0–100), **not** a badge reader count.

## Simulator

```bash
pip install -r iot\requirements.txt
python -m iot.simulator --help
python -m iot.simulator --dry-run
python -m iot.simulator
python -m iot.simulator --accelerate --interval 1
```

Logs print a **SIMULATED** banner. Patterns include weekday vs weekend, working hours vs night, diurnal temperature, and occupancy–energy correlation. This is a generator, not a physical sensor.

Building ids: `--buildings …`, else `data/processed/pipeline_summary.json`, else `data/raw/download_manifest.json`, else documented demo ids `building_001`, `building_002`, `building_003`.

Admin start/stop from the API is **not** required in this phase. Start the process from the CLI.

## Backend consumer

FastAPI lifespan starts `iot.subscriber.MqttSubscriber` (paho, background thread). Each valid message is upserted to:

- `sensor_readings` — energy (required) plus temperature, humidity, occupancy, occupancy_source, schema_version
- `weather` — air temperature and humidity (`data_origin` from the payload)
- `occupancy` — occupancy + `occupancy_source=simulated` for simulator traffic

Unknown origin → drop + log. Historical CSV ingest (`python -m ml ingest`) still sets `mqtt_publish: false` and never opens a publisher.

Alembic revision `004_phase6` adds the optional IoT columns. `create_all` on a fresh SQLite/dev DB also creates them.

## Connection status (`GET /api/v1/iot/status`)

Auth required (same JWT as the rest of `/api/v1`). The UI is not wired in this phase.

| Status | Meaning |
| --- | --- |
| `OFFLINE` | Broker unreachable (or MQTT disabled). Historical queries still work. |
| `SIMULATED` | Broker connected. Honest demo label for the simulator path. |
| `LIVE` | Reserved for `real_iot` traffic. **Never** applied to `simulated_iot`. |

Response also includes `broker_connected`, `last_message_at`, `last_data_origin`, `last_building_id`, and `topic`.

## Tests

```bash
pytest tests/iot tests/backend/test_iot.py
```

The suite mocks paho / uses `--dry-run`. It does not require a live Mosquitto process.

## Out of scope (later)

- Frontend IoT page (poll, charts) — empty shell left untouched
- Admin start/stop of the simulator
- Real meter bridge (`real_iot` / `LIVE`)
- Full demo polish / definition of done
