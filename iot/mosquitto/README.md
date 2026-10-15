# Mosquitto

`allow_anonymous false` plus `password_file`. Generate the file from `.env` (not from source):

```bash
pip install python-dotenv   # already in backend extras if you prefer
py -3.11 iot/mosquitto/write_passwd.py
```

Alternatively copy the example file (hash of the `.env.example` MQTT password) to `passwd`:

```bash
copy iot\mosquitto\config\passwd.example iot\mosquitto\config\passwd
```

Then `docker compose up -d`.

## Payload contract (MQTT)

Topic: `smartbuilding/{building_id}/sensors`

Required JSON fields:

- `building_id` (text, matches `buildings.id`)
- `timestamp` (ISO-8601 UTC; alias `ts` accepted)
- `temperature` (°C), `humidity` (%), `occupancy` (simulated 0–100)
- `energy_kwh` (float)
- `schema_version` (string, currently `1`)
- `occupancy_source`: **`simulated`** for the simulator
- `data_origin`: **`simulated_iot`** or **`real_iot`** only

Missing `data_origin` is rejected. `historical` is valid in the database for BDG2 ingest and **must not** be published on MQTT.

The Python simulator always publishes `data_origin=simulated_iot`. See [docs/PHASE6_MQTT.md](../docs/PHASE6_MQTT.md).
