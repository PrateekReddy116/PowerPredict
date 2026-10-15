# Architecture

PowerPredict is a **modular monolith** for campus electricity forecasting. One Git repo, four packages (`frontend/`, `backend/`, `ml/`, `iot/`), one Postgres database. No Kubernetes, no microservices, no Redis, no WebSockets.

## Phase 1 (this tree)

Runnable shell only:

- FastAPI with `GET /health`, JWT login, `/api/v1/auth/me`
- CORS from `ALLOWED_ORIGINS` (wildcard rejected)
- IP and user rate limits with `429` and `X-RateLimit-*` headers
- SQLAlchemy models (empty tables) + Alembic initial revision
- React shell with auth gate and provenance badges
- ML package layout, chronological split helper, and Phase 3 offline training CLI
- MQTT simulator CLI + FastAPI consumer (`data_origin=simulated_iot`; see [PHASE6_MQTT.md](PHASE6_MQTT.md))

The API **does not train**. It **loads saved artifacts** only.

## Later phases

| Phase | Intent | Status in this tree |
| --- | --- | --- |
| 2 | Download/ingest BDG2 cleaned **electricity** for 2–3 buildings; feature pipeline; `data_origin=historical` ingest (no MQTT live publisher) | Implemented — see [PHASE2.md](PHASE2.md) |
| 3 | Offline CLI: Linear / RF / XGBoost / LSTM / CNN-LSTM; real MAE/RMSE/R²/MAPE to `model_metrics` and `artifacts/metrics.json` | Implemented — see [PHASE3_MODELS.md](PHASE3_MODELS.md). No fabricated scores. |
| 4 | SHAP / coefficients into `explanations` artifacts (TreeSHAP / LinearExplainer live-capable; LSTM/CNN-LSTM offline permutation SHAP). API loads JSON only — no KernelSHAP per request. | Implemented — see [PHASE4_SHAP.md](PHASE4_SHAP.md) |
| 5 | Isolation Forest on chronological train only; API reads persisted anomaly rows | Implemented — see [PHASE5_ANOMALIES.md](PHASE5_ANOMALIES.md) |
| 6 | MQTT simulator + consumer; status OFFLINE / SIMULATED / LIVE; no IoT dashboard | Implemented — see [PHASE6_MQTT.md](PHASE6_MQTT.md). Simulator is never labeled LIVE. |
| 7+ | IoT UI poll, full dashboard rewrite, demo polish | Not started |

The API **does not train**. It **loads saved artifacts** only.

## Data and splits

- Source: [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2) cleaned electricity meters. Do not shuffle time-series rows.
- **Phase 2 default:** global unique-timestamp cutoffs at 70% and 85% (same dates for every selected building). See [PHASE2.md](PHASE2.md).
- Training CLI can still use calendar windows (through 2017-06-30 / 2017-09-30) or fall back to chronological ratios.
- Target: next-hour electricity kWh (`y_{t+1}`).

## Occupancy policy

BDG2 does not include measured occupancy. When occupancy features are added, generate a profile from `primary_use` + hour of day and set **`occupancy_source=simulated`**. UI and API must never present it as a real sensor.

## Data origin

| Value | Path |
| --- | --- |
| `historical` | Batch CSV ingest only (never published to MQTT) |
| `simulated_iot` | Mosquitto simulator |
| `real_iot` | Real meter bridge |

The dashboard uses distinct `OriginBadge` styles for each. MQTT payloads missing `data_origin` are dropped.

## Table names vs earlier plan wording

| Plan name | Implemented table | Reason |
| --- | --- | --- |
| `readings` | `sensor_readings` | Matches the Phase 1 deliverable name; unique on `(building_id, ts, data_origin, meter_type)` |
| `model_runs` | `model_metrics` | Matches the deliverable; API still exposes the row as a “model run”. `forecasts.model_metrics_id` is the run foreign key (`model_run_id` in JSON) |

`users` is required for JWT and was not in the original table list.

Stub tables `weather` and `occupancy` are written by the Phase 6 MQTT consumer (simulator occupancy stays `occupancy_source=simulated`). `explanations` may receive optional CLI inserts; the API **reads SHAP JSON from `ARTIFACTS_DIR`**, not dummy rows.

## Security baseline

- JWT secret and DB URL from environment only
- Roles: `viewer` (read), `admin` (later: simulator start/stop, ingest)
- Public: `GET /health`, `POST /api/v1/auth/login`
- Login rate limit is stricter (default 10/min/IP)
