# PowerPredict

4th-year academic project: **smart-building electricity forecasting** on Building Data Genome Project 2 (BDG2), with a FastAPI backend, React dashboard, offline ML training, and MQTT IoT (simulator labeled as simulated).

Phase 1 is the runnable shell: health check, JWT login, empty dashboard.

Phase 2 adds BDG2 subset download, preprocessing, features, and historical Postgres ingest (`python -m ml`). See [docs/PHASE2.md](docs/PHASE2.md).

Phase 3 adds an **offline** training CLI (`python -m ml.train` / `python -m ml.evaluate`) for five next-hour models. It writes `artifacts/metrics.json` only after a real fit. See [docs/PHASE3_MODELS.md](docs/PHASE3_MODELS.md).

Phase 6 adds a **simulated** MQTT publisher and FastAPI consumer. See [docs/PHASE6_MQTT.md](docs/PHASE6_MQTT.md). Simulator traffic is never labeled live.

Phase 8 is the React operations desk (overview, forecasting, model comparison, SHAP, anomalies, IoT poll, historical analytics). See [docs/PHASE8_DASHBOARD.md](docs/PHASE8_DASHBOARD.md). Tiles stay empty until the APIs return rows — scores and SHAP are never invented.

## Stack

| Layer | Choice |
| --- | --- |
| Frontend | React, Vite, TypeScript, Tailwind CSS, Recharts |
| Backend | Python, FastAPI, Pydantic, SQLAlchemy, Postgres |
| ML | pandas / scikit-learn / XGBoost / PyTorch (training is an offline CLI: `python -m ml.train`) |
| IoT | Eclipse Mosquitto + Python simulator (`data_origin=simulated_iot`) |
| Local infra | Docker Compose: **Postgres 15** and **Mosquitto** only |

API and Vite run on the host, not in Compose.

## Prerequisites

- Python 3.11 (recommended; 3.13 is also fine)
- Node.js 20+
- Docker Desktop (for Postgres + Mosquitto). If Docker is unavailable you can still run pytest (SQLite) and the frontend build. To run the API without Postgres, set `DATABASE_URL=sqlite:///./powerpredict-dev.sqlite3` in `.env` (Phase 1 `create_all` will create tables).

## Setup

### 1. Environment

```bash
copy .env.example .env
```

Edit `.env`: set `POSTGRES_PASSWORD`, `JWT_SECRET` (32+ characters), `SEED_ADMIN_PASSWORD`, and `MQTT_PASSWORD`. Do not commit `.env`.

### 2. MQTT password file

Mosquitto refuses anonymous clients. Generate `iot/mosquitto/config/passwd` from `.env` (never hardcode credentials in Python):

```bash
py -3.11 iot/mosquitto/write_passwd.py
```

### 3. Infrastructure

```bash
docker compose up -d
```

Postgres and Mosquitto only. If Mosquitto is down, the API and historical queries still work.

### IoT simulator (simulated data only)

```bash
pip install -r iot\requirements.txt
python -m iot.simulator --help
python -m iot.simulator --dry-run
python -m iot.simulator
```

Topic: `smartbuilding/{building_id}/sensors`. Every payload includes `data_origin=simulated_iot` and `occupancy_source=simulated`. This is **not** live physical sensor data. Historical CSV ingest never publishes MQTT.

`GET /api/v1/iot/status` (JWT) returns `OFFLINE` | `SIMULATED` | `LIVE`. `LIVE` is reserved for `real_iot`; the simulator is always `SIMULATED` when the broker is up. Details: [docs/PHASE6_MQTT.md](docs/PHASE6_MQTT.md).

### 4. Database schema

Phase 1 also calls SQLAlchemy `create_all` on API startup so an empty database is usable immediately. For an explicit migration, from the repo root:

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r backend\requirements.txt
cd backend
alembic upgrade head
```

### 5. Backend

From `backend/` with the repo venv active (`..\.venv` if you created it at the repo root):

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Health: `GET http://127.0.0.1:8000/health`
- Login: `POST http://127.0.0.1:8000/api/v1/auth/login` with `{"email":"...","password":"..."}` from `SEED_ADMIN_*`

### 6. Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` and `/health` to FastAPI (`http://127.0.0.1:8000`). Open `http://localhost:5173` and sign in with the seed admin.

## Tests (no live Postgres required)

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r backend\requirements.txt -r ml\requirements.txt
pip install pytest ruff
pytest
cd frontend
npm install
npm test
npm run build
```

## Data origin

Every energy reading is labeled:

| `data_origin` | Meaning |
| --- | --- |
| `historical` | BDG2 CSV ingest (not MQTT) |
| `simulated_iot` | MQTT simulator |
| `real_iot` | Real sensor path |

Occupancy in the Phase 2 feature table is **simulated** (`occupancy_source=simulated`) from primary use + local hour. It is not a measured sensor.

## Phase 2 data pipeline

```bash
pip install -r ml\requirements.txt
python -m ml download          # BDG2 subset → data/raw (gitignored)
python -m ml prepare           # features → data/processed
python -m ml ingest            # buildings + sensor_readings, data_origin=historical (no MQTT)
python -m ml ingest --dry-run  # counts only
```

If BDG2 cannot be downloaded:

```bash
python -m ml prepare --fixture   # TEST FIXTURE, not BDG2
```

`python -m ml run` tries download + prepare + ingest; ingest dry-runs when `DATABASE_URL` is unreachable.

Splits are chronological **70/15/15** on unique timestamps (same cuts for all selected buildings). Time-series rows are never shuffled.

Offline training (`python -m ml.train`) is a later CLI; it loads processed features and does not run during Phase 2 prepare. SHAP and Isolation Forest fitting are separate CLIs. MQTT is a simulated publisher (`python -m iot.simulator`); it is not a live meter.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/PHASE2.md](docs/PHASE2.md), [docs/PHASE2_DATA.md](docs/PHASE2_DATA.md), and [docs/PHASE6_MQTT.md](docs/PHASE6_MQTT.md).
