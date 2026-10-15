# Phase 8 — React dashboard

Campus energy desk: Vite + React + TypeScript + Tailwind + Recharts. The UI **only displays API payloads**. It does not invent MAE/RMSE/R²/MAPE, forecasts, SHAP values, or anomaly counts.

## Run UI + API

From the repo root, copy `.env.example` → `.env` and set `JWT_SECRET`, `SEED_ADMIN_*`, and `ALLOWED_ORIGINS` (include `http://127.0.0.1:5173`).

**API** (from `backend/` with the repo venv):

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Postgres via Docker Compose is the default `DATABASE_URL`. To boot without Postgres:

```bash
# PowerShell, repo root
$env:DATABASE_URL="sqlite:///./powerpredict-dev.sqlite3"
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**UI:**

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` and `/health` to `http://127.0.0.1:8000`. Open `http://127.0.0.1:5173` and sign in with `SEED_ADMIN_EMAIL` / `SEED_ADMIN_PASSWORD`.

All routes except `/login` require a JWT (`Authorization: Bearer`).

## Pages and APIs

| Route | Page | APIs |
| --- | --- | --- |
| `/login` | Sign in | `POST /api/v1/auth/login` |
| `/` | Overview | `/api/buildings`, `/api/energy/history`, `/api/forecast`, `/api/iot/status`, `/api/iot/latest`, `/api/anomalies` |
| `/forecast` | Forecasting | `/api/buildings`, `/api/forecast?building_id&model`, `/api/model-performance` |
| `/models` | Model comparison | `/api/model-performance` only |
| `/explain` | Explainable AI | `/api/explainability?model=` (global); `?model=&row=0` (local). **404 → empty** |
| `/anomalies` | Anomalies | `/api/buildings`, `/api/anomalies?building_id&origin` |
| `/iot` | IoT monitoring (poll ~5s) | `/api/iot/status`, `/api/iot/latest`, `/api/energy/history?origin=simulated_iot\|real_iot` |
| `/buildings` | Historical analytics | `/api/buildings`, `/api/buildings/{id}`, `/api/energy/history` |

`GET /api/health` is public (proxied). Session restore uses `GET /api/v1/auth/me`.

The forecast API has **no horizon parameter**. The Forecasting page does not show a horizon control.

Daily / weekly / monthly / weekday charts are **aggregated in the browser** from hourly `/api/energy/history` and labeled as aggregations. The API cap is 10,000 rows per request; the client pages forward on `from` (up to three pages).

## Provenance

| `data_origin` | Badge |
| --- | --- |
| `historical` | BDG2 archive |
| `simulated_iot` | MQTT simulator |
| `real_iot` | Live meter |

IoT connection chip: `OFFLINE` / `SIMULATED` / `LIVE` from `/api/iot/status`. **The simulator is never labeled LIVE.** Occupancy is shown as a simulated profile unless `occupancy_source` is a measured source.

## Honest empty states

- No buildings / history → empty catalog, no invented kWh.
- No `model_metrics` and no `artifacts/metrics.json` → Model comparison is empty (no placeholder scores).
- No forecast rows → empty actual vs predicted chart.
- Missing SHAP JSON → Explainable AI empty (no fabricated contributions).
- No Isolation Forest rows → Anomalies empty.
- MQTT down → IoT status `OFFLINE`; historical queries still work.

## Tests / build

```bash
cd frontend
npm test
npm run build
```
