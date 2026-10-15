"""Phase 7 HTTP API — FastAPI integration.

The dashboard (Phase 8) should call these `/api/...` paths. `/api/v1/...` remains as
aliases that share the same services. Training stays an offline CLI; this API only
reads Postgres and files under `ARTIFACTS_DIR`.

## Auth

| Path | Auth |
| --- | --- |
| `GET /health` | Public |
| `GET /api/health` | Public |
| `POST /api/v1/auth/login` | Public (stricter IP rate limit) |
| Everything else under `/api` and `/api/v1` | JWT Bearer (`viewer` or `admin`) |

Unauthenticated protected routes return **401**. Secrets (`JWT_SECRET`, DB passwords, MQTT passwords) are never returned.

CORS is `ALLOWED_ORIGINS` only (no `*`). Rate limits send `429` plus `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

## Routes

| Method | Path | Auth | Data source |
| --- | --- | --- | --- |
| GET | `/api/health` | Public | Process liveness. MQTT down does not fail this. |
| GET | `/api/buildings` | JWT | `buildings` table (empty if none ingested) |
| GET | `/api/buildings/{id}` | JWT | `buildings` row; **404** if missing |
| GET | `/api/energy/history` | JWT | `sensor_readings` for `building_id` (`from`, `to`, `origin` optional) |
| GET | `/api/forecast` | JWT | `forecasts` (+ `model_metrics.model_name` when `model` is set) |
| GET | `/api/anomalies` | JWT | `anomalies` (optional `building_id`, `from`, `to`, `origin`) |
| GET | `/api/models` | JWT | `model_metrics` rows only |
| GET | `/api/model-performance` | JWT | `model_metrics`, else `artifacts/metrics.json` if present |
| GET | `/api/explainability` | JWT | Stored SHAP JSON under `ARTIFACTS_DIR/explanations/` |
| GET | `/api/iot/status` | JWT | MQTT runtime (OFFLINE / SIMULATED / LIVE) |
| GET | `/api/iot/latest` | JWT | Latest `simulated_iot` / `real_iot` `sensor_readings` |

Query validation: building ids, datetime ranges (`from` ≤ `to`), and enums (`origin`, `model`). Bad input → **422**. Unknown building on catalog/history/forecast/anomaly-scoped calls → **404**.

Occupancy on readings includes `occupancy_source` (BDG2 occupancy is **simulated** unless a measured source exists).

## Empty artifacts are empty or 404 — not fake scores

- No training rows → `GET /api/models` and `GET /api/model-performance` return **`[]`**. The API never fills MAE/RMSE/R²/MAPE with placeholders.
- No forecast rows for a building → `GET /api/forecast` returns **`[]`**. It does not invent `y_pred`.
- No Isolation Forest rows → `GET /api/anomalies` returns **`[]`**.
- Missing SHAP files or no matching `building_id` / `ts` / `row` → `GET /api/explainability` returns **404**. It does not run KernelSHAP and does not return dummy contribution arrays.
- MQTT broker down → `GET /api/iot/status` is **200** with `status: "OFFLINE"`. Historical energy queries still work. `GET /api/iot/latest` is the latest persisted IoT rows (empty if none).

## Explainability query params

- `model` (required): `linear_regression` \| `random_forest` \| `xgboost` \| `lstm` \| `cnn_lstm`
- Omit `building_id` / `ts` / `row` / `row_index` → **global** explanation JSON
- Any of those set → **local** explanation for that row

## `/api/v1` aliases

Existing Phase 1–6 paths still work (`/api/v1/auth/login`, `/api/v1/buildings`, `/api/v1/readings`, `/api/v1/forecasts`, `/api/v1/anomalies`, `/api/v1/models`, `/api/v1/models/{name}/explanations`, `/api/v1/iot/status`, building-scoped anomaly summary/timeline). They call the same services. Additional aliases: `/api/v1/health`, `/api/v1/energy/history`, `/api/v1/forecast`, `/api/v1/explainability`, `/api/v1/model-performance`, `/api/v1/iot/latest`.
