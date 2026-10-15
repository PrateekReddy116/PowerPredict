# PowerPredict final audit

Strict examiner review of `E:\PowerPredict` (architecture, `.cursor/rules`, ML, IoT, backend, frontend, docs, tests, `.env.example`). Evidence is file-cited. No invented bugs. Last E2E context (historical SQLite path, MQTT/Docker down, tabular models trained, occupancy simulated, no hardcoded KPIs) was re-checked.

**Counts:** 1 critical, 6 major, 8 minor. Untrained LSTM/CNN-LSTM on BDG2 is a **limitation**, not a fake-metric defect (UI does not invent DL scores).

---

## 1. Critical issues

### C1. Login / IP rate limits key on untrusted `X-Forwarded-For`

`backend/app/core/rate_limit.py` (`client_ip`) uses the first `X-Forwarded-For` hop with no trusted-proxy check. Combined with the stricter login bucket (`LOGIN_PATH = "/api/v1/auth/login"`, default 10/min), a client can rotate spoofed headers and bypass the brute-force cap. User-based limits only apply after a valid JWT (`try_decode_user_id`).

This is a real attack against the public login surface. JWT secret, CORS, and 429 headers themselves are implemented (`backend/app/core/config.py`, `backend/app/main.py`, `tests/backend/test_rate_limit.py`, `tests/backend/test_cors.py`).

---

## 2. Major issues

### M1. Sequence-model test predictions omit `building_id`

In `ml/train.py`, tabular runs pass `prepared.groups_test` into `_write_predictions`; sequence models pass `None`:

```text
groups_out = None if is_sequence_model(name) else prepared.groups_test
```

`ml/evaluate/persist.py` `try_insert_forecasts` then skips files missing `building_id`. LSTM/CNN-LSTM forecasts therefore never land in `forecasts`, so `/api/forecast?model=lstm` stays empty even after a real DL fit.

`ml/explain/run.py` already reconstructs sequence group ids for SHAP; training does not reuse that for the CSV. `make_sequences_grouped` / `make_sequences_with_history_grouped` in `ml/models/dataset.py` return `(X, y, ts)` only.

### M2. Overview “next-hour forecast” mixes models

`frontend/src/pages/Overview.tsx` calls `GET /api/forecast` **without** `model`. `backend/app/services/forecasts.py` orders by `target_ts` only (no model). With three tabular runs sharing the same test hours, the last row is whichever model the database returns last at that timestamp. The tile is labeled as a single next-hour forecast.

The Forecasting page does pass `model` (`frontend/src/pages/Forecast.tsx`).

### M3. Occupancy units disagree (0–1 vs 0–100)

- Phase 2 features: `ml/features/occupancy.py` `occupancy_profile` returns **0–1**.
- Historical ingest writes that column (`ml/ingest/db.py`).
- MQTT simulator: `iot/simulator/patterns.py` `occupancy_percent` returns **0–100** (documented in `docs/PHASE6_MQTT.md`).
- Persist stores the wire value (`backend/app/services/iot_ingest.py`; `tests/backend/test_iot.py` asserts `occupancy == 82`).

Overview (`frontend/src/pages/Overview.tsx`) and IoT (`frontend/src/pages/Iot.tsx`) print the raw number. Switching origin on the same building shows ~0.9 vs ~88 with no unit. Occupancy is still labeled **simulated** (not measured) — provenance is honest; the **scale** is not.

### M4. Forecast KPIs treat a failed performance fetch as “no model”

`frontend/src/pages/Forecast.tsx` swallows `/api/model-performance` errors (except 401) with `setPerformance([])`. KPI tiles then show “No row for this model” instead of an error state. Models page surfaces the same failure via `ErrorBanner`.

### M5. Stale series while building / origin changes

Overview, Forecast, Anomalies, and Buildings keep the previous payload until the new request resolves (`setLoading(true)` without clearing `history` / `series` / `rows`). Charts and KPIs can show **another building’s** kWh for one fetch. Forecast empty-chart logic (`loading && !series`) also skips the skeleton when `series` is a leftover array.

### M6. `forecasts.target_ts` is the feature time `t`, not `t+1`

`ml/pipeline.py` `add_target` sets `target_energy_kwh` = energy at **t+1**. `ml/train.py` writes `prepared.ts_test` (time **t**) into `test_predictions.csv`. `try_insert_forecasts` copies that into `forecasts.target_ts`. Overview labels the timestamp as the next-hour forecast time. Internal y_true/y_pred pairing is consistent; the **clock label is one hour early**.

---

## 3. Minor issues

1. **`docs/ARCHITECTURE.md` is stale.** “Phase 7+ … Not started” while `docs/PHASE7_API.md` and `docs/PHASE8_DASHBOARD.md` exist and the React desk is implemented.
2. **No hourly `asfreq` reindex.** Lags/rolling assume consecutive rows are consecutive hours (`ml/features/lags.py`, `ml/features/rolling.py`). BDG2 cleaned meters are usually hourly-with-NaN; a missing row (not a NaN) would make `lag_1` “previous row”, not “t−1h”.
3. **MQTT publish is fire-and-forget.** `iot/simulator/publisher.py` does not inspect `publish()` rc or `wait_for_publish`. Subscriber uses QoS 1 and paho `loop_start` (`iot/subscriber.py`). Fine for a demo; not a reliable meter bridge.
4. **Broker-up with no messages still reports `SIMULATED`.** `backend/app/services/mqtt_consumer.py` `connection_status`: connected and not `real_iot` → `SIMULATED`. Documented; can look like traffic is flowing.
5. **In-memory rate limiter** (`backend/app/core/rate_limit.py`) is per process. Honest for one uvicorn worker; not shared across workers.
6. **`require_admin` unused.** `backend/app/core/security.py` defines it; no simulator start/stop or ingest route uses it (CLI-only, matching Phase 6).
7. **Occupancy / weather stub tables** (`backend/app/models/stubs.py`) have no unique `(building_id, ts, origin)` constraint. MQTT upserts by first match; mixed origins can overwrite occupancy.
8. **Interpolation is index-linear, not time-aware** (`ml/preprocess/quality.py` `interpolate_short_gaps`). Safe if the frame is already regular hourly.

Hardcoded MAE/RMSE/R²/MAPE were **not** found in `frontend/src`. CORS is not `*`. `JWT_SECRET` has no source default. KernelExplainer is not used (`ml/explain/compute.py`, `ml/explain/registry.py`). Simulator is never `LIVE` (`mqtt_consumer.py`, `IotStatusChip.tsx`). Scaler is train-only (`ml/models/dataset.py` `scaler.fit(x_train)`). Isolation Forest fits train only (`ml/anomaly/detect.py`).

---

## 4. Academic / viva concerns

| Topic | Can they explain it? | Risk |
| --- | --- | --- |
| Problem / motivation | Campus next-hour kWh from BDG2; README + architecture | Low if they stick to electricity-only, 2–3 buildings |
| Dataset (BDG2) | Cleaned electricity, site-local timestamps → UTC (`docs/PHASE2.md`, `ml/preprocess/timestamps.py`) | Must not claim all 1,636 buildings |
| Preprocess | Train-only IQR + medians; interpolate ≤6h inside each split (`ml/pipeline.py`) | Must say BDG2 already NaN’d S-H-ESD / long zeros |
| Features | Lags past-only; rolling `shift(1)` then roll; occupancy **simulated** | Must not call occupancy a sensor |
| Model choice | Linear / RF / XGBoost / LSTM / CNN-LSTM; training is offline CLI | **LSTM/CNN-LSTM were not in the last BDG2 `artifacts/metrics.json`** |
| LSTM / CNN-LSTM | `ml/models/lstm.py`, `cnn_lstm.py`, `torch_seq.py` (last hidden, early stop on val RMSE, no DataLoader shuffle) | Cannot quote BDG2 DL MAE. Smoke artifacts live under `artifacts/_phase3_fixture_smoke/` only |
| Evaluation | MAE/RMSE/R²; MAPE with 1 kWh floor (`ml/evaluate/metrics.py`) | Tabular **R² ≈ 0.99** (`artifacts/metrics.json`) because `energy_kwh` at *t* is a feature for *y_{t+1}* — persistence-like. No naive/seasonal baseline in the comparison table |
| SHAP | TreeSHAP / LinearExplainer; DL = offline PermutationExplainer, not KernelSHAP-per-request | No LSTM/CNN-LSTM JSON under production `artifacts/explanations/` (only linear/RF/XGBoost). UI empty-states that correctly |
| Isolation Forest | Train-only per building/origin; residual optional (`ml/anomaly/detect.py`, `residuals.py`) | Independent of BDG2 cleaning — do not claim “we found the same outliers” |
| IoT | Simulator + fail-soft consumer; `data_origin=simulated_iot` | Demo MQTT needs Docker/Mosquitto + `iot/mosquitto/config/passwd` |
| Limitations | Occupancy simulated; DL untrained on BDG2; SQLite vs Postgres; no real_iot bridge | Must say these in the viva |

**Not a fake-metric bug:** `artifacts/metrics.json` contains only `linear_regression`, `random_forest`, `xgboost` with `smoke: false` and `data_source: "bdg2:..."`. Models / Forecast pages omit missing DL rows rather than filling zeros (`frontend/src/pages/Models.tsx`).

---

## 5. Recommended fixes

**Critical / major (this pass):**

1. Ignore `X-Forwarded-For` unless an explicit env flag enables it (`rate_limit.py` + `.env.example`).
2. Return per-window `building_id` from sequence helpers; write it on LSTM/CNN-LSTM `test_predictions.csv`.
3. Overview forecast query must include the selected `model`.
4. Display occupancy as a percent whether the stored value is a fraction or 0–100.
5. Forecast page: performance fetch failure → error banner, not empty metrics.
6. Clear building-scoped series at the start of each fetch (skeleton / empty, not the previous building).
7. Label `target_ts` as forecast origin (features at *t* → target *t+1*), or persist *t+1* consistently.

**Leave as minor / viva:**

- Architecture doc drift; hourly reindex; MQTT puback/reconnect; `SIMULATED` while idle; per-process limiter; unused admin role; stub uniqueness; train DL on BDG2; add a persistence baseline for R² discussion.
- Do **not** claim simulated IoT is real. Do **not** invent LSTM scores.

**Known remaining demo risk (not a code defect):** Mosquitto/Postgres via Docker; MQTT E2E fails without Compose + passwd file. Historical + JWT + tabular metrics work on SQLite.
