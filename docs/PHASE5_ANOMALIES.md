# Phase 5 — Isolation Forest energy anomalies

Offline detector only. The API **does not fit** Isolation Forest; it reads rows written by `python -m ml.anomaly`. This document does not invent anomaly counts — run the CLI and read `artifacts/anomaly/summary.json`.

## Method

For each selected building and `data_origin`, Isolation Forest is fit on the **chronological train split only** (no shuffle, no test leakage). A `StandardScaler` is also fit on that train window.

Scored timestamps (default: **val + test**) receive:

| Field | Meaning |
| --- | --- |
| `score` | sklearn `IsolationForest.decision_function` (higher = more normal; negative ⇒ IF outlier) |
| `is_anomaly` | true if IF `predict == -1` **or** `\|deviation_pct\|` exceeds the warning band |
| `deviation_pct` | `100 * (actual − expected) / max(\|expected\|, expected_floor_kwh)` |
| `severity` | `normal` / `warning` / `critical` from **config**, not the UI |
| `data_origin` | `historical` \| `simulated_iot` \| `real_iot` (always persisted) |

IoT origins are scored **separately** when that origin has its own train rows. If an origin has no train window, the historical IF for the same building is used as a fallback; the stored `data_origin` is still the row’s true origin. Phase 5 processed BDG2 is historical only.

### Features (Isolation Forest)

Configured in `ml/configs/default.yaml` under `anomaly.features` (override the list there). Defaults:

- `energy_kwh`
- lags `lag_1` … `lag_168` when present
- `hour`
- `air_temperature` (`air_temperature_c` alias)
- `residual` when expected consumption could be computed

Required columns `energy_kwh` and `hour`: **drop** the row if missing. Optional IF columns: **train-only median impute**. Rows without an expected value are dropped — residuals are never invented.

### Expected consumption (residual)

1. If `artifacts/metrics.json` exists and its `data_source` matches the processed matrix (BDG2 vs TEST FIXTURE), load a saved tabular forecaster (`xgboost` / `random_forest` / `linear_regression`) and use its predictions as expected \(y_{t+1}\).
2. Otherwise fit **per building / origin** `LinearRegression` on **train only** to predict `energy_kwh` from Phase 2 features **excluding** current energy and the next-hour target (so the baseline cannot cheat).

Phase 3 smoke artifacts under `artifacts/_phase3_fixture_smoke` are fixture-only and are **not** applied to BDG2.

### Flag and severity (configurable)

Default (YAML + env):

- Flag if Isolation Forest outlier **or** `|deviation_pct| > ANOMALY_WARNING_DEVIATION_PCT` (default **25**).
- `critical` if `|deviation_pct| ≥ ANOMALY_CRITICAL_DEVIATION_PCT` (default **50**) or score ≤ optional `ANOMALY_CRITICAL_SCORE_CUTOFF`.
- Else flagged rows are `warning`; the rest are `normal`.

The UI must render the `severity` string and `severity_config` from the API. Do not hardcode bands in the dashboard.

Env keys (also in `.env.example`):

| Key | Role |
| --- | --- |
| `ANOMALY_CONTAMINATION` | Isolation Forest contamination (default `0.01`) |
| `ANOMALY_RANDOM_SEED` | `random_state` (default `42`, or YAML `random_seed`) |
| `ANOMALY_N_ESTIMATORS` | trees (default `100`) |
| `ANOMALY_WARNING_DEVIATION_PCT` | warning `|dev|` band |
| `ANOMALY_CRITICAL_DEVIATION_PCT` | critical `|dev|` band |
| `ANOMALY_WARNING_SCORE_CUTOFF` | optional `decision_function` cutoff |
| `ANOMALY_CRITICAL_SCORE_CUTOFF` | optional `decision_function` cutoff |
| `ANOMALY_FLAG_IF_OUTLIER` | include IF `-1` in the flag rule |
| `ANOMALY_EXPECTED_FLOOR_KWH` | safe denominator (default `1.0`) |
| `ANOMALY_MIN_TRAIN_ROWS` | skip a building/origin below this (default `24`) |

## BDG2 cleaning caveat

BDG2 **cleaned** electricity already replaced Seasonal Hybrid ESD outliers, long zero runs (>24 h), and electricity zeros with NaN. Phase 2 then applied conservative train-only IQR winsorization. Isolation Forest is an **independent second layer** on the remaining series (and residuals). It does not undo or repeat BDG2 cleaning. Flags on historical data may be unusual occupancy, weather, or residual structure — not “the outliers BDG2 missed,” and not fabricated dashboard events.

## How to run

From the repo root, with the venv active and Phase 2 `data/processed` present:

```bash
python -m ml.anomaly
python -m ml.anomaly --no-db
python -m ml.anomaly --fixture
python -m ml.anomaly --score-train
```

`--fixture` uses the labeled **TEST FIXTURE** (not BDG2). `--no-db` skips Postgres; a down database does **not** fail the phase. Artifacts:

```
artifacts/anomaly/scores.parquet
artifacts/anomaly/summary.json
artifacts/anomaly/isolation_forest.joblib
```

`summary.json` contains real `n_scored` / `n_flagged` / `by_building` from that run.

### API (JWT, CORS, rate limits unchanged)

Authenticated:

- `GET /api/v1/anomalies` — existing global list (unchanged path)
- `GET /api/v1/buildings/{id}/anomalies?from&to&origin`
- `GET /api/v1/buildings/{id}/anomalies/timeline`
- `GET /api/v1/buildings/{id}/anomalies/summary`

If nothing has been scored, lists are **empty**. Dummy flags are not returned.

Alembic `003_phase5` adds `deviation_pct`, `severity`, `actual_kwh`, `expected_kwh`, and uniqueness on `(building_id, ts, data_origin)`.

## Tests

```bash
pytest tests/ml/test_anomaly.py tests/backend/test_anomalies.py tests/ml tests/backend
```

## This run (23 Aug 2026)

`python -m ml.anomaly --no-db` against Phase 2 `data/processed` (BDG2 Hog education meters). Counts are from `artifacts/anomaly/summary.json`, not invented.

| | |
| --- | --- |
| data_source | `bdg2:E:/PowerPredict/data/processed` |
| expected | train-only per-building LinearRegression → `energy_kwh` (no Phase 3 BDG2 forecast artifacts; fixture-smoke models were not applied) |
| IF features | energy_kwh, lag_1/2/3/6/12/24/168, hour, air_temperature, residual |
| contamination | 0.01 |
| random_seed | 42 |
| n_estimators | 100 |
| splits scored | val + test |
| n_scored | 15786 |
| n_flagged | 1714 |
| severity | normal 14072, warning 1025, critical 689 |
| origin | historical 15786 |

Per building (5262 scored each): Casandra 136 flagged, Donnie 1395 flagged, Jewel 183 flagged. Donnie’s high rate is a residual/IF concern (see below), not a fabricated dashboard count.
