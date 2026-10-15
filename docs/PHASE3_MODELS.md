# Phase 3 — Energy forecasting models

Offline training only. The API still **loads saved artifacts**; it does not train. Metrics in `artifacts/metrics.json` are computed from a run. This document does not contain MAE/RMSE/R² values.

Phase 2 feature engineering is **not reimplemented here**. Training consumes a processed feature matrix (parquet/csv under `data/processed/`) with the intended columns below. Tests train on a labeled **synthetic fixture**, not BDG2.

## Target

Next-hour electricity consumption \(y_{t+1}\) (`y_next`). If the column is missing, it is `energy_kwh.shift(-1)` per `building_id`.

## Models

| Name | Library | Input | Artifact |
| --- | --- | --- | --- |
| `linear_regression` | scikit-learn `LinearRegression` | tabular features at \(t\) | `artifacts/linear_regression/model.joblib` |
| `random_forest` | scikit-learn `RandomForestRegressor` | tabular features at \(t\) | `artifacts/random_forest/model.joblib` |
| `xgboost` | `XGBRegressor` | tabular features at \(t\) | `artifacts/xgboost/model.joblib` |
| `lstm` | PyTorch | sequences of length `seq_len` ending at \(t\) | `artifacts/lstm/model.pt` |
| `cnn_lstm` | PyTorch | same sequences | `artifacts/cnn_lstm/model.pt` |

Shared:

- Chronological train / val / test (same test window for every model).
- Feature `StandardScaler` fit on **train only** (`artifacts/feature_scaler.joblib`).
- Sequence models also fit a target scaler on **train \(y\) only** and invert predictions before metrics.
- `random_seed` from `ml/configs/default.yaml` (Python, NumPy, Torch).
- No `shuffle=True` and no `train_test_split`.

### LSTM

`Input (batch, seq_len, n_features) → LSTM(hidden≈64, 1–2 layers) → last hidden → Linear → 1`.

Validation RMSE (original kWh after inverse scaling) drives early stopping; the best checkpoint is restored.

### CNN-LSTM

`Input → Conv1d (features as channels) → ReLU → MaxPool1d → LSTM → Linear → 1`.

CNN-LSTM is **not** assumed to beat LSTM or the tabular models. Compare `artifacts/metrics.json` after a real run.

Default `seq_len` is **24** (one day of hourly steps). Set `seq_len: 48` in the YAML for a two-day window.

## Feature matrix contract (Phase 2)

Training does **not** rebuild features. It consumes Phase 2 output (`data/processed/features_train.parquet` etc., columns from `ml.pipeline.FEATURE_COLUMNS`):

- Current: `energy_kwh`, `sqm`
- Calendar (local time): `hour`, `day`, `day_of_week`, `month`, `weekend`, plus `hour_sin`/`hour_cos`/`dow_sin`/`dow_cos`
- Lags: `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`, `lag_24`, `lag_168`
- Rolling means (t excluded): `rolling_mean_3`, `rolling_mean_6`, `rolling_mean_24`, `rolling_mean_168`
- Weather: `air_temperature`, `dew_temperature`, `humidity`, `wind_speed`, `precip_depth_1hr`
- Occupancy: `occupancy` with `occupancy_source=simulated`
- Target: `target_energy_kwh` (\(y_{t+1}\))
- Meta: `ts` (UTC), `building_id`, `data_origin`, `split`

Sequence models build windows **per building** so a series never crosses building boundaries.

If some intended names are missing, remaining numeric columns except meta/target are used.

## Splits

Phase 2 default: chronological **70 / 15 / 15** on **unique timestamps** (same cuts for every building). No shuffle.

Legacy calendar windows (`2017-06-30` / `2017-09-30`) remain in YAML for `chronological_split` but are **not** used by `python -m ml prepare` or by Phase 3 training unless `split.method: date`.

Sequence models may use the end of train (resp. train+val) as warm-up so every val/test row can be scored without leaking the future.

## Metrics

Computed in `ml/evaluate/metrics.py`:

- **MAE**, **RMSE**, **R²** on the test set in original kWh
- **MAPE** (%) only on rows with `|y_true| >= mape_floor_kwh` (default **1.0 kWh**). If no rows pass, `mape` is JSON `null`. Near-zero cleaned electricity would otherwise explode relative error.
- **training_time_sec** (wall clock for `fit`)

`artifacts/metrics.json` rows match `model_metrics`: `model_name`, train/val/test windows, `mae`, `rmse`, `r2`, `mape`, `artifact_path`, `config_json`, `random_seed`, `training_time_sec`.

Optional: if `DATABASE_URL` works, each row is inserted into Postgres `model_metrics`. A down database **does not** fail training (`--no-db` skips the attempt). `training_time_sec` is a nullable column added in Alembic `002_phase3`.

## How to train

Processed BDG2 must exist first (Phase 2). Then, from the repo root with the venv active:

```bash
pip install -r ml/requirements.txt
python -m ml.train --config ml/configs/default.yaml
python -m ml.evaluate --config ml/configs/default.yaml
```

Useful flags:

- `--models linear_regression,random_forest,xgboost` — skip DL
- `--smoke` — 1 epoch, tiny trees/hidden size (tests use this; not a production result)
- `--data path/to/frame.parquet` — explicit matrix
- `--force-ratio-split` — 70/15/15 even if date windows are non-empty
- `--no-db`

If `data/processed` has no parquet/csv, `python -m ml.train` exits with code 2 and writes **no** metrics file. That is expected until Phase 2 ingest exists.

## Plots

`python -m ml.evaluate` reads **computed** `test_predictions.csv` per model plus `metrics.json`:

- `artifacts/plots/actual_vs_pred_{model}.png`
- `artifacts/plots/residuals_{model}.png`
- `artifacts/plots/comparison_table.png` and `comparison.csv`

Images are gitignored under `artifacts/`. Plot helpers are unit-tested on fixture predictions; they do not invent BDG2 charts.

## Tests

```bash
pytest tests/ml
```

DL tests use `--smoke` settings (1 epoch, `seq_len=8`, hidden 16). They do not assert a real-world MAE.
