# Processed data

Phase 2 writes aligned hourly readings and model-ready feature tables here (`python -m ml prepare`).

Typical files (gitignored):

- `buildings.parquet` / `.csv`
- `readings_hourly.parquet` / `.csv` — BDG2 (or fixture) energy + weather, `data_origin=historical`
- `features.parquet` plus `features_train|val|test` after chronological 70/15/15 and dropna on lags/rolling/target
- `pipeline_summary.json` — actual counts, cut timestamps, missingness

Splits are applied in code (`ml.features.splits`) — never shuffle. Occupancy, if present, is `occupancy_source=simulated`.
