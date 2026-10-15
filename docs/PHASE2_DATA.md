# Phase 2 data report

Counts below come from an actual run of `python -m ml download` then `python -m ml prepare` on 23 Aug 2026 against BDG2 cleaned electricity (GitHub LFS). They are **not** invented.

Source files:

- https://github.com/buds-lab/building-data-genome-project-2 `data/meters/cleaned/electricity_cleaned.csv`
- Matching `data/metadata/metadata.csv` and `data/weather/weather.csv`
- Citation: Miller et al., *Scientific Data* 7, 368 (2020). Cleaned meters: S-H-ESD outliers, long zero runs, and electricity zeros replaced with NaN.

## Status

- BDG2 download: **succeeded**
- Processed tables: `data/processed/*.parquet` and `pipeline_summary.json` (gitignored)
- Postgres ingest: **dry-run only** (`DATABASE_URL` password auth failed on localhost:5432). Would upsert 3 buildings and 52,632 historical electricity rows (`data_origin=historical`, no MQTT).

## Buildings (3)

All Education, site **Hog**, timezone **US/Central**. Selected as the highest-completeness Education/Office electricity meters (threshold 90%); not the full 1,636-building corpus.

| building_id | site | primary_use | completeness | timezone |
| --- | --- | --- | --- | --- |
| Hog_education_Casandra | Hog | Education | 100.00% | US/Central |
| Hog_education_Donnie | Hog | Education | 100.00% | US/Central |
| Hog_education_Jewel | Hog | Education | 100.00% | US/Central |

These three tied at 100% non-null on the cleaned wide file among Education/Office candidates (900 such columns were streamed).

## Record counts

| Stage | Count |
| --- | --- |
| Raw cleaned electricity rows (hourly, 2016–2017) | 17,544 |
| Building columns kept after selection | 3 |
| Aligned hourly rows (3 buildings × 17,544) | 52,632 |
| Aligned rows with non-null energy | 52,632 |
| Feature rows before dropna | 52,632 |
| Feature rows after dropna (lags / rolling / target) | 52,125 |
| Train after dropna | 36,339 |
| Val after dropna | 7,896 |
| Test after dropna | 7,890 |

Dropna removes the 168-hour lag/rolling warmup plus the last hour per building (no y_{t+1}): 52,632 − 52,125 = 507 = 168×3 + 3.

Unique UTC timestamps after local→UTC: **17,542** (two fewer than 17,544 local hours; US/Central DST).

## Split cuts (global unique timestamps, 70/15/15)

| | |
| --- | --- |
| method | global unique-timestamp fractions (no shuffle) |
| train_end | 2017-05-26 21:00:00+00:00 |
| val_end | 2017-09-13 13:00:00+00:00 |
| actual fractions | train 0.7000, val 0.1500, test 0.1500 (rounding on 17,542 stamps) |

Legacy calendar windows (train through 2017-06-30, val through 2017-09-30) are **not** used by Phase 2 prepare.

## Dataset schema

### BDG2 inputs

| File | Required columns |
| --- | --- |
| metadata | `building_id`, `site_id`, `primaryspaceusage`, `timezone`, `electricity` (+ `sqm`) |
| electricity wide | `timestamp` + one column per building (cleaned kWh) |
| weather | `timestamp`, `site_id`, `airTemperature`, `dewTemperature`, `windSpeed`, `precipDepth1HR` |

### Internal / stored

| Column | Meaning |
| --- | --- |
| `ts` | Hour start, timezone-aware **UTC** (BDG2 CSV timestamps are site-local naive) |
| `ts_local` | Same instant in `US/Central` |
| `energy_kwh` | Electricity at t |
| `target_energy_kwh` / `y_next` | y_{t+1} |
| `data_origin` | `historical` |
| `split` | `train` / `val` / `test` |

## Feature list

- Current / static: `energy_kwh`, `sqm`
- Temporal from **local** time: `hour`, `day`, `day_of_week`, `month`, `weekend`
- Cyclic extras: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
- Lags (past only): `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`, `lag_24`, `lag_168`
- Rolling means (past only; **current t not included** — `shift(1)` then `rolling`): `rolling_mean_3`, `rolling_mean_6`, `rolling_mean_24`, `rolling_mean_168`
- Extra rolling: `rolling_std_24` (same past-only rule)
- Environmental: `air_temperature`, `dew_temperature`, `humidity` (Magnus RH from T and Td; BDG2 has no RH column), `wind_speed`, `precip_depth_1hr`
- Occupancy: `occupancy` ∈ [0, 1], **`occupancy_source=simulated`** on every row (education/office profile from `primary_use` + local hour + weekend)

Aliases `y_next`, `is_weekend`, `roll_mean_24`, `roll_mean_168`, `roll_std_24`, `air_temperature_c`, `dew_temperature_c` are copies for the training-matrix contract.

## Missing-value summary (aligned hourly, before impute)

| column | n_missing | pct |
| --- | --- | --- |
| energy_kwh | 0 | 0.00% |
| air_temperature | 12 | 0.023% |
| dew_temperature | 12 | 0.023% |
| humidity | 12 | 0.023% |
| wind_speed | 12 | 0.023% |
| precip_depth_1hr | 159 | 0.302% |

After within-split interpolation (≤6 h) and train-only median impute, those weather columns are 0 missing. Energy was already complete for these three cleaned meters.

## Preprocessing decisions

- Prefer BDG2 **cleaned** electricity (S-H-ESD + long-zero + electricity-zero → NaN). These three meters had no remaining energy NaNs.
- Extra handling is conservative: negatives → NaN; train-only 3×IQR winsorization per building (bounds can be negative on the lower side; non-negative kWh are unchanged there).
- Imputers and IQR bounds fit on **train only**.
- Humidity derived with Magnus a=17.625, b=243.04.
- Occupancy is simulated; never presented as measured.

## Occupancy

Included. Every feature row has `occupancy_source=simulated`.
