# Phase 2 — BDG2 ingest and features

This phase builds a **reproducible Python pipeline** (functions + CLI). It does not train models, fit Isolation Forest, compute SHAP, or run the MQTT live simulator.

## How to obtain BDG2

Official sources:

- GitHub: https://github.com/buds-lab/building-data-genome-project-2
- Nature Scientific Data: https://doi.org/10.1038/s41597-020-00712-x
- Zenodo v1.0: https://doi.org/10.5281/zenodo.3887306

PowerPredict uses **cleaned electricity only** (2–3 Education/Office buildings), plus matching weather and metadata. Do not ingest all 1,636 buildings.

BDG2 cleaned meters (`data/meters/cleaned/`) already:

- replaced Seasonal Hybrid ESD (S-H-ESD) outliers with NaN
- removed zero runs longer than 24 continuous hours (NaN)
- removed electricity zeros (NaN)

On GitHub the cleaned file is `electricity_cleaned.csv` (Git LFS). The paper’s path `data/meters/cleaned/electricity.csv` is treated as an alias.

### CLI download (preferred)

From the repo root, with the project venv active:

```bash
pip install -r ml\requirements.txt
python -m ml download
```

This writes gitignored files under `data/raw/`:

- `metadata.csv`
- `weather.csv` (selected sites only)
- `electricity_selected.csv` (selected building columns only)
- `download_manifest.json`

If GitHub LFS / the network is unavailable, either place those three CSVs yourself or run the pipeline on the synthetic **TEST FIXTURE** (not BDG2):

```bash
python -m ml prepare --fixture
```

## Pipeline CLI

| Command | What it does |
| --- | --- |
| `python -m ml download` | Fetch BDG2 subset into `data/raw` |
| `python -m ml prepare` | Load → validate → clean → features → `data/processed` |
| `python -m ml prepare --fixture` | Same, on the labeled TEST FIXTURE |
| `python -m ml ingest` | Upsert `buildings` + historical `sensor_readings` (`data_origin=historical`). No MQTT. |
| `python -m ml ingest --dry-run` | Print counts without writing the DB |
| `python -m ml run` | download + prepare + ingest (ingest dry-runs if Postgres is down) |

Processed outputs (gitignored): `buildings`, `readings_hourly`, `features`, `features_{train,val,test}` as parquet (CSV fallback) and `pipeline_summary.json`.

## Preprocessing decisions

See [PHASE2_DATA.md](PHASE2_DATA.md) for schema, features, counts, and cut timestamps.

| Step | Decision |
| --- | --- |
| Timestamps | BDG2 CSVs are **site-local naive**. Localized with metadata `timezone`, stored as **UTC** (`ts`). Calendar features use `ts_local`. |
| Splits | Chronological **70% / 15% / 15%** on sorted **unique timestamps** (same cuts for every selected building). No shuffle. Legacy calendar windows (through 2017-06-30 / 2017-09-30) remain in `chronological_split` but are not the Phase 2 default. |
| Invalid energy | `energy_kwh < 0` → NaN (BDG2 already NaN’d many outliers/zeros). |
| Outliers | Optional train-only IQR winsorization (k=3) per building. Never fit on val/test. |
| Missing | Linear interpolate gaps ≤ 6 hours **within each split**, then column medians **fit on train only**. |
| Weather | Aligned on `(site_id, ts)`. Humidity is **not** in BDG2; derived from air and dew temperature (Magnus formula). |
| Lags | `lag_{1,2,3,6,12,24,168}` = energy at t−k. Extra lags vs an older 1/24/168-only sketch. |
| Rolling | `shift(1)` then `rolling(w)`. **Current t is not included.** Window at t uses t−1 … t−w. |
| Target | `target_energy_kwh` = y_{t+1}. Label only; never used as a feature. |
| Occupancy | Included. Profile from `primary_use` + local hour + weekend. Every row has `occupancy_source=simulated`. Not a measured sensor. |
| DB ingest | Historical CSV only. `data_origin=historical`. Does not publish MQTT. Null energy rows are skipped (`energy_kwh` is NOT NULL). |

## Tests

```bash
pytest tests
```

Phase 2 tests use the tiny **TEST FIXTURE**, not the full BDG2 extract.
