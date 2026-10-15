# Raw data

Place BDG2 extracts here. This directory is gitignored except for this README.

## How to obtain BDG2

1. Prefer `python -m ml download` from the repo root (GitHub LFS: metadata, weather, cleaned electricity **columns for selected buildings only**).
2. Official project: [building-data-genome-project-2](https://github.com/buds-lab/building-data-genome-project-2), [Scientific Data paper](https://doi.org/10.1038/s41597-020-00712-x), [Zenodo v1.0](https://doi.org/10.5281/zenodo.3887306).
3. Use **cleaned electricity** (S-H-ESD outliers, long zeros, and electricity zeros already set to NaN). GitHub filename: `data/meters/cleaned/electricity_cleaned.csv`.
4. Select **2–3** Education/Office buildings with high completeness; store `selected_reason` on each `buildings` row.
5. Timestamps in the CSVs are **site-local naive**. The pipeline localizes with metadata `timezone` and stores **UTC**.
6. Ingest must set `data_origin=historical`. Historical CSV ingest must not publish to MQTT.

If download fails, run `python -m ml prepare --fixture`. That path is a **TEST FIXTURE, not BDG2**.
