"""Phase 2 CLI: download BDG2 subset, prepare features, ingest historical readings.

  python -m ml download
  python -m ml prepare
  python -m ml ingest [--dry-run]
  python -m ml run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml.constants import MIN_COMPLETENESS, N_BUILDINGS, PREFERRED_PRIMARY_USES
from ml.ingest.download import (
    DATA_RAW,
    METADATA_PATHS,
    WEATHER_PATHS,
    DownloadError,
    default_raw_paths,
    download_to_file,
    stream_electricity_columns,
    write_download_manifest,
)
from ml.ingest.fixture import (
    FIXTURE_LABEL,
    make_fixture_electricity_wide,
    make_fixture_metadata,
    make_fixture_weather,
)
from ml.ingest.load import (
    electricity_completeness,
    load_electricity_wide,
    load_metadata,
    load_weather,
)
from ml.ingest.select import candidate_metadata, select_buildings
from ml.paths import (
    DATA_PROCESSED,
    RAW_ELECTRICITY_SUBSET,
    RAW_METADATA,
    RAW_WEATHER,
    ensure_data_dirs,
)
from ml.pipeline import persist_processed, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ml", description="PowerPredict Phase 2 data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download BDG2 metadata, weather, and a cleaned-electricity subset")
    dl.add_argument("--force", action="store_true", help="Re-download even if raw files exist")

    prep = sub.add_parser("prepare", help="Run preprocessing + features; write data/processed")
    prep.add_argument("--fixture", action="store_true", help=f"Use the synthetic {FIXTURE_LABEL} instead of BDG2")
    prep.add_argument("--metadata", type=Path, default=None)
    prep.add_argument("--electricity", type=Path, default=None)
    prep.add_argument("--weather", type=Path, default=None)

    ing = sub.add_parser("ingest", help="Upsert buildings + historical sensor_readings (no MQTT)")
    ing.add_argument("--dry-run", action="store_true")
    ing.add_argument("--database-url", default=None)

    run = sub.add_parser("run", help="download (if needed) + prepare + ingest (dry-run if DB is down)")
    run.add_argument("--fixture", action="store_true")
    run.add_argument("--skip-ingest", action="store_true")
    run.add_argument("--force-download", action="store_true")
    return parser


def cmd_download(*, force: bool = False) -> dict:
    ensure_data_dirs()
    paths = default_raw_paths()
    if not force and paths["metadata"].exists() and paths["electricity_subset"].exists() and paths["weather"].exists():
        print("Raw BDG2 files already present; pass --force to re-download.")
        return {"status": "skipped", "paths": {k: str(v) for k, v in paths.items()}}

    print("Downloading BDG2 metadata…")
    meta_url = download_to_file(METADATA_PATHS, paths["metadata"])
    metadata = load_metadata(paths["metadata"])
    candidates = candidate_metadata(metadata, primary_uses=PREFERRED_PRIMARY_USES)
    candidate_ids = candidates["building_id"].tolist()
    print(f"Education/Office electricity candidates: {len(candidate_ids)}")

    print("Streaming cleaned electricity (selected candidate columns only)…")
    tmp_candidates = DATA_RAW / "electricity_candidates.csv"
    elec_url, present, _non_null, n_rows = stream_electricity_columns(candidate_ids, tmp_candidates)
    print(f"Electricity rows={n_rows} building_columns={len(present) - 1} source={elec_url}")

    wide = load_electricity_wide(tmp_candidates)
    completeness = electricity_completeness(wide)
    choices = select_buildings(metadata, completeness, n=N_BUILDINGS, min_completeness=MIN_COMPLETENESS)
    selected_ids = [c.building_id for c in choices]
    keep_cols = ["timestamp", *selected_ids]
    wide.loc[:, keep_cols].to_csv(paths["electricity_subset"], index=False)
    tmp_candidates.unlink(missing_ok=True)
    print("Selected buildings:")
    for choice in choices:
        print(f"  {choice.building_id}  completeness={choice.completeness:.4f}  {choice.primary_use}  {choice.site_id}")

    print("Downloading weather…")
    weather_url = download_to_file(WEATHER_PATHS, paths["weather"])
    weather = load_weather(paths["weather"])
    sites = {c.site_id for c in choices}
    weather_filtered = weather.loc[weather["site_id"].isin(sites)]
    weather_filtered.to_csv(paths["weather"], index=False)

    manifest = {
        "status": "ok",
        "source": "BDG2",
        "citation": "Miller et al., Scientific Data 7, 368 (2020); cleaned meters: S-H-ESD, long-zero removal",
        "urls": {"metadata": meta_url, "electricity": elec_url, "weather": weather_url},
        "selected_building_ids": selected_ids,
        "electricity_rows": int(n_rows),
    }
    write_download_manifest(manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def _load_inputs(args: argparse.Namespace) -> tuple[str, object, object, object]:
    if args.fixture:
        print(f"Using {FIXTURE_LABEL}")
        return (
            "test_fixture",
            make_fixture_metadata(),
            make_fixture_electricity_wide(),
            make_fixture_weather(),
        )
    metadata_path = args.metadata or RAW_METADATA
    electricity_path = args.electricity or RAW_ELECTRICITY_SUBSET
    weather_path = args.weather or RAW_WEATHER
    if not (metadata_path.exists() and electricity_path.exists() and weather_path.exists()):
        raise FileNotFoundError(
            "BDG2 raw files not found. Run `python -m ml download` or `python -m ml prepare --fixture`."
        )
    return (
        "bdg2",
        load_metadata(metadata_path),
        load_electricity_wide(electricity_path),
        load_weather(weather_path),
    )


def cmd_prepare(args: argparse.Namespace) -> dict:
    source, metadata, electricity, weather = _load_inputs(args)
    n_buildings = 2 if source == "test_fixture" else N_BUILDINGS
    result = run_pipeline(metadata, electricity, weather, source=source, n_buildings=n_buildings)
    written = persist_processed(result)
    print(json.dumps(result.summary["records"], indent=2))
    print("Wrote:")
    for key, path in written.items():
        print(f"  {key}: {path}")
    return {"summary": result.summary, "written": written, "result": result}


def cmd_ingest(*, dry_run: bool, database_url: str | None, result=None) -> dict:
    from ml.ingest.db import database_reachable, ingest_historical

    if result is None:
        buildings_path = DATA_PROCESSED / "buildings.parquet"
        readings_path = DATA_PROCESSED / "readings_hourly.parquet"
        buildings_csv = DATA_PROCESSED / "buildings.csv"
        readings_csv = DATA_PROCESSED / "readings_hourly.csv"
        if buildings_path.exists():
            buildings = __import__("pandas").read_parquet(buildings_path)
        elif buildings_csv.exists():
            buildings = __import__("pandas").read_csv(buildings_csv)
        else:
            raise FileNotFoundError("Run `python -m ml prepare` before ingest.")
        if readings_path.exists():
            readings = __import__("pandas").read_parquet(readings_path)
        elif readings_csv.exists():
            readings = __import__("pandas").read_csv(readings_csv, parse_dates=["ts"])
        else:
            raise FileNotFoundError("Run `python -m ml prepare` before ingest.")
    else:
        buildings, readings = result.buildings, result.readings

    if dry_run:
        payload = ingest_historical(buildings, readings, database_url=database_url, dry_run=True)
        print(json.dumps(payload, indent=2))
        return payload

    ok, reason = database_reachable(database_url)
    if not ok:
        payload = ingest_historical(buildings, readings, dry_run=True)
        payload["status"] = "database_unreachable"
        payload["error"] = reason
        print("DATABASE_URL not reachable; dry-run only.")
        print(json.dumps(payload, indent=2, default=str))
        return payload
    payload = ingest_historical(buildings, readings, database_url=database_url, dry_run=False)
    print(json.dumps(payload, indent=2, default=str))
    return payload


def cmd_run(args: argparse.Namespace) -> int:
    if args.fixture:
        prep_args = argparse.Namespace(fixture=True, metadata=None, electricity=None, weather=None)
        prepared = cmd_prepare(prep_args)
    else:
        try:
            cmd_download(force=args.force_download)
            prep_args = argparse.Namespace(fixture=False, metadata=None, electricity=None, weather=None)
            prepared = cmd_prepare(prep_args)
        except (DownloadError, FileNotFoundError, OSError, ValueError) as exc:
            print(f"BDG2 download/prepare failed: {exc}", file=sys.stderr)
            print(f"Falling back to {FIXTURE_LABEL}", file=sys.stderr)
            prep_args = argparse.Namespace(fixture=True, metadata=None, electricity=None, weather=None)
            prepared = cmd_prepare(prep_args)
    if args.skip_ingest:
        return 0
    cmd_ingest(dry_run=False, database_url=None, result=prepared["result"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "download":
        cmd_download(force=args.force)
        return 0
    if args.command == "prepare":
        cmd_prepare(args)
        return 0
    if args.command == "ingest":
        cmd_ingest(dry_run=args.dry_run, database_url=args.database_url)
        return 0
    if args.command == "run":
        return cmd_run(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
