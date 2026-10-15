"""Download a BDG2 subset (metadata, weather, cleaned electricity for selected columns)."""

from __future__ import annotations

import csv
import io
import json
import ssl
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from ml.paths import (
    DATA_RAW,
    RAW_ELECTRICITY_SUBSET,
    RAW_METADATA,
    RAW_WEATHER,
    ensure_data_dirs,
)

GITHUB_LFS = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master"
GITHUB_RAW = "https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/master"

METADATA_PATHS = ("data/metadata/metadata.csv",)
WEATHER_PATHS = ("data/weather/weather.csv",)
ELECTRICITY_PATHS = (
    "data/meters/cleaned/electricity_cleaned.csv",
    "data/meters/cleaned/electricity.csv",
)

USER_AGENT = "PowerPredict/0.2 (academic; BDG2 subset download)"


class DownloadError(RuntimeError):
    pass


def _context() -> ssl.SSLContext:
    return ssl.create_default_context()


def _open(url: str, timeout: int = 120):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout, context=_context())


def _try_urls(rel_paths: Iterable[str]) -> tuple[str, object]:
    errors: list[str] = []
    for rel in rel_paths:
        for base in (GITHUB_LFS, GITHUB_RAW):
            url = f"{base}/{rel}"
            try:
                response = _open(url)
                return url, response
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
                errors.append(f"{url}: {exc}")
    raise DownloadError("all download URLs failed:\n" + "\n".join(errors))


def download_to_file(rel_paths: Iterable[str], dest: Path) -> str:
    ensure_data_dirs()
    dest.parent.mkdir(parents=True, exist_ok=True)
    url, response = _try_urls(rel_paths)
    try:
        with dest.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    finally:
        response.close()
    return url


def _header_and_response(rel_paths: Iterable[str]) -> tuple[str, list[str], object]:
    url, response = _try_urls(rel_paths)
    text = io.TextIOWrapper(response, encoding="utf-8", newline="")
    reader = csv.reader(text)
    header = next(reader)
    header = [col.strip() for col in header]
    # Keep wrapper alive by attaching to response
    response._pp_text = text  # type: ignore[attr-defined]
    response._pp_reader = reader  # type: ignore[attr-defined]
    return url, header, response


def stream_electricity_columns(
    keep_building_ids: Iterable[str],
    dest: Path,
    *,
    rel_paths: Iterable[str] = ELECTRICITY_PATHS,
) -> tuple[str, list[str], dict[str, int], int]:
    """Stream the wide electricity CSV and persist only timestamp + requested buildings."""
    ensure_data_dirs()
    wanted = {item.strip() for item in keep_building_ids}
    url, header, response = _header_and_response(rel_paths)
    try:
        if "timestamp" not in header:
            raise DownloadError("electricity CSV has no timestamp column")
        present = [col for col in header if col == "timestamp" or col in wanted]
        if len(present) < 2:
            raise DownloadError("none of the candidate building columns were in the electricity file")
        indices = [header.index(col) for col in present]
        non_null = {col: 0 for col in present if col != "timestamp"}
        n_rows = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(present)
            reader = response._pp_reader  # type: ignore[attr-defined]
            for row in reader:
                if not row:
                    continue
                n_rows += 1
                selected = []
                for col, idx in zip(present, indices, strict=True):
                    value = row[idx] if idx < len(row) else ""
                    selected.append(value)
                    if col != "timestamp" and value.strip() != "":
                        non_null[col] += 1
                writer.writerow(selected)
    finally:
        response.close()
    return url, present, non_null, n_rows


def write_download_manifest(payload: dict) -> Path:
    ensure_data_dirs()
    path = DATA_RAW / "download_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def default_raw_paths() -> dict[str, Path]:
    return {
        "metadata": RAW_METADATA,
        "weather": RAW_WEATHER,
        "electricity_subset": RAW_ELECTRICITY_SUBSET,
    }
