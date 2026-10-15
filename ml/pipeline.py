"""Phase 2 reproducible pipeline: load → clean → features → 70/15/15 split."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ml.constants import (
    DATA_ORIGIN_HISTORICAL,
    INTERPOLATE_LIMIT_HOURS,
    IQR_K,
    LAGS,
    MIN_COMPLETENESS,
    N_BUILDINGS,
    PREFERRED_PRIMARY_USES,
    ROLLING_WINDOWS,
    TRAIN_FRAC,
    VAL_FRAC,
)
from ml.features.lags import add_lag_features
from ml.features.occupancy import add_simulated_occupancy
from ml.features.rolling import ROLLING_INCLUDES_CURRENT_T, add_rolling_mean_features
from ml.features.splits import GlobalCutoffs, chronological_split, chronological_split_fractions
from ml.features.temporal import add_temporal_features
from ml.ingest.load import (
    electricity_completeness,
    load_electricity_wide,
    load_metadata,
    load_weather,
    melt_electricity,
    normalize_metadata,
)
from ml.ingest.select import BuildingChoice, choices_frame, select_buildings
from ml.paths import DATA_PROCESSED, ensure_data_dirs
from ml.preprocess.quality import (
    TrainIqrWinsorizer,
    TrainMedianImputer,
    clip_invalid_energy,
    interpolate_short_gaps,
    missing_value_summary,
)
from ml.preprocess.timestamps import convert_frame_local_to_utc, sort_chronologically
from ml.preprocess.weather import (
    WEATHER_FEATURE_COLUMNS,
    add_humidity,
    align_weather,
    normalize_weather_columns,
)

LAG_COLUMNS = [f"lag_{lag}" for lag in LAGS]
ROLLING_COLUMNS = [f"rolling_mean_{window}" for window in ROLLING_WINDOWS]
ROLLING_EXTRA_COLUMNS = ["rolling_std_24"]
TEMPORAL_COLUMNS = [
    "hour",
    "day",
    "day_of_week",
    "month",
    "weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]
FEATURE_COLUMNS = [
    "energy_kwh",
    *LAG_COLUMNS,
    *ROLLING_COLUMNS,
    *ROLLING_EXTRA_COLUMNS,
    *TEMPORAL_COLUMNS,
    *WEATHER_FEATURE_COLUMNS,
    "occupancy",
    "sqm",
]
IMPUTE_COLUMNS = ["energy_kwh", *WEATHER_FEATURE_COLUMNS]
MODEL_DROPNA_COLUMNS = [*LAG_COLUMNS, *ROLLING_COLUMNS, *ROLLING_EXTRA_COLUMNS, "target_energy_kwh", "energy_kwh"]


@dataclass
class PipelineResult:
    source: str
    buildings: pd.DataFrame
    readings: pd.DataFrame
    features: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cuts: GlobalCutoffs
    summary: dict[str, Any] = field(default_factory=dict)
    choices: list[BuildingChoice] = field(default_factory=list)


def add_target(df: pd.DataFrame, *, energy_col: str = "energy_kwh", group_col: str = "building_id") -> pd.DataFrame:
    """y_{t+1}: next-hour energy. Feature row at t must not use this column as an input."""
    out = df.copy()
    out["target_energy_kwh"] = out.groupby(group_col, sort=False)[energy_col].shift(-1)
    return out


def _attach_building_meta(energy: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [col for col in ("building_id", "site_id", "primary_use", "sqm", "timezone") if col in metadata.columns]
    return energy.merge(metadata.loc[:, meta_cols], on="building_id", how="left")


def build_aligned_readings(
    metadata: pd.DataFrame,
    electricity_wide: pd.DataFrame,
    weather: pd.DataFrame,
    choices: list[BuildingChoice],
) -> pd.DataFrame:
    ids = [choice.building_id for choice in choices]
    wide = electricity_wide.loc[:, ["timestamp", *[col for col in ids if col in electricity_wide.columns]]]
    long = melt_electricity(wide)
    long = _attach_building_meta(long, metadata)
    if long["timezone"].isna().any():
        raise ValueError("selected buildings are missing timezone in metadata")
    long = convert_frame_local_to_utc(long, group_cols=("building_id",))
    long = clip_invalid_energy(long)
    long["data_origin"] = DATA_ORIGIN_HISTORICAL

    weather_n = normalize_weather_columns(weather)
    site_tz = metadata.loc[:, ["site_id", "timezone"]].drop_duplicates("site_id")
    weather_n = weather_n.merge(site_tz, on="site_id", how="left")
    sites = {choice.site_id for choice in choices}
    weather_n = weather_n.loc[weather_n["site_id"].isin(sites)].copy()
    if weather_n["timezone"].isna().any():
        raise ValueError("weather rows missing site timezone")
    weather_n = convert_frame_local_to_utc(weather_n, group_cols=("site_id",))
    weather_n = add_humidity(weather_n)
    keep_weather = ["site_id", "ts", *[col for col in WEATHER_FEATURE_COLUMNS if col in weather_n.columns]]
    if "dew_temperature" in weather_n.columns and "dew_temperature" not in keep_weather:
        keep_weather.append("dew_temperature")
    extra = [col for col in ("air_temperature", "dew_temperature", "humidity", "wind_speed", "precip_depth_1hr") if col in weather_n.columns]
    keep_weather = list(dict.fromkeys(["site_id", "ts", *extra]))
    aligned = align_weather(long, weather_n.loc[:, keep_weather])
    aligned = sort_chronologically(aligned, time_col="ts")
    return aligned


def _apply_split_cleaning(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    winsorize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stats: dict[str, Any] = {"interpolate_limit_hours": INTERPOLATE_LIMIT_HOURS, "iqr_k": IQR_K, "winsorize": winsorize}
    if winsorize:
        clipper = TrainIqrWinsorizer.fit(train, k=IQR_K)
        stats["winsorize_bounds"] = {key: list(val_) for key, val_ in clipper.bounds.items()}
        train = clipper.transform(train)
        val = clipper.transform(val)
        test = clipper.transform(test)

    impute_cols = [col for col in IMPUTE_COLUMNS if col in train.columns]
    train = interpolate_short_gaps(train, impute_cols, limit=INTERPOLATE_LIMIT_HOURS)
    val = interpolate_short_gaps(val, impute_cols, limit=INTERPOLATE_LIMIT_HOURS)
    test = interpolate_short_gaps(test, impute_cols, limit=INTERPOLATE_LIMIT_HOURS)
    imputer = TrainMedianImputer.fit(train, impute_cols)
    stats["train_medians"] = imputer.medians
    train = imputer.transform(train)
    val = imputer.transform(val)
    test = imputer.transform(test)
    train = add_humidity(train)
    val = add_humidity(val)
    test = add_humidity(test)
    return train, val, test, stats


def add_training_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate canonical Phase 2 names to the Phase 3 feature-matrix contract names."""
    out = df.copy()
    aliases = {
        "target_energy_kwh": "y_next",
        "weekend": "is_weekend",
        "rolling_mean_24": "roll_mean_24",
        "rolling_mean_168": "roll_mean_168",
        "rolling_std_24": "roll_std_24",
        "air_temperature": "air_temperature_c",
        "dew_temperature": "dew_temperature_c",
    }
    for src, dest in aliases.items():
        if src in out.columns and dest not in out.columns:
            out[dest] = out[src]
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_temporal_features(df)
    out = add_simulated_occupancy(out)
    out = add_lag_features(out, lags=LAGS)
    out = add_rolling_mean_features(out, windows=ROLLING_WINDOWS)
    out = add_target(out)
    return add_training_aliases(out)


def dropna_model_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in MODEL_DROPNA_COLUMNS if col in df.columns]
    return df.dropna(subset=cols).reset_index(drop=True)


def run_pipeline(
    metadata: pd.DataFrame,
    electricity_wide: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    source: str,
    n_buildings: int = N_BUILDINGS,
    min_completeness: float = MIN_COMPLETENESS,
    winsorize: bool = True,
) -> PipelineResult:
    completeness = electricity_completeness(electricity_wide)
    metadata = normalize_metadata(metadata)
    choices = select_buildings(
        metadata,
        completeness,
        n=n_buildings,
        min_completeness=min_completeness,
        primary_uses=PREFERRED_PRIMARY_USES,
    )
    buildings = choices_frame(choices)
    readings = build_aligned_readings(metadata, electricity_wide, weather, choices)
    missing_before = missing_value_summary(readings, ["energy_kwh", *WEATHER_FEATURE_COLUMNS])

    train_raw, val_raw, test_raw, cuts = chronological_split_fractions(
        readings,
        time_col="ts",
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
    )
    train_c, val_c, test_c, clean_stats = _apply_split_cleaning(train_raw, val_raw, test_raw, winsorize=winsorize)

    cleaned = pd.concat([train_c, val_c, test_c], ignore_index=True)
    cleaned = cleaned.sort_values(["building_id", "ts"], kind="mergesort").reset_index(drop=True)
    featured = engineer_features(cleaned)
    train_f, val_f, test_f = chronological_split(
        featured,
        time_col="ts",
        train_end=cuts.train_end,
        val_end=cuts.val_end,
    )
    train_f = train_f.copy()
    val_f = val_f.copy()
    test_f = test_f.copy()
    train_f["split"] = "train"
    val_f["split"] = "val"
    test_f["split"] = "test"
    features = pd.concat([train_f, val_f, test_f], ignore_index=True)
    features = features.sort_values(["building_id", "ts"], kind="mergesort").reset_index(drop=True)

    train_m = dropna_model_rows(train_f)
    val_m = dropna_model_rows(val_f)
    test_m = dropna_model_rows(test_f)

    summary = {
        "source": source,
        "n_buildings": len(choices),
        "building_ids": [c.building_id for c in choices],
        "buildings": [
            {
                "id": c.building_id,
                "site_id": c.site_id,
                "primary_use": c.primary_use,
                "timezone": c.timezone,
                "completeness": c.completeness,
                "selected_reason": c.selected_reason,
            }
            for c in choices
        ],
        "records": {
            "raw_electricity_wide_rows": int(len(electricity_wide)),
            "raw_electricity_wide_building_cols": int(electricity_wide.shape[1] - 1),
            "aligned_hourly": int(len(readings)),
            "aligned_with_energy": int(readings["energy_kwh"].notna().sum()),
            "features_before_dropna": int(len(features)),
            "features_after_dropna": int(len(train_m) + len(val_m) + len(test_m)),
            "train_after_dropna": int(len(train_m)),
            "val_after_dropna": int(len(val_m)),
            "test_after_dropna": int(len(test_m)),
        },
        "missing_before_impute": missing_before,
        "missing_after_impute": missing_value_summary(
            pd.concat([train_c, val_c, test_c], ignore_index=True),
            ["energy_kwh", *WEATHER_FEATURE_COLUMNS],
        ),
        "split": {
            "method": "global_unique_timestamp_fractions",
            "train_frac_requested": TRAIN_FRAC,
            "val_frac_requested": VAL_FRAC,
            "test_frac_requested": 1.0 - TRAIN_FRAC - VAL_FRAC,
            "train_end": str(cuts.train_end),
            "val_end": str(cuts.val_end),
            "n_unique_timestamps": cuts.n_unique_timestamps,
            "train_frac_actual": cuts.train_frac_actual,
            "val_frac_actual": cuts.val_frac_actual,
            "test_frac_actual": cuts.test_frac_actual,
            "calendar_split_not_used": True,
            "calendar_train_end_legacy": "2017-06-30T23:00:00Z",
            "calendar_val_end_legacy": "2017-09-30T23:00:00Z",
        },
        "features": {
            "target": "target_energy_kwh (y_{t+1})",
            "lags": list(LAGS),
            "rolling_means": list(ROLLING_WINDOWS),
            "rolling_includes_current_t": ROLLING_INCLUDES_CURRENT_T,
            "temporal": TEMPORAL_COLUMNS,
            "environmental": list(WEATHER_FEATURE_COLUMNS),
            "occupancy": True,
            "occupancy_source": "simulated",
            "columns": FEATURE_COLUMNS,
        },
        "preprocess": {
            "bdg2_cleaning_cited": (
                "BDG2 cleaned electricity already replaced S-H-ESD outliers, "
                "zero runs longer than 24h, and electricity zeros with NaN."
            ),
            "invalid_energy": "clip negatives to NaN",
            "timestamps": "BDG2 naive local (site timezone in metadata) → UTC for storage; temporal features from ts_local",
            **clean_stats,
        },
        "leakage": {
            "target": "shift(-1) per building — label only",
            "lags": "shift(k) for k in lags — past only",
            "rolling": "shift(1) then rolling; current t excluded",
            "imputers_and_iqr": "fit on train split only",
        },
    }
    return PipelineResult(
        source=source,
        buildings=buildings,
        readings=readings,
        features=features,
        train=train_m,
        val=val_m,
        test=test_m,
        cuts=cuts,
        summary=summary,
        choices=choices,
    )


def persist_processed(result: PipelineResult, dest: Path | None = None) -> dict[str, str]:
    ensure_data_dirs()
    root = dest or DATA_PROCESSED
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def _write(name: str, frame: pd.DataFrame) -> None:
        parquet = root / f"{name}.parquet"
        csv_path = root / f"{name}.csv"
        try:
            frame.to_parquet(parquet, index=False)
            written[name] = str(parquet)
        except Exception:
            frame.to_csv(csv_path, index=False)
            written[name] = str(csv_path)

    _write("buildings", result.buildings)
    _write("readings_hourly", result.readings)
    _write("features", result.features)
    _write("features_train", result.train)
    _write("features_val", result.val)
    _write("features_test", result.test)
    summary_path = root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(result.summary, indent=2, default=str), encoding="utf-8")
    written["pipeline_summary"] = str(summary_path)
    return written


def load_from_paths(metadata_path: Path, electricity_path: Path, weather_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_metadata(metadata_path), load_electricity_wide(electricity_path), load_weather(weather_path)
