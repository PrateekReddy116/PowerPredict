"""Invalid values, conservative outliers, and train-only missing-value handling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.constants import INTERPOLATE_LIMIT_HOURS, IQR_K


def clip_invalid_energy(df: pd.DataFrame, *, energy_col: str = "energy_kwh") -> pd.DataFrame:
    """Replace negative electricity with NaN. BDG2 cleaned files already NaN many issues."""
    out = df.copy()
    energy = pd.to_numeric(out[energy_col], errors="coerce")
    out[energy_col] = energy.where(energy >= 0)
    return out


@dataclass
class TrainIqrWinsorizer:
    """Clip energy using train-only IQR bounds per building. Never fit on val/test."""

    bounds: dict[str, tuple[float, float]]
    energy_col: str = "energy_kwh"
    group_col: str = "building_id"
    k: float = IQR_K

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        *,
        energy_col: str = "energy_kwh",
        group_col: str = "building_id",
        k: float = IQR_K,
    ) -> TrainIqrWinsorizer:
        bounds: dict[str, tuple[float, float]] = {}
        for building_id, group in train.groupby(group_col, sort=False):
            series = pd.to_numeric(group[energy_col], errors="coerce").dropna()
            if series.empty:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                bounds[str(building_id)] = (q1, q3)
            else:
                bounds[str(building_id)] = (q1 - k * iqr, q3 + k * iqr)
        return cls(bounds=bounds, energy_col=energy_col, group_col=group_col, k=k)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        energy = pd.to_numeric(out[self.energy_col], errors="coerce")
        clipped = energy.copy()
        for building_id, (lo, hi) in self.bounds.items():
            mask = out[self.group_col].astype(str) == str(building_id)
            clipped.loc[mask] = energy.loc[mask].clip(lower=lo, upper=hi)
        out[self.energy_col] = clipped
        return out


def interpolate_short_gaps(
    df: pd.DataFrame,
    columns: list[str],
    *,
    group_col: str = "building_id",
    time_col: str = "ts",
    limit: int = INTERPOLATE_LIMIT_HOURS,
) -> pd.DataFrame:
    """Linear interpolate short interior gaps within each group. Does not cross groups."""
    out = df.sort_values([group_col, time_col], kind="mergesort").copy()
    present = [col for col in columns if col in out.columns]
    if not present:
        return out

    def _interp(group: pd.DataFrame) -> pd.DataFrame:
        work = group.copy()
        work[present] = work[present].interpolate(
            method="linear",
            limit=limit,
            limit_area="inside",
        )
        return work

    parts = [_interp(g) for _, g in out.groupby(group_col, sort=False)]
    return pd.concat(parts, ignore_index=True)


@dataclass
class TrainMedianImputer:
    """Column medians from train only. Applied after within-split interpolation."""

    medians: dict[str, float]

    @classmethod
    def fit(cls, train: pd.DataFrame, columns: list[str]) -> TrainMedianImputer:
        medians: dict[str, float] = {}
        for col in columns:
            if col not in train.columns:
                continue
            series = pd.to_numeric(train[col], errors="coerce")
            if series.notna().any():
                medians[col] = float(series.median())
        return cls(medians=medians)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, value in self.medians.items():
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(value)
        return out


def missing_value_summary(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float | int]]:
    n = len(df)
    summary: dict[str, dict[str, float | int]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        summary[col] = {
            "n_missing": n_missing,
            "pct_missing": (100.0 * n_missing / n) if n else 0.0,
        }
    return summary
