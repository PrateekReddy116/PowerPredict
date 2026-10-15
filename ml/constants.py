"""Shared Phase 2 constants. Target is next-hour electricity (y_{t+1})."""

from __future__ import annotations

DATA_ORIGIN_HISTORICAL = "historical"
OCCUPANCY_SOURCE_SIMULATED = "simulated"

PREFERRED_PRIMARY_USES = ("Education", "Office")
N_BUILDINGS = 3
MIN_COMPLETENESS = 0.90

LAGS = (1, 2, 3, 6, 12, 24, 168)
ROLLING_WINDOWS = (3, 6, 24, 168)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Short-gap interpolation within a split only (hours). Remaining NaNs use train medians.
INTERPOLATE_LIMIT_HOURS = 6
# Conservative train-only winsorization (BDG2 already NaN'd S-H-ESD outliers and long zeros).
IQR_K = 3.0

# Magnus coefficients for relative humidity from air / dew temperature (°C).
MAGNUS_A = 17.625
MAGNUS_B = 243.04
