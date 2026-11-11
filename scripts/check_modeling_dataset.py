"""Step 5: Sanity checks on data/modeling_dataset.csv (built by step4_features.py).

Checks structure, missing values, possible target leakage, whether the
baseline reproduces the team's numbers, and whether lag features look right.

Run from the repository root:
    uv run python scripts/check_modeling_dataset.py
"""
import re

import pandas as pd

from cv_utils import DATA_PATH, DATE_COL, NON_FEATURES, STATION_COL, TARGET

raw = pd.read_csv(DATA_PATH)
raw[DATE_COL] = pd.to_datetime(raw[DATE_COL])

# 1. Structure and missing values
print("shape:", raw.shape)
print("columns:", raw.columns.tolist())
print("stations:", raw[STATION_COL].nunique(), "| shifts:", raw[DATE_COL].nunique())
print("date range:", raw[DATE_COL].min().date(), "->", raw[DATE_COL].max().date())
missing = raw.isna().sum()
print("\ncolumns with missing values:")
print(missing[missing > 0] if (missing > 0).any() else "none")

# 2. Leakage: call-group columns add up to the target and must not be features
groups = [c for c in ["fire", "medical", "non_emergency", "other"] if c in raw.columns]
if groups:
    same = (raw[groups].sum(axis=1) == raw[TARGET]).all()
    print(f"\ngroup columns {groups} sum to the target: {same} -> exclude them from features")
print("columns excluded from features:", [c for c in NON_FEATURES if c in raw.columns])

# 3. Baseline reproduction. Reference values from the README (after removing the two
#    stopped codes): training mean 9.05, test MAE 5.98, RMSE 9.11 per station-shift;
#    citywide training mean 307.8, test MAE 23.4, RMSE 29.7.
d = raw[~raw["Masked"].astype(str).str.lower().isin(["true", "1"])] if "Masked" in raw.columns else raw
train = d[d[DATE_COL] < "2025-01-01"]
test = d[d[DATE_COL] >= "2026-01-01"]
mean = train[TARGET].mean()
err = test[TARGET] - mean
city_test = test.groupby(DATE_COL)[TARGET].sum()
city_mean = train.groupby(DATE_COL)[TARGET].sum().mean()
city_err = city_test - city_mean
print(f"\nstation-shift: training mean {mean:.2f} (README 9.05) | test MAE {err.abs().mean():.2f} (5.98) "
      f"| test RMSE {(err ** 2).mean() ** 0.5:.2f} (9.11)")
print(f"citywide: training mean {city_mean:.1f} (307.8) | test MAE {city_err.abs().mean():.1f} (23.4) "
      f"| test RMSE {(city_err ** 2).mean() ** 0.5:.1f} (29.7)")

# 4. Lag features: compare with calls from k days earlier (one shift = one day)
calls = raw.set_index([STATION_COL, DATE_COL])[TARGET]
print("\nlag feature check (heuristic: finds columns named like lag3 / lag_7).")
print("A few mismatches next to outage shifts are expected; investigate if many.")
for col in raw.columns:
    found = re.search(r"lag[_ ]?(\d+)", col.lower())
    if not found:
        continue
    k = int(found.group(1))
    earlier = raw.assign(**{DATE_COL: raw[DATE_COL] - pd.Timedelta(days=k)})
    expected = earlier.set_index([STATION_COL, DATE_COL]).index.map(calls.to_dict().get)
    both = raw[col].notna().to_numpy() & pd.Series(expected).notna().to_numpy()
    match = (raw[col].to_numpy()[both] == pd.Series(expected).to_numpy()[both]).mean()
    warn = "  WARNING: forecast is made 3 shifts ahead, so lag < 3 is not available" if k < 3 else ""
    print(f"  {col}: matches calls from {k} days earlier in {match:.1%} of comparable rows{warn}")

windows = [c for c in raw.columns if re.search(r"roll|avg|mean|same_weekday", c.lower())]
if windows:
    print("\nCheck rolling/average columns by hand (SFD 10, first 12 rows):")
    first_station = "SFD 10" if "SFD 10" in raw[STATION_COL].values else raw[STATION_COL].iloc[0]
    print(raw[raw[STATION_COL] == first_station].sort_values(DATE_COL)[[DATE_COL, TARGET] + windows].head(12).to_string())
