"""Step 3: Baseline score for Section 3.

The baseline uses no inputs: it always predicts the mean of the training
set. Every model we build later has to beat these test-set numbers.

Split by time (outage shifts excluded):
    train = 2022-2024, validation = 2025, test = 2026

Run from the repository root:
    uv run python scripts/step3_baseline.py
"""
import numpy as np
import pandas as pd

DATA_PATH = "data/station_shift_counts.csv"
TRAIN_END_YEAR = 2024
VAL_YEAR = 2025
TEST_YEAR = 2026

df = pd.read_csv(DATA_PATH)
df["ShiftDate"] = pd.to_datetime(df["ShiftDate"])
d = df[~df["Masked"]].copy()  # drop outage shifts from training and evaluation
year = d["ShiftDate"].dt.year

train = d[year <= TRAIN_END_YEAR]
val = d[year == VAL_YEAR]
test = d[year == TEST_YEAR]
print("rows  train/val/test:", len(train), len(val), len(test))
print("shifts train/val/test:", train["ShiftDate"].nunique(),
      val["ShiftDate"].nunique(), test["ShiftDate"].nunique())
print("test date range:", test["ShiftDate"].min().date(), "->", test["ShiftDate"].max().date())


def report(name, actual, predicted):
    """Print MAE and RMSE for a constant or per-row prediction."""
    error = actual - predicted
    mae = error.abs().mean()
    rmse = np.sqrt((error ** 2).mean())
    print(f"{name}: MAE={mae:.3f}  RMSE={rmse:.3f}")


# Baseline 1: calls per station per shift (the main target)
train_mean = train["calls"].mean()
print(f"\ntraining mean = {train_mean:.3f} | val mean = {val['calls'].mean():.3f} "
      f"| test mean = {test['calls'].mean():.3f}")
report("station-shift VAL ", val["calls"], train_mean)
report("station-shift TEST", test["calls"], train_mean)

# Baseline 2: citywide calls per shift (sum over stations), used for the
# second decision (whether to add a dispatcher)
city = d.groupby("ShiftDate")["calls"].sum().reset_index()
city_year = city["ShiftDate"].dt.year
city_train_mean = city.loc[city_year <= TRAIN_END_YEAR, "calls"].mean()
print(f"\ncitywide training mean = {city_train_mean:.1f}")
report("citywide VAL ", city.loc[city_year == VAL_YEAR, "calls"], city_train_mean)
report("citywide TEST", city.loc[city_year == TEST_YEAR, "calls"], city_train_mean)