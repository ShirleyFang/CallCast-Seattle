"""Step 2: Numbers and figure for Section 2 (data reality check).

Removes masked (outage) shifts, then reports the target distribution,
saves a histogram, shows one example row, and measures the drop in
non-emergency calls after the 2025-05-31 break.

Run from the repository root:
    uv run python scripts/step2_sec2.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = "data/station_shift_counts.csv"
FIGURE_PATH = "figures/target_dist_station_shift_forchecka_candelete.png"
GROUP_COLUMNS = ["fire", "medical", "non_emergency", "other"]
BREAK_DATE = "2025-05-31"  # last day before two call types stopped appearing
RANDOM_SEED = 42

df = pd.read_csv(DATA_PATH)
df["ShiftDate"] = pd.to_datetime(df["ShiftDate"])

# 1. Sanity check: the total must equal the sum of the four call groups
assert (df[GROUP_COLUMNS].sum(axis=1) == df["calls"]).all(), "calls != sum of groups"
print("calls == sum of groups: OK")

# 2. Remove shifts with a data-feed outage (they would show fake low counts)
n_masked_rows = df["Masked"].sum()
n_masked_shifts = df.loc[df["Masked"], "ShiftDate"].nunique()
print(f"masked rows: {n_masked_rows} ({df['Masked'].mean():.1%}) | masked shifts: {n_masked_shifts}")
d = df[~df["Masked"]].copy()
print("rows kept:", len(d), "| shifts kept:", d["ShiftDate"].nunique())

# 3. Distribution of the target (calls per station per shift)
y = d["calls"]
print(y.describe())
print("median:", y.median(), "| zero share:", f"{(y == 0).mean():.1%}",
      "| variance/mean:", round(y.var() / y.mean(), 2))
print("share with 25+ calls:", f"{(y >= 25).mean():.1%}")

# Histogram; the x-axis stops at the 99.9th percentile to hide extreme outliers
os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(y, bins=range(0, int(y.quantile(0.999)) + 2))
ax.set_xlabel("Calls in one 08:00-08:00 shift (one station area)")
ax.set_ylabel("Station-shifts")
ax.set_title("Calls per station per shift, 2022-2026 (outage shifts excluded)")
fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")

# 4. One random row for the "one row, explained" paragraph (fixed seed = repeatable)
print(d.sample(1, random_state=RANDOM_SEED).T)

# 5. The surprise: compare average counts before and after the break
d["period"] = np.where(d["ShiftDate"] <= BREAK_DATE, "before", "after")
print(d.groupby("period")[GROUP_COLUMNS + ["calls"]].mean().round(2))

citywide = d.groupby("ShiftDate")["calls"].sum()  # sum over stations per shift
print("citywide calls/shift  before:", round(citywide[citywide.index <= BREAK_DATE].mean(), 1),
      "| after:", round(citywide[citywide.index > BREAK_DATE].mean(), 1))

# Average calls per shift by station (top five)
print("\nAverage calls per shift, top five stations:")
print(d.groupby("Station")["calls"].mean().sort_values(ascending=False).head(5))