"""Step 1: Inspect the station-shift table.

Checks that data/station_shift_counts.csv has the structure we expect
(columns, number of stations and shifts, date range) before we use it.
Nothing is filtered or modified here.

Run from the repository root:
    uv run python scripts/step1_inspect.py
"""
import pandas as pd

DATA_PATH = "data/station_shift_counts.csv"
EXPECTED_COLUMNS = ["Station", "ShiftDate", "calls", "OutageHours", "Masked"]
NON_COUNT_COLUMNS = ["Station", "ShiftDate", "OutageHours", "Masked"]

df = pd.read_csv(DATA_PATH)

# Basic shape and a first look at the rows
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print(df.head())
print(df.dtypes)

# Make sure every column we rely on later is present
for col in EXPECTED_COLUMNS:
    print(col, "OK" if col in df.columns else "MISSING")

# One row should exist for every (station, shift) pair
df["ShiftDate"] = pd.to_datetime(df["ShiftDate"])
n_stations = df["Station"].nunique()
n_shifts = df["ShiftDate"].nunique()
print("date range:", df["ShiftDate"].min(), "->", df["ShiftDate"].max())
print("stations:", n_stations, "| shifts:", n_shifts)
print("rows == stations x shifts:", len(df) == n_stations * n_shifts)

# Masked marks shifts affected by a data-feed outage (excluded later)
print("Masked values:", df["Masked"].unique())

# Summary statistics for the count columns (calls and the call groups).
# These still include masked rows, so they are not final numbers.
print(df.drop(columns=NON_COUNT_COLUMNS).describe().T)