"""Shared helpers for model evaluation: data loading, features, folds, metrics.

Import from other scripts in this folder, for example:
    from cv_utils import load_dataset, build_features, iter_folds, metrics
"""
import numpy as np
import pandas as pd

DATA_PATH = "data/modeling_dataset.csv"
TARGET = "calls"
STATION_COL = "Station"
DATE_COL = "ShiftDate"

# Forecasts are made MIN_LAG shifts ahead (same value as MIN_LAG in step4_features.py).
# At the forecast origin the labels of the last MIN_LAG - 1 shifts before a validation
# period are not known yet, so training stops that many shifts earlier (a gap).
MIN_LAG = 3
GAP_SHIFTS = MIN_LAG - 1

# Never use these as features: they are parts of the target or outage flags.
NON_FEATURES = [TARGET, "fire", "medical", "non_emergency", "other", "OutageHours", "Masked"]

# Rolling CV over 2025: quarterly validation folds (start inclusive, end exclusive).
# Each fold trains on every shift before its start date. 2026 is the final test
# set and must not be used while comparing models or feature sets.
FOLDS = [("2025-01-01", "2025-04-01"), ("2025-04-01", "2025-07-01"),
         ("2025-07-01", "2025-10-01"), ("2025-10-01", "2026-01-01")]


def load_dataset(path=DATA_PATH):
    """Read the modeling table, drop outage rows if flagged, add a weekday column."""
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    if "Masked" in df.columns:
        df = df[~df["Masked"].astype(str).str.lower().isin(["true", "1"])]
    df = df.sort_values([DATE_COL, STATION_COL]).reset_index(drop=True)
    df["weekday"] = df[DATE_COL].dt.dayofweek
    return df


def build_features(df, extra_exclude=()):
    """Return (X, numeric_columns): numeric features plus station and weekday dummies."""
    excluded = set(NON_FEATURES) | set(extra_exclude) | {"weekday"}
    numeric = [c for c in df.select_dtypes(include=["number", "bool"]).columns if c not in excluded]
    X = pd.concat([
        df[numeric].astype(float),
        pd.get_dummies(df[STATION_COL], prefix="st", dtype=float),
        pd.get_dummies(df["weekday"], prefix="wd", dtype=float),
    ], axis=1)
    # Leakage guard: no feature may be an exact copy of the target
    for col in X.columns:
        if np.allclose(X[col].fillna(-1), df[TARGET]):
            raise ValueError(f"Feature {col} is identical to the target")
    return X, numeric


def iter_folds(df, folds=FOLDS, gap=GAP_SHIFTS):
    """Yield (fold_start, train_rows, validation_rows) for each rolling-CV fold.

    Training uses shifts before fold_start minus `gap` shifts, so every training
    label would already be known when the first validation shift is forecast.
    """
    for start, end in folds:
        train_end = pd.Timestamp(start) - pd.Timedelta(days=gap)
        yield start, df[df[DATE_COL] < train_end], df[(df[DATE_COL] >= start) & (df[DATE_COL] < end)]


def metrics(val, pred):
    """Station MAE, station RMSE and citywide MAE (sum of stations per shift)."""
    error = val[TARGET].to_numpy() - pred
    city = val.assign(pred=pred).groupby(DATE_COL)[[TARGET, "pred"]].sum()
    return {"station_MAE": np.abs(error).mean(),
            "station_RMSE": np.sqrt((error ** 2).mean()),
            "citywide_MAE": (city[TARGET] - city["pred"]).abs().mean()}
