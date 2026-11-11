"""Poisson GBM prediction adapter for CallCast Seattle.

Ryan's app_data.py calls:
    predict_shift(feature_rows: pd.DataFrame) -> DataFrame[Station, predicted_calls]

Standalone usage also works:
    predict_shift("2026-10-24")          # all stations on a date
    predict_shift("2026-10-24", "SFD 10") # one station

Historical model predictions are only permitted for targets at least 3 shifts
later than the last training label. The future 3-shift feature CSV is supported
by the standalone date API; Ryan's current date picker remains historical-only.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent  # predict.py belongs in repo root
HISTORY_PATH = ROOT / "data" / "modeling_dataset.csv"
FORECAST_PATH = ROOT / "data" / "forecast_features.csv"
MODEL_PATH = ROOT / "models" / "poisson_gbm.joblib"
STATION_COL, DATE_COL = "Station", "ShiftDate"
_cache = {}


def _bundle():
    if "bundle" not in _cache:
        _cache["bundle"] = joblib.load(MODEL_PATH)
    return _cache["bundle"]


def _data():
    if "data" not in _cache:
        history = pd.read_csv(HISTORY_PATH).assign(mode="historical")
        frames = [history]
        if FORECAST_PATH.exists():
            frames.append(pd.read_csv(FORECAST_PATH).assign(mode="forecast"))
        data = pd.concat(frames, ignore_index=True)
        data[DATE_COL] = pd.to_datetime(data[DATE_COL])
        data["weekday"] = data[DATE_COL].dt.dayofweek
        _cache["data"] = data
    return _cache["data"]


def _earliest_model_date(bundle):
    # TRAIN_END is exclusive; last training shift is the preceding calendar day.
    # Ryan requires trained_through <= target - 3 days.
    return pd.Timestamp(bundle["train_end"]) + pd.Timedelta(days=2)


def available_dates():
    """Available dates for *out-of-sample* model predictions (not training dates)."""
    first = _earliest_model_date(_bundle())
    return sorted(_data().loc[_data()[DATE_COL] >= first, DATE_COL].dt.date.unique())


def supported_dates():
    dates = available_dates()
    if not dates:
        raise ValueError("No supported out-of-sample prediction dates")
    return dates[0], dates[-1]


def _predict_rows(rows):
    """Apply the exact saved training schema to raw model feature rows."""
    bundle = _bundle()
    required = {STATION_COL, "weekday", *bundle["numeric"]}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Missing prediction features: {sorted(missing)}")
    if rows.empty:
        raise ValueError("No feature rows to predict")
    if rows[STATION_COL].isna().any() or rows[STATION_COL].duplicated().any():
        raise ValueError("Prediction requires one row per non-null Station")
    numeric = rows[bundle["numeric"]].astype(float)
    X = pd.concat(
        [numeric,
         pd.get_dummies(rows[STATION_COL], prefix="st", dtype=float),
         pd.get_dummies(rows["weekday"], prefix="wd", dtype=float)],
        axis=1,
    )
    # fill_value=0 adds absent one-hot categories; it does not fill existing NaNs.
    X = X.reindex(columns=bundle["columns"], fill_value=0.0)
    pred = np.asarray(bundle["model"].predict(X), dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("Model returned non-finite predictions")
    return pd.DataFrame({
        "Station": rows[STATION_COL].to_numpy(),
        "predicted_calls": np.maximum(pred, 0.0),
    }).sort_values("Station").reset_index(drop=True)


def predict_shift(feature_rows, station=None):
    """Predict using either Ryan's DataFrame interface or a standalone shift date.

    DataFrame input: returns Station, predicted_calls (Ryan's required schema).
    Date input: returns those columns plus date, usual, actual and mode for review.
    """
    if isinstance(feature_rows, pd.DataFrame):
        if station is not None:
            raise ValueError("station argument is only supported with a date input")
        return _predict_rows(feature_rows)

    day = pd.Timestamp(feature_rows).normalize()
    earliest = _earliest_model_date(_bundle())
    if day < earliest:
        raise ValueError(
            f"{day.date()} is before the first out-of-sample model date "
            f"{earliest.date()}; use historical reference or a separately "
            "trained backtest model for earlier dates."
        )
    data = _data()
    rows = data.loc[data[DATE_COL] == day].copy()
    if station is not None:
        rows = rows.loc[rows[STATION_COL] == station]
    if rows.empty:
        raise ValueError(f"No usable feature rows for {day.date()}")

    result = _predict_rows(rows)
    details = rows[[STATION_COL, DATE_COL, "calls", "mode", "weekday"]].copy()
    usual = _bundle()["usual"]
    details["usual_calls"] = [
        usual.get((s, int(w)), np.nan)
        for s, w in zip(details[STATION_COL], details["weekday"])
    ]
    details = details.rename(columns={"calls": "actual_calls"})
    return result.merge(
        details.drop(columns=["weekday"]), on=STATION_COL, validate="one_to_one"
    ).assign(expected_calls=lambda d: d["predicted_calls"])[
        ["Station", "ShiftDate", "predicted_calls", "expected_calls",
         "usual_calls", "actual_calls", "mode"]
    ]
