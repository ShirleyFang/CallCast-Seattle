"""Read the CSVs and prepare the counts, model inputs, and hourly data shown by app.py."""

import importlib.util
import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Find data/ beside this file, regardless of where the app was started.
ROOT = Path(__file__).resolve().parent
# These count table columns describe the four call categories shown in history charts.
GROUPS = ["fire", "medical", "non_emergency", "other"]


def load_tables(root=ROOT):
    """Read station locations, past call totals, and model input rows (past and forecast) from four CSVs."""
    # sfd_stations.csv: names and coordinates for the station menu and map.
    stations = pd.read_csv(root / "data/sfd_stations.csv")
    # station_shift_counts.csv: past totals, categories, and suspected outage flags.
    counts = pd.read_csv(root / "data/station_shift_counts.csv")
    # modeling_dataset.csv: calendar and earlier call inputs, plus the calls label.
    features = pd.read_csv(root / "data/modeling_dataset.csv")
    # Check that each table has rows and the columns the app needs.
    required = [
        (stations, {"station", "lat", "lon"}, "stations"),
        (
            counts,
            {"Station", "ShiftDate", "calls", "Masked", "OutageHours", *GROUPS},
            "counts",
        ),
        (
            features,
            {"Station", "ShiftDate", "calls", "weekday", "month", "year"},
            "features",
        ),
    ]
    for table, columns, name in required:
        if table.empty or columns - set(table.columns):
            raise ValueError(f"Empty table or missing required columns in {name}.")
    # Station names must match the form used in all three tables, such as SFD 10.
    if (
        stations["station"].isna().any()
        or not stations["station"].str.fullmatch(r"SFD \d+").all()
    ):
        raise ValueError(
            "Station identifiers must use the form SFD followed by a number."
        )
    # Reject missing or impossible coordinates before drawing the station map.
    for col, limit in [("lat", 90), ("lon", 180)]:
        stations[col] = pd.to_numeric(stations[col], errors="raise")
        if (
            stations[col].isna().any()
            or not np.isfinite(stations[col]).all()
            or (stations[col].abs() > limit).any()
        ):
            raise ValueError(
                "Station coordinates must be finite and in geographic bounds."
            )
    # Read ShiftDate as a date in both tables so they can be matched correctly.
    for table in [counts, features]:
        dates = pd.to_datetime(table["ShiftDate"], format="%Y-%m-%d", errors="raise")
        if (
            dates.isna().any()
            or dates.dt.tz is not None
            or not dates.eq(dates.dt.normalize()).all()
            or table["Station"].isna().any()
        ):
            raise ValueError(
                "Each row needs a station and a start date without a time zone."
            )
        table["ShiftDate"] = dates
    # Call and outage counts must be whole numbers, not blanks or negative values.
    for col in ["calls", "OutageHours", *GROUPS]:
        counts[col] = pd.to_numeric(counts[col], errors="raise")
        if (
            counts[col].isna().any()
            or not np.isfinite(counts[col]).all()
            or (counts[col] < 0).any()
            or (counts[col] % 1 != 0).any()
        ):
            raise ValueError(f"{col} must contain whole numbers of 0 or more.")
    # Check that fire + medical + non_emergency + other equals each row's total calls.
    if not counts[GROUPS].sum(axis=1).eq(counts["calls"]).all():
        raise ValueError("Call groups do not sum to calls.")
    if stations["station"].duplicated().any():
        raise ValueError("Duplicate station identifiers in sfd_stations.csv")
    # Turn the CSV's True/False flags into values Python can use to exclude unreliable shifts.
    mask = counts["Masked"].astype(str).str.lower().map({"true": True, "false": False})
    if mask.isna().any():
        raise ValueError("Masked must contain only True or False.")
    counts["Masked"] = mask.astype(bool)
    # Each station and shift must appear only once; duplicate rows could inflate displayed results.
    for name, table in [("counts", counts), ("features", features)]:
        if table.duplicated(["Station", "ShiftDate"]).any():
            raise ValueError(f"Duplicate station and shift rows in {name}.")
    if set(counts["Station"]) != set(stations["station"]):
        raise ValueError("Station identifiers do not match the map data.")
    # Require a row for every station and date, including shifts with genuinely zero calls.
    dates = pd.date_range(counts["ShiftDate"].min(), counts["ShiftDate"].max())
    if len(counts) != len(stations) * len(dates):
        raise ValueError(
            "Counts must include every station and calendar shift, including zeros."
        )
    # A citywide feed gap should mark the same shift unreliable for every station.
    if not counts["Masked"].eq(counts["OutageHours"] > 0).all():
        raise ValueError("Masked flags disagree with OutageHours.")
    mask_by_date = counts.groupby("ShiftDate")["Masked"].nunique()
    if (mask_by_date != 1).any():
        raise ValueError(
            "Citywide outage masks must agree across stations for a shift."
        )
    # Check modeling_dataset.csv has exactly the usable count table rows and the same labels.
    expected = (
        counts.loc[~counts["Masked"]]
        .set_index(["Station", "ShiftDate"])["calls"]
        .sort_index()
    )
    actual = features.set_index(["Station", "ShiftDate"])["calls"].sort_index()
    if (
        not actual.index.equals(expected.index)
        or not pd.to_numeric(actual, errors="raise").eq(expected).all()
    ):
        raise ValueError(
            "Modeling rows/targets do not match the count table rows without an outage."
        )
    # The shifts after the last count date have model inputs but no calls yet.
    forecast = read_forecast_features(
        root, set(stations["station"]), counts["ShiftDate"].max(), features.columns
    )
    if forecast is not None:
        features = pd.concat([features, forecast], ignore_index=True)
    return stations, counts.sort_values(["Station", "ShiftDate"]), features


def read_forecast_features(root, station_ids, last_count, columns):
    """Read forecast_features.csv, check it, and return its rows (None if the file is absent)."""
    path = root / "data/forecast_features.csv"
    if not path.exists():
        return None
    forecast = pd.read_csv(path, parse_dates=["ShiftDate"])
    if forecast.empty or set(columns) - set(forecast.columns):
        raise ValueError("forecast_features.csv is empty or missing columns.")
    # These shifts have not happened, so they cannot have a call count.
    if forecast["calls"].notna().any():
        raise ValueError(
            "forecast_features.csv must have blank calls: those shifts have not happened yet."
        )
    if forecast["Station"].isna().any() or set(forecast["Station"]) != station_ids:
        raise ValueError(
            "forecast_features.csv stations do not match the station list."
        )
    per_date = forecast.groupby("ShiftDate")["Station"].nunique()
    if (
        forecast.duplicated(["Station", "ShiftDate"]).any()
        or (per_date != len(station_ids)).any()
    ):
        raise ValueError(
            "forecast_features.csv needs exactly one row for every station on each forecast date."
        )
    # The first forecast date is the day after the last counts, with no gap after it.
    expected = pd.date_range(last_count + pd.Timedelta(days=1), periods=len(per_date))
    if not per_date.index.equals(expected):
        raise ValueError(
            "Forecast dates must follow the last day of counts, one day after another."
        )
    return forecast[list(columns)]


def first_model_date(root=ROOT):
    # First target date the trained model can serve, or None if its dates can't be read.
    # Same rule as learned_predictions: a target must be at least 3 days after the
    # model's last training shift, so earlier dates would only replay its training data.
    try:
        metadata = json.loads((root / "model_metadata.json").read_text("utf-8"))
        trained = pd.Timestamp(date.fromisoformat(metadata["trained_through"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (trained + pd.Timedelta(days=3)).date()


def target_dates(counts, earliest=None, latest=None):
    """Use the loaded count table to choose the date picker's range and starting date.

    earliest is the first date the model can serve; the picker starts there if it is later.
    latest is the last forecast date; the picker ends there if it is later than the counts.
    """
    # Leave an initial month of history before the first selectable target shift.
    first = counts["ShiftDate"].min().normalize() + pd.Timedelta(days=31)
    if earliest is not None:
        first = max(first, pd.Timestamp(earliest).normalize())
    last = counts["ShiftDate"].max().normalize()
    if latest is not None:
        last = max(last, pd.Timestamp(latest).normalize())
    if first > last:
        raise ValueError(
            "At least 32 calendar shifts are needed for this historical demo."
        )
    # Prefer the example date, but keep it inside whichever snapshot is installed.
    default = min(max(pd.Timestamp("2026-10-24"), first), last)
    return first.date(), last.date(), default.date()


def forecast_origin(target):
    # Planning time is 08:00 Seattle time two days before the shift (Thursday for Saturday).
    day = pd.Timestamp(target).normalize()
    return (day - pd.Timedelta(days=2) + pd.Timedelta(hours=8)).tz_localize(
        "America/Los_Angeles"
    )


def known_history(counts, target):
    # Keep shifts that finished by planning time (Wednesday for a Saturday) and drop outage shifts.
    latest_complete = pd.Timestamp(target).normalize() - pd.Timedelta(days=3)
    return counts.loc[
        (counts["ShiftDate"] <= latest_complete) & ~counts["Masked"]
    ].copy()


def reference_predictions(counts, target, station_ids):
    """Average earlier matching weekdays from the loaded station_shift_counts.csv table."""
    history = known_history(counts, target)
    if history.empty:
        raise ValueError(
            "No completed, unmasked history exists before this prediction time."
        )
    # For a Saturday target, select only earlier Saturdays with usable counts.
    same_day = history[
        history["ShiftDate"].dt.dayofweek == pd.Timestamp(target).dayofweek
    ]
    # One average per station.
    usual = same_day.groupby("Station")["calls"].mean()
    missing = set(station_ids) - set(usual.index)
    if missing:
        raise ValueError(
            "Not enough same weekday history for: " + ", ".join(sorted(missing))
        )
    return usual.reindex(station_ids).rename("predicted_calls").reset_index()


def validate_predictions(result, station_ids):
    """Check an estimate table before app.py uses it for totals, the map, and downloads."""
    if not isinstance(result, pd.DataFrame) or not {
        "Station",
        "predicted_calls",
    } <= set(result.columns):
        raise ValueError(
            "predict_shift must return a DataFrame with Station and predicted_calls."
        )
    result = result[["Station", "predicted_calls"]].copy()
    if result["Station"].duplicated().any() or set(result["Station"]) != set(
        station_ids
    ):
        raise ValueError(
            "Prediction output must contain exactly one row for every station."
        )
    # Every station needs a real count; missing, infinite, or negative estimates are invalid.
    if result["predicted_calls"].isna().any() or pd.api.types.is_bool_dtype(
        result["predicted_calls"]
    ):
        raise ValueError("Predicted counts must be numbers, present, and 0 or more.")
    result["predicted_calls"] = pd.to_numeric(result["predicted_calls"], errors="raise")
    if (
        not np.isfinite(result["predicted_calls"]).all()
        or (result["predicted_calls"] < 0).any()
    ):
        raise ValueError("Predicted counts must be finite and 0 or more.")
    return result


def _load_predict_module(path):
    """Load predict.py once; it keeps the model in memory, so a fresh copy per click would reload it."""
    info = path.stat()
    return _predict_module(path, info.st_mtime_ns, info.st_size)


@lru_cache(maxsize=4)
def _predict_module(path, mtime_ns, size):
    # mtime_ns and size are only cache keys: editing predict.py loads a fresh copy.
    spec = importlib.util.spec_from_file_location("callcast_predict", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def learned_predictions(features, target, station_ids, root=ROOT):
    """Send loaded modeling_dataset.csv inputs to predict.py and check the returned counts."""
    # predict.py runs the saved model; the JSON describes its inputs and training cutoff.
    paths = [root / "predict.py", root / "model_metadata.json"]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise ValueError(
            "Trained model not connected yet (missing " + ", ".join(missing) + ")."
        )
    metadata = json.loads(paths[1].read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or not {"model_name", "trained_through", "feature_columns"} <= metadata.keys()
    ):
        raise ValueError(
            "Model metadata needs model_name, trained_through, and feature_columns."
        )
    # A replay can't use a model trained on shifts after the planning time.
    target = pd.Timestamp(target).normalize()
    try:
        trained = pd.Timestamp(date.fromisoformat(metadata["trained_through"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "trained_through must be a real shift start date (year, month, day)."
        ) from exc
    if (
        not isinstance(metadata["model_name"], str)
        or not metadata["model_name"].strip()
    ):
        raise ValueError("model_name must be a nonempty string.")
    accepts_missing = metadata.get("accepts_missing", False)
    if not isinstance(accepts_missing, bool):
        raise TypeError("accepts_missing must be a JSON boolean, not text.")
    if trained > target - pd.Timedelta(days=3):
        raise ValueError(
            "This model was trained using shifts unavailable at this planning time."
        )
    # Only the model's input columns; never calls or the same shift's category counts.
    columns = metadata["feature_columns"]
    forbidden = {"calls", *GROUPS, "Masked", "OutageHours", "ShiftDate"}
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(c, str) for c in columns)
    ):
        raise ValueError(
            "feature_columns must be a nonempty list of input column names."
        )
    if forbidden.intersection(columns) or set(columns) - set(features.columns):
        raise ValueError(
            "Model inputs contain target information or unavailable columns."
        )
    if len(set(columns)) != len(columns):
        raise ValueError("Model input column names must be unique.")
    # Select one modeling_dataset.csv input row per station for the requested target date.
    rows = features.loc[features["ShiftDate"] == target].copy()
    if rows.empty:
        # Shifts with a gap in the call data were dropped from modeling_dataset.csv.
        raise ValueError(
            "No estimate for this date: the call data for this shift has a gap."
        )
    if rows["Station"].duplicated().any() or set(rows["Station"]) != set(station_ids):
        raise ValueError(
            "modeling_dataset.csv needs one row per station for this shift."
        )
    # Blank history values are allowed only if the supplied model pipeline supports them.
    if rows[columns].isna().any().any() and not accepts_missing:
        raise ValueError(
            "Some model inputs are blank for this shift, and the model isn't set up for blanks."
        )
    # predict.py is the model team's code and can fail in any way, so show its error on screen.
    try:
        module = _load_predict_module(paths[0])
        output = module.predict_shift(rows[columns].copy())
    except (
        ImportError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        raise ValueError(
            f"Model prediction failed ({type(exc).__name__}): {exc}"
        ) from exc
    return validate_predictions(output, station_ids), str(metadata["model_name"])


def hourly_allocation(root, target, station, total):
    """Use hourly_profile.csv to divide one station's shift estimate into hourly amounts."""
    path = root / "data/hourly_profile.csv"
    if not path.exists():
        raise ValueError(
            "Hourly breakdown not available yet (needs data/hourly_profile.csv)."
        )
    # Check the real local shift length; daylight saving shifts need a different profile.
    start = (pd.Timestamp(target).normalize() + pd.Timedelta(hours=8)).tz_localize(
        "America/Los_Angeles"
    )
    end = (
        pd.Timestamp(target).normalize() + pd.Timedelta(days=1, hours=8)
    ).tz_localize("America/Los_Angeles")
    hours = (end - start).total_seconds() / 3600
    if hours != 24:
        raise ValueError(
            f"No hourly breakdown for this shift: the clocks change, so it is {hours:.0f} hours long."
        )
    # Each row gives one station, one shift hour, its share of calls, and its history cutoff.
    profile = pd.read_csv(path)
    expected = {"Station", "shift_hour", "share", "history_through"}
    if not expected <= set(profile):
        raise ValueError(
            "Hourly profile needs Station, shift_hour, share, history_through."
        )
    # Keep the selected station's 24 rows: shift hour 0 is 08:00 and 23 is 07:00 next day.
    part = profile.loc[profile["Station"] == station].copy()
    if len(part) != 24 or set(part["shift_hour"]) != set(range(24)):
        raise ValueError(
            "Hourly profile must have each shift_hour 0 to 23 exactly once per station."
        )
    # The shares must add up to 1, so allocating them preserves the full shift estimate.
    shares = pd.to_numeric(part["share"], errors="raise")
    if (
        not np.isfinite(shares).all()
        or (shares < 0).any()
        or not np.isclose(shares.sum(), 1)
    ):
        raise ValueError(
            "Hourly proportions must be finite, 0 or more, and add up to 1."
        )
    # Reject an hourly pattern built using calls that were not known at planning time.
    through = pd.to_datetime(part["history_through"], errors="raise")
    if (
        through.isna().any()
        or (through > pd.Timestamp(target) - pd.Timedelta(days=3)).any()
    ):
        raise ValueError(
            "The hourly profile uses history unavailable at the planning time."
        )
    # Example: a 5% share of a 30 call estimate gives 1.5 expected calls in that hour.
    part["expected_calls"] = shares * total
    part["hour"] = part["shift_hour"].map(lambda h: f"{(int(h) + 8) % 24:02d}:00")
    return part.sort_values("shift_hour")


def load_ranges(root=ROOT):
    """Read prediction_ranges.csv: how far real calls usually land from an estimate."""
    ranges = pd.read_csv(root / "data/prediction_ranges.csv")
    if {"level", "predicted_from", "predicted_to", "low", "high"} - set(ranges.columns):
        raise ValueError("prediction_ranges.csv is missing columns.")
    bands = ranges[ranges["level"] == "station"].sort_values("predicted_from")
    city = ranges[ranges["level"] == "citywide"]
    if set(ranges["level"]) - {"station", "citywide"} or len(city) != 1 or bands.empty:
        raise ValueError(
            "prediction_ranges.csv needs one citywide row and station bands."
        )
    # Every estimate must land in exactly one band, so the bands start at 0 with no gap.
    starts, ends = bands["predicted_from"].to_numpy(), bands["predicted_to"].to_numpy()
    if starts[0] != 0 or (starts[1:] != ends[:-1]).any():
        raise ValueError(
            "Station bands must start at 0 and follow each other with no gap."
        )
    if (
        ranges[["low", "high"]].isna().any().any()
        or (ranges["low"] > 0).any()
        or (ranges["high"] < 0).any()
    ):
        raise ValueError(
            "Every range needs a low of 0 or less and a high of 0 or more."
        )
    return bands.reset_index(drop=True), city.iloc[0]


def add_likely_range(view, bands):
    """Add whole-number low and high columns: each estimate plus the usual misses for its size."""
    band = np.searchsorted(
        bands["predicted_to"].to_numpy(), view["predicted_calls"], side="left"
    )
    band = band.clip(max=len(bands) - 1)
    view = view.copy()
    view["low"] = (
        (view["predicted_calls"] + bands["low"].to_numpy()[band])
        .clip(lower=0)
        .round()
        .astype(int)
    )
    view["high"] = (
        (view["predicted_calls"] + bands["high"].to_numpy()[band]).round().astype(int)
    )
    return view


def citywide_range(total, city):
    return round(max(total + city["low"], 0)), round(total + city["high"])
