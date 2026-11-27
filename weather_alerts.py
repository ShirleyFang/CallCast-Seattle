"""Weather alerts for the Prediction tab: strong wind and hard cold. They never change the estimate.

get_weather_alerts(target_date, now) returns a dict:
    status    "alert", "clear", "not_yet" (too far ahead) or "unavailable" (forecast could not be read)
    messages  plain-English lines to show
    kinds     for an alert, the MESSAGES key of each line, e.g. ["wind", "freeze_spell"], for its icon
    source    where the forecast came from ("stored", "previous_runs" or "live"), for tests

Which forecast is used, so the alert matches what was known at the planning time
(08:00 two days before the shift, e.g. Thursday 08:00 for Saturday):
    stored         shifts covered by data/weather_hourly_gfs_global_day3.csv (step8_weather.py)
    previous_runs  later shifts whose planning time has passed: the archived GFS day-3 forecast
    live           planning time still ahead and the shift within ALERT_DAYS: the latest forecast
weather_now.py picks its forecast the same way (forecast_source) and shares the same download.
Under the alerts, show_nws(target_date, now) says when to point to the official NWS watches and
warnings (a link only, no NWS API): for every shift not yet over and within ALERT_DAYS, alert or not,
since our forecast has no snow or ice.
Thresholds and their check: README Step 15 (a) and (e), scripts/check_alert_thresholds.py.
With CALLCAST_OFFLINE=1 (set by the tests) nothing is downloaded: those shifts show "unavailable".
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from weather_utils import events, to_shifts  # noqa: E402

# forecast thresholds (Step 15): ERA5 threshold plus the GFS bias
WIND_GUST = 65.0  # km/h, shift gust_max >=
HARD_FREEZE = -3.0  # degC, shift temp_min <=
SPELL_TEMP = -1.0  # degC, temp_min <= on at least SPELL_SHIFTS consecutive shifts
SPELL_SHIFTS = 2
ALERT_DAYS = 4  # alerts only for shifts ending within 4 days, the lead times checked in Steps 9 and 15
PLANNING_DAYS = 2  # planning time: 08:00 two days before the shift

STORED_PATH = "data/weather_hourly_gfs_global_day3.csv"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLES = ["temperature_2m", "wind_gusts_10m"]
# One download serves the alerts and the weather summary (weather_now.py), so it asks for both.
FETCH_VARIABLES = [*VARIABLES, "precipitation", "snowfall", "weather_code", "cloud_cover"]
# Where each downloaded forecast comes from: (url, model, column suffix).
SOURCES = {
    "previous_runs": (PREVIOUS_RUNS_URL, "gfs_global", "_previous_day3"),
    "live": (FORECAST_URL, "gfs_seamless", ""),
}
# seconds to connect and to read, so a slow network never holds up the page
TIMEOUT = (3, 5)
# after a failed download, show "unavailable" for 5 minutes before trying again
RETRY_SECONDS = 300

MESSAGES = {
    "wind": "Strong wind is forecast for this shift. Windstorms usually bring more calls than usual.",
    "freeze": "Very cold weather is forecast for this shift, well below freezing. "
    "Hard freezes usually bring more calls than usual.",
    "spell": "A cold spell is forecast, below freezing for two or more days in a row. "
    "Cold spells usually bring more calls than usual.",
    "freeze_spell": "Very cold weather is forecast for this shift, as part of a cold spell of two or "
    "more days. Cold spells usually bring more calls than usual.",
    "clear": "No weather alerts for this shift.",
    "not_yet": "No weather forecast available yet. Weather alerts appear about 4 days before the shift.",
    "unavailable": "Weather forecast unavailable right now.",
}
NOTE = (
    "Weather alerts are informational and are not included in the forecast. "
    "Forecasts a few days ahead often change."
)

NWS_URL = "https://www.weather.gov/sew/"
NWS_NOTE = (
    "For snow, ice and the latest official watches and warnings, check the National Weather Service: "
    f"[NWS Seattle]({NWS_URL})."
)


@st.cache_data(ttl=3600, show_spinner=False)
def _download(url, params):
    """Hourly forecast JSON from Open-Meteo; cached for an hour."""
    response = requests.get(url, params=dict(params), timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


# (url, params) -> time of the last failed download; failures are not kept by the cache above
_failed = {}


def _download_or_skip(url, params):
    # every rerun (a map click, a table tick) would wait for the timeout again without this pause
    if time.monotonic() - _failed.get((url, params), -RETRY_SECONDS) < RETRY_SECONDS:
        raise OSError("forecast download failed a moment ago")
    try:
        return _download(url, params)
    except requests.RequestException:
        _failed[(url, params)] = time.monotonic()
        raise


def _location(root):
    # one point for the whole city, the mean of the station locations (as in step8_weather.py)
    stations = pd.read_csv(root / "data/sfd_stations.csv")
    return round(stations["lat"].mean(), 4), round(stations["lon"].mean(), 4)


def _stored_shifts(root):
    path = root / STORED_PATH
    if not path.exists():
        return pd.DataFrame(columns=["temp_min", "gust_max"])
    hourly = pd.read_csv(path, parse_dates=["time"])
    hourly = hourly.rename(columns=lambda c: c.removesuffix("_previous_day3"))
    return to_shifts(hourly[["time", *VARIABLES]]).dropna(subset=["temp_min", "gust_max"])


# Which forecast was known at the planning time for the shift on `day`: "stored",
# "previous_runs", "live", or None when the shift is too far ahead.
def forecast_source(day, now, stored_days):
    planning = day - pd.Timedelta(days=PLANNING_DAYS) + pd.Timedelta(hours=8)
    shift_end = day + pd.Timedelta(days=1, hours=8)
    if day in stored_days:
        return "stored"
    if planning <= now:
        return "previous_runs"
    if shift_end <= now + pd.Timedelta(days=ALERT_DAYS):
        return "live"


def fetched_hourly(source, day, root=ROOT, fetch=None):
    """Hourly weather (UTC) from the API for day-2 to day+3, suffix removed from the columns."""
    url, model, suffix = SOURCES[source]
    lat, lon = _location(Path(root))
    params = (
        ("latitude", lat),
        ("longitude", lon),
        ("hourly", ",".join(v + suffix for v in FETCH_VARIABLES)),
        ("models", model),
        ("timezone", "GMT"),
        # a day on each side so the 08:00-08:00 shifts are complete in UTC
        ("start_date", str((day - pd.Timedelta(days=2)).date())),
        ("end_date", str((day + pd.Timedelta(days=3)).date())),
    )
    if fetch is None and os.environ.get("CALLCAST_OFFLINE"):
        raise OSError("offline")
    body = (fetch or _download_or_skip)(url, params)
    hourly = pd.DataFrame(body["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    return hourly.rename(columns=lambda c: c.removesuffix(suffix))


def _cold_spell(temp_min, day):
    """True if `day` is in a run of at least SPELL_SHIFTS shifts with temp_min <= SPELL_TEMP."""
    cold = temp_min.dropna() <= SPELL_TEMP
    if not cold.get(day, False):
        return False
    run = events(cold.index[cold])
    return (run == run[day]).sum() >= SPELL_SHIFTS


def get_weather_alerts(target_date, now, root=ROOT, fetch=None):
    """Weather alerts for the shift starting at 08:00 on `target_date`; `now` is Seattle local time."""
    root = Path(root)
    day = pd.Timestamp(target_date).normalize()
    now = pd.Timestamp(now)

    try:
        stored = _stored_shifts(root)
        source = forecast_source(day, now, stored.index)
        if source is None:
            return {"status": "not_yet", "messages": [MESSAGES["not_yet"]], "source": None}
        if source == "stored":
            shifts = stored
        else:
            shifts = to_shifts(fetched_hourly(source, day, root, fetch)[["time", *VARIABLES]])
        today = shifts.loc[day, ["temp_min", "gust_max"]]
        if today.isna().any():
            raise ValueError("incomplete forecast for this shift")
    except (OSError, ValueError, KeyError, TypeError, requests.RequestException):
        return {"status": "unavailable", "messages": [MESSAGES["unavailable"]], "source": None}

    kinds = []
    if today["gust_max"] >= WIND_GUST:
        kinds.append("wind")
    freeze = today["temp_min"] <= HARD_FREEZE
    spell = _cold_spell(shifts["temp_min"], day)
    if freeze or spell:
        kinds.append("freeze_spell" if freeze and spell else "freeze" if freeze else "spell")
    if not kinds:
        return {"status": "clear", "messages": [MESSAGES["clear"]], "source": source}
    return {"status": "alert", "messages": [MESSAGES[k] for k in kinds], "kinds": kinds, "source": source}


def show_nws(target_date, now):
    """True if the shift is not over yet and ends within ALERT_DAYS of `now` (Seattle local time)."""
    shift_end = pd.Timestamp(target_date).normalize() + pd.Timedelta(days=1, hours=8)
    now = pd.Timestamp(now)
    return now < shift_end <= now + pd.Timedelta(days=ALERT_DAYS)
