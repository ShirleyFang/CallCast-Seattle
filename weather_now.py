"""Weather summary for the Prediction tab: one icon with the shift's low and high and any rain or
snow. It sits next to the estimate and never changes it.

get_shift_weather(target_date, now) returns None when there is no forecast for the shift, or a dict:
    condition  one of CONDITIONS, e.g. "rain"
    label      the words for the icon, also its alt text, e.g. "Rain"
    temp_min, temp_max (degC), rain_mm, snow_cm, source
It uses the forecast the weather alerts use (weather_alerts.forecast_source), so only weather known
at the planning time. Saved shifts take their weather codes from data/weather_codes_gfs_global_day3.csv
(scripts/make_weather_codes.py); later shifts use the alerts' download, which asks for the codes too.
Icons: optional local SVGs in static/weather_icons (git-ignored); without one the line shows
its words only. Each weather alert gets its own icon the same way (ALERT_ICONS, alert_icon_css), and the NWS
note the NWS logo (nws_note_html).
"""
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from weather_alerts import STORED_PATH, WIND_GUST, fetched_hourly, forecast_source  # noqa: E402
from weather_utils import to_shifts  # noqa: E402

CODES_PATH = "data/weather_codes_gfs_global_day3.csv"
ICON_DIR = ROOT / "static" / "weather_icons"
# Streamlit serves static/ at app/static/ (enableStaticServing in .streamlit/config.toml)
ICON_URL = "app/static/weather_icons/"
TZ = "America/Los_Angeles"

# Most notable first: a shift shows the first one that happens in any of its hours.
CONDITIONS = [
    "thunder", "snow", "sleet", "heavy_rain", "rain", "fog", "wind", "cloudy", "partly_cloudy", "sun",
]
LABELS = {
    "thunder": "Thunderstorms",
    "snow": "Snow",
    "sleet": "Sleet or freezing rain",
    "heavy_rain": "Heavy rain",
    "rain": "Rain",
    "fog": "Fog",
    "wind": "Strong wind",
    "cloudy": "Cloudy",
    "partly_cloudy": "Partly cloudy",
    "sun": "Sunny",
}
# WMO weather codes from Open-Meteo. 0 to 3 (clear to overcast) are left out: dry shifts take
# their icon from the wind and the cloud cover instead.
CODE_CONDITION = {
    **dict.fromkeys([45, 48], "fog"),
    **dict.fromkeys([51, 53, 55, 61, 63, 80, 81], "rain"),
    **dict.fromkeys([65, 82], "heavy_rain"),
    **dict.fromkeys([56, 57, 66, 67], "sleet"),
    **dict.fromkeys([71, 73, 75, 77, 85, 86], "snow"),
    **dict.fromkeys([95, 96, 99], "thunder"),
}
# The icon beside each weather alert, by its kind in weather_alerts.MESSAGES. The freeze and
# cold spell message shows the hard freeze snowflake.
ALERT_ICONS = {
    "wind": "alert_wind",
    "freeze": "alert_freeze",
    "freeze_spell": "alert_freeze",
    "spell": "alert_spell",
}
# The NWS logo beside the NWS note, a local file like the icons.
NWS_LOGO = "nws.png"
CLOUDY = 70  # mean cloud cover (%) at or above this is cloudy
PARTLY = 30  # at or above this is partly cloudy, below it sunny
SHIFT_VARIABLES = ["precipitation", "snowfall", "temperature_2m", "wind_gusts_10m"]
NEEDED = ["temp_min", "temp_max", "precip_sum", "snow_sum", "gust_max", "weather_code", "cloud_cover"]


# The most notable code of a shift's hours: one hour of thunder (95) beats a day of rain (61).
def notable_code(codes):
    def rank(code):
        return CONDITIONS.index(CODE_CONDITION[code]) if code in CODE_CONDITION else len(CONDITIONS)
    return min(codes.astype(int), key=lambda c: (rank(c), -c))


def code_shifts(hourly):
    """Hourly UTC weather_code and cloud_cover -> per 08:00-08:00 shift, the most notable code and
    the mean cloud cover. A shift counts only if every hour is there, as in to_shifts."""
    h = hourly.set_index("time")
    # the code describes the hour before its time stamp, like precipitation in to_shifts
    w = h[["weather_code"]].shift(-1, freq="h").join(h[["cloud_cover"]], how="outer")
    local = w.index.tz_localize("UTC").tz_convert(TZ).tz_localize(None)
    shift = pd.Series((local - pd.Timedelta(hours=8)).normalize(), index=w.index)
    full = w.notna().all(axis=1).groupby(shift).transform("all")
    w, shift = w[full], shift[full]
    out = pd.DataFrame({
        "weather_code": w["weather_code"].groupby(shift).agg(notable_code),
        "cloud_cover": w["cloud_cover"].groupby(shift).mean().round(1),
    })
    out.index.name = "ShiftDate"
    return out


# The icon for a shift: code 63 gives "rain"; a dry shift with gusts of 65 km/h gives "wind".
def shift_condition(code, cloud, gust):
    if code in CODE_CONDITION:
        return CODE_CONDITION[code]
    if gust >= WIND_GUST:
        return "wind"
    return "cloudy" if cloud >= CLOUDY else "partly_cloudy" if cloud >= PARTLY else "sun"


@st.cache_data(show_spinner=False)
def _read_saved(stored, codes, stamps):
    if not Path(stored).exists():
        return pd.DataFrame(columns=NEEDED)
    hourly = pd.read_csv(stored, parse_dates=["time"])
    hourly = hourly.rename(columns=lambda c: c.removesuffix("_previous_day3"))
    # the same shifts the alerts call "stored": every hour of temperature and gusts is there
    shifts = to_shifts(hourly[["time", *SHIFT_VARIABLES]]).dropna(subset=["temp_min", "gust_max"])
    if Path(codes).exists():
        return shifts.join(pd.read_csv(codes, parse_dates=["ShiftDate"], index_col="ShiftDate"))
    return shifts.assign(weather_code=np.nan, cloud_cover=np.nan)


def get_shift_weather(target_date, now, root=ROOT, fetch=None):
    """Weather summary for the shift starting at 08:00 on `target_date`; `now` is Seattle local time."""
    root = Path(root)
    day = pd.Timestamp(target_date).normalize()
    stored, codes = root / STORED_PATH, root / CODES_PATH
    try:
        # file times in the cache key, so a rebuilt CSV is read again
        stamps = tuple(p.stat().st_mtime if p.exists() else 0 for p in (stored, codes))
        saved = _read_saved(str(stored), str(codes), stamps)
        source = forecast_source(day, pd.Timestamp(now), saved.index)
        if source is None:
            return None
        if source == "stored":
            shifts = saved
        else:
            hourly = fetched_hourly(source, day, root, fetch)
            shifts = to_shifts(hourly[["time", *SHIFT_VARIABLES]]).join(code_shifts(hourly))
        row = shifts.loc[day, NEEDED]
        if row.isna().any():
            return None
    except (OSError, ValueError, KeyError, TypeError, requests.RequestException):
        return None
    condition = shift_condition(int(row["weather_code"]), row["cloud_cover"], row["gust_max"])
    return {
        "condition": condition,
        "label": LABELS[condition],
        "temp_min": float(row["temp_min"]),
        "temp_max": float(row["temp_max"]),
        "rain_mm": float(row["precip_sum"]),
        "snow_cm": float(row["snow_sum"]),
        "source": source,
    }


# The words beside the icon, e.g. "Rain · Low 4 °C, high 9 °C · 12.3 mm rain".
def weather_text(summary):
    parts = [summary["label"], f"Low {round(summary['temp_min'])} °C, high {round(summary['temp_max'])} °C"]
    # precipitation includes melted snow, so a snowy shift shows only its snow
    if round(summary["snow_cm"], 1) > 0:
        parts.append(f"{summary['snow_cm']:.1f} cm snow")
    elif round(summary["rain_mm"], 1) > 0:
        parts.append(f"{summary['rain_mm']:.1f} mm rain")
    return " · ".join(parts)


def weather_now_html(summary, icon_dir=None):
    """The icon and its line, or "" when there is no summary. Without the icon file (it is kept out
    of git) the line shows its words only, never a broken image."""
    if summary is None:
        return ""
    label, rest = escape(summary["label"]), escape(weather_text(summary).removeprefix(summary["label"]))
    icon = ""
    if (Path(icon_dir or ICON_DIR) / f"{summary['condition']}.svg").exists():
        icon = (f'<img src="{ICON_URL}{summary["condition"]}.svg" alt="{label}" width="30" height="30" '
                'style="flex:none;display:block;">')
    return (
        '<div style="display:flex;align-items:center;gap:10px;margin:0 0 2px 2px;'
        'color:var(--cc-ink);font-size:var(--fs-md);line-height:1.4;">'
        f'{icon}<span><strong style="font-weight:600;">{label}</strong>{rest}</span></div>'
    )


def alert_icon_css(kinds, icon_dir=None):
    """A <style> that puts each alert's icon at its left, sized like the weather line's icon, or ""
    when no icon file is there (it is kept out of git), so the alerts look as before."""
    rules = []
    for kind in dict.fromkeys(kinds):
        name = ALERT_ICONS.get(kind)
        if name is None or not (Path(icon_dir or ICON_DIR) / f"{name}.svg").exists():
            continue
        box = f'.st-key-weather_alert_{kind} [data-testid="stAlertContentWarning"]'
        rules.append(
            f"{box}{{display:flex;align-items:center;gap:10px;}}"
            f'{box}::before{{content:"";flex:none;width:30px;height:30px;'
            f'background:url("{ICON_URL}{name}.svg") center/contain no-repeat;}}'
        )
    return f"<style>{''.join(rules)}</style>" if rules else ""


# The NWS note with the logo (56 px, shown at 28) in place of its info mark, plus the <style>
# that hides the mark. Without the logo file (kept out of git) the note is as before.
def nws_note_html(note_html, icon_dir=None):
    if not (Path(icon_dir or ICON_DIR) / NWS_LOGO).exists():
        return note_html, ""
    img = (f'<img src="{ICON_URL}{NWS_LOGO}" alt="National Weather Service" width="28" height="28" '
           'style="flex:none;display:block;">')
    style = (".st-key-prediction_nws .cc-note{align-items:center;}"
             ".st-key-prediction_nws .cc-note::before{display:none;}")
    return note_html.replace("<span>", img + "<span>", 1), f"<style>{style}</style>"
