"""Pieces of the Prediction tab that are plain data work: the watch list, the map shading, the
call type mix, picking stations on the map and the freshness line."""

import pandas as pd

from station_areas import shade
from station_compare import CITY, GROUP_NAMES, MAX_COMPARE

# A station is "likely busier" when its estimate is at least this many calls and this share
# above its usual. Smaller gaps are inside the normal swing: the likely ranges are about 20
# calls wide, so a rule based on the range would never fire.
WATCH_MIN_EXTRA = 2
WATCH_MIN_SHARE = 0.10
WATCH_MAX = 5

# Quieter (blue) to busier (orange): colour-blind safe, and the legend names both ends.
DIVERGE_STOPS = ["#5B8DB8", "#B9D3E8", "#F4F1EA", "#F4C38D", "#C8641A"]
# Shading never stretches a gap smaller than this (calls) to the full colour range.
MIN_SPAN = WATCH_MIN_EXTRA

MIX_TYPES = ["medical", "fire", "non_emergency", "other"]
# A shift dated D ends at 08:00 on D+1, so 32 hours after midnight on D.
SHIFT_END_HOURS = 24 + 8


def stations_to_watch(view):
    above = view[
        (view["difference"] >= WATCH_MIN_EXTRA)
        & (view["difference"] >= WATCH_MIN_SHARE * view["usual_calls"])
    ]
    return above.sort_values("difference", ascending=False).head(WATCH_MAX)


def difference_fills(view):
    # Blue below usual, light at usual, orange above, using the same span on both sides.
    diff = view.set_index("Station")["difference"]
    span = max(diff.abs().max(), MIN_SPAN)
    return shade(diff, -span, span, DIVERGE_STOPS)


def season_mix(history, target):
    # Average past calls per shift in the target's month (earlier years only, already stopped
    # at the planning time with outage shifts dropped).
    target = pd.Timestamp(target)
    season = history[
        (history["ShiftDate"].dt.month == target.month)
        & (history["ShiftDate"].dt.year < target.year)
    ]
    mix = season.groupby("Station")[MIX_TYPES].mean()
    mix.loc[CITY] = season.groupby("ShiftDate")[MIX_TYPES].sum().mean()
    return mix.rename(columns=GROUP_NAMES)


def mix_table(mix, names):
    return mix.loc[names].T.round(1)


def toggle_picks(picked, clicked, limit=MAX_COMPARE):
    """Apply map clicks to the picked stations: a click picks, a second click unpicks.

    Returns the new list and the stations that were refused because `limit` are picked.
    """
    picked, refused = list(picked), []
    for name in clicked:
        if name in picked:
            picked.remove(name)
        elif len(picked) < limit:
            picked.append(name)
        else:
            refused.append(name)
    return picked, refused


def data_through(history):
    end = history["ShiftDate"].max() + pd.Timedelta(hours=SHIFT_END_HOURS)
    return f"{end:%a %b %d %H:%M}"
