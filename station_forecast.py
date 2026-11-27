"""Pieces of the Prediction tab that are plain data work: the briefing, range bars, watch list,
map shading, call type mix, the ranked table and download, picking stations on the map or in
the table, what a table click opens, the history window's summary, and the freshness line."""

import pandas as pd

from app_data import GAP_MESSAGE
from chart_theme import DIVERGE_STOPS
from station_areas import shade
from station_compare import CITY, GROUP_NAMES, MAX_COMPARE

# A station is "likely busier" when its estimate is at least this many calls and this share
# above its usual. Smaller gaps are inside the normal swing: the likely ranges are 5 to 19
# calls wide, so a rule based on the range would never fire.
WATCH_MIN_EXTRA = 2
WATCH_MIN_SHARE = 0.10
WATCH_MAX = 5

# Within 5 calls (~1.7% of a ~300-call shift) is inside normal day-to-day noise.
BRIEF_USUAL_CALLS = 5
# 20+ extra calls (~6% above usual) is a clear citywide bump worth calling out.
BRIEF_BUSIER_CALLS = 20
# Name at most 3 stations so the top briefing stays a single readable sentence.
BRIEF_TOP_STATIONS = 3
# Within 0.5 calls of usual reads as "about usual" on a single station's bar and hover card.
STATION_USUAL_CALLS = 0.5

# The map scale, quieter (blue) to busier (salmon), is DIVERGE_STOPS in chart_theme.py.
# Shading never stretches a gap smaller than this (calls) to the full colour range.
MIN_SPAN = WATCH_MIN_EXTRA

MIX_TYPES = ["medical", "fire", "non_emergency", "other"]
# A shift dated D ends at 08:00 on D+1, so 32 hours after midnight on D.
SHIFT_END_HOURS = 24 + 8
# The history window lists each station's last 4 finished shifts on the target's weekday.
SAME_WEEKDAYS = 4
# The table's Last 4 weeks bars: one per shift, 4 weeks of 7 days.
RECENT_DAYS = 28
# What a click in the table can open: Last 4 weeks opens history, Expected the hourly window.
WINDOWS = ("history", "hourly")
# The history window's time spans, in days back from the last known shift (None = all).
HISTORY_SPANS = {"4 weeks": 28, "3 months": 91, "All": None}

# Plain-word column names for the download.
COLUMN_NAMES = {
    "predicted_calls": "Expected calls",
    "usual_calls": "Usual for this weekday",
    "difference": "Difference",
    "low": "Low",
    "high": "High",
    "ShiftDate": "Shift date",
    "AsOf": "Planning time",
}
# Column order for the download.
TABLE_COLUMNS = [
    "Station",
    "predicted_calls",
    "low",
    "high",
    "usual_calls",
    "difference",
]


def _join_names(names):
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def station_status_words(diff):
    # Qualitative phrase for one station's hover card (never percentages).
    if diff <= -WATCH_MIN_EXTRA:
        return "Quieter than usual"
    if diff < -STATION_USUAL_CALLS:
        return "A little quieter than usual"
    if diff <= STATION_USUAL_CALLS:
        return "About usual"
    if diff < WATCH_MIN_EXTRA:
        return "A little busier than usual"
    return "Busier than usual"


def shift_briefing(target, view=None, reason=None):
    # One plain sentence summarizing the target shift for a supervisor, with no percentages.
    day = f"{pd.Timestamp(target):%A}"
    if view is None:
        if reason == GAP_MESSAGE:
            return f"{day} has no call estimate because the call data for this shift has a gap."
        return f"{day} has no call estimate available yet."
    diff = float(view["difference"].sum())
    if diff < -BRIEF_USUAL_CALLS:
        quieter = view[
            (view["difference"] <= -WATCH_MIN_EXTRA)
            & (-view["difference"] >= WATCH_MIN_SHARE * view["usual_calls"])
        ].sort_values("difference")
        drivers = quieter.head(BRIEF_TOP_STATIONS)["Station"].tolist()
        where = f", led by {_join_names(drivers)}" if drivers else ", spread across the city"
        watch = stations_to_watch(view).head(BRIEF_TOP_STATIONS)["Station"].tolist()
        extra = f", though {_join_names(watch)} may see more calls" if watch else ""
        return f"{day} looks quieter than usual{where}{extra}."
    if diff <= BRIEF_USUAL_CALLS:
        watch = stations_to_watch(view)
        if watch.empty:
            return f"{day} looks about usual across Seattle."
        drivers = watch.head(BRIEF_TOP_STATIONS)["Station"].tolist()
        return f"{day} looks about usual overall, though {_join_names(drivers)} may see more calls."
    label = "a little busier than usual" if diff <= BRIEF_BUSIER_CALLS else "busier than usual"
    drivers = stations_to_watch(view).head(BRIEF_TOP_STATIONS)["Station"].tolist()
    where = f", led by {_join_names(drivers)}" if drivers else ", spread across the city"
    return f"{day} looks {label}{where}."


def biggest_move(view):
    # Pick the station driving the shift's direction so card 2 matches the briefing.
    if float(view["difference"].sum()) < -BRIEF_USUAL_CALLS:
        return "Biggest drop vs usual", view.sort_values("difference").iloc[0]
    top = view.sort_values("difference", ascending=False).iloc[0]
    if top["difference"] <= 0:
        return "Biggest drop vs usual", view.sort_values("difference").iloc[0]
    return "Biggest rise vs usual", top


def range_bar_spec(expected, low, high, usual, threshold=STATION_USUAL_CALLS):
    # Scale the range bar around min/max of range, expected and usual so nothing clips.
    expected = float(expected)
    usual = float(usual)
    lo_val = min(float(low), expected, usual)
    hi_val = max(float(high), expected, usual)
    pad = max((hi_val - lo_val) * 0.12, 1.0)
    axis_min = max(0.0, lo_val - pad) if lo_val > 0 else 0.0
    axis_max = hi_val + pad
    scale = max(axis_max - axis_min, 1.0)
    band_left = (float(low) - axis_min) / scale * 100
    band_right = (float(high) - axis_min) / scale * 100
    diff = expected - usual
    tone = "busier" if diff > threshold else ("quieter" if diff < -threshold else "usual")
    return {
        "low": round(low),
        "high": round(high),
        "expected": expected,
        "usual": usual,
        "diff": diff,
        "band_left": band_left,
        "band_width": max(band_right - band_left, 2.0),
        "expected_pct": (expected - axis_min) / scale * 100,
        "usual_pct": (usual - axis_min) / scale * 100,
        "tone": tone,
    }


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


def same_weekday_calls(history, target, n=SAME_WEEKDAYS):
    # Each station's last n finished shifts on the target's weekday, oldest first (outage
    # shifts are already dropped, so a missing week is skipped, not counted as 0).
    day = pd.Timestamp(target).dayofweek
    same = history[history["ShiftDate"].dt.dayofweek == day]
    last = same.sort_values("ShiftDate").groupby("Station").tail(n)
    return last.groupby("Station")["calls"].apply(lambda c: [int(x) for x in c])


def recent_days(target, n=RECENT_DAYS):
    # The last n shifts finished by the planning time, oldest first. A shift dated D ends 08:00
    # on D+1 and planning is 08:00 two days before the target, so the newest is the target
    # minus 3 days: Sat Oct 24 (planned Thu Oct 22) gives Sep 24 to Oct 21.
    end = pd.Timestamp(target).normalize() - pd.Timedelta(days=3)
    return list(pd.date_range(end=end, periods=n))


def last_weeks(history, dates, last, estimates, stations):
    # Calls per station on each date. A date after the last day of data (`last`) takes the
    # app's estimate for it (`estimates[date]`); a date with a data gap stays empty (None).
    known = history.pivot(index="Station", columns="ShiftDate", values="calls")
    weeks = pd.DataFrame(index=list(stations))
    for d in dates:
        weeks[d] = estimates[d].round(1) if d > pd.Timestamp(last) else known.get(d)
    return weeks.astype(object).where(weeks.notna(), None)


def ranked_rows(view, weeks):
    # Busiest station first, with its 28 daily values for the Last 4 weeks bars.
    rows = view.sort_values(
        "predicted_calls", ascending=False, kind="stable"
    ).reset_index(drop=True)
    return pd.DataFrame(
        {
            "Station": rows["Station"],
            "Expected calls": rows["predicted_calls"],
            "Likely range": [
                f"{lo} to {hi}" for lo, hi in zip(rows["low"], rows["high"])
            ],
            "Difference": rows["difference"],
            "Last 4 weeks": [
                list(weeks.loc[s]) if s in weeks.index else [None] * len(weeks.columns)
                for s in rows["Station"]
            ],
        }
    )


def table_action(action, picked, stations):
    # One click sent by the table, e.g. {"act": "history", "station": "SFD 10"}. A tick picks or
    # unpicks that row's station like a map click; history and hourly name the window to open.
    # Returns the picks, the refused stations and the window, e.g. ("history", "SFD 10").
    name = (action or {}).get("station")
    if name not in stations:
        return list(picked), [], None
    if action.get("act") == "tick":
        picked, refused = toggle_picks(picked, [name])
        return picked, refused, None
    window = (action["act"], name) if action.get("act") in WINDOWS else None
    return list(picked), [], window


def history_summary(target, recent, row, through=None):
    # Plain lines for the top of the history window, e.g. "Last 4 Saturdays: 28, 31, 25, 30."
    # and "Expected 40.0 on Sat Oct 24 (likely 31 to 50) ...". A third line says how old the
    # newest known shift is when it is more than a week before this one.
    target = pd.Timestamp(target)
    day = f"{target:%A}s"
    if recent:
        first = f"Last {len(recent)} {day}: {', '.join(map(str, recent))}."
        if len(recent) == 1:
            first = f"Last {target:%A}: {recent[0]}."
    else:
        first = f"No finished {day} are known yet."
    second = (
        f"Expected {row['predicted_calls']:.1f} on {target:%a %b %d} "
        f"(likely {int(row['low'])} to {int(row['high'])}); "
        f"usual for a {target:%A} {row['usual_calls']:.1f}."
    )
    if through is None or (target - pd.Timestamp(through)).days <= 7:
        return [first, second]
    through = pd.Timestamp(through)
    days = (target - through).days
    return [first, second, f"The newest known shift is {through:%a %b %d}, {days} days earlier."]


def download_rows(view, target, origin):
    # The counts as shown, the likely range, the shift date and the planning time.
    return (
        view[TABLE_COLUMNS]
        .round(1)
        .assign(ShiftDate=str(target), AsOf=origin.isoformat())
        .rename(columns=COLUMN_NAMES)
    )


def data_through(history):
    end = history["ShiftDate"].max() + pd.Timedelta(hours=SHIFT_END_HOURS)
    return f"{end:%a %b %d %H:%M}"
