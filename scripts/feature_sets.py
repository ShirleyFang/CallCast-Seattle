"""Feature sets for the evaluation scripts: the fixed baseline and the candidate features.

BASELINE_NUMERIC   numeric inputs of the team's baseline model (the first-pass GBM of
                   Step 6, the same features as modeling_dataset.csv), listed explicitly
                   so that columns added to modeling_dataset.csv later do not change it.
                   With baseline_columns(X) the model also gets the station and weekday
                   dummies.
add_special_days   New Year's Eve, the day after Thanksgiving, Dec 24-26: the candidates
                   with the clearest evidence (Step 12), proposed for the final model.
add_*_candidates   add candidate features to a copy of the modeling table, in memory
                   (nothing is saved), and return (df, groups): groups maps a group
                   name to its columns. Every candidate is known before the forecast
                   shift (calendar, schedules, or calls at least MIN_LAG shifts back).
add_candidates     all candidate groups at once, special days included.

modeling_dataset.csv (step4_features.py) holds the baseline features only; which
candidates enter the final model is decided in the model evaluation.

Import from other scripts in this folder, for example:
    from feature_sets import BASELINE_NUMERIC, baseline_columns, add_calendar_candidates
"""
import numpy as np
import pandas as pd

from cv_utils import DATE_COL, MIN_LAG, STATION_COL, TARGET

BASELINE_NUMERIC = ["month", "is_holiday", "is_holiday_eve",
                    "lag3", "lag7", "avg7", "avg28", "same_weekday_avg4"]
SPECIAL_DAYS = ["is_nye", "is_day_after_thanksgiving", "is_xmas"]


def baseline_columns(X):
    """Columns of X (from cv_utils.build_features) that make up the baseline model."""
    missing = [c for c in BASELINE_NUMERIC if c not in X.columns]
    if missing:
        raise ValueError(f"Baseline columns missing from X: {missing}")
    return BASELINE_NUMERIC + [c for c in X.columns if c.startswith(("st_", "wd_"))]


def _stations():
    return pd.read_csv("data/sfd_stations.csv").set_index("station")


# ---------------------------------------------------------------- special days (Step 12)

def add_special_days(df):
    """Special days that is_holiday / is_holiday_eve miss (found in the error analysis):
    is_nye                     Dec 31: higher every year; is_holiday_eve mixes it with quieter eves
    is_day_after_thanksgiving  Black Friday: lower every year; not a federal holiday, so not flagged
    is_xmas                    Dec 24-26: is_holiday flags only Dec 25
    """
    df = df.copy()
    day = df[DATE_COL]
    df["is_nye"] = (
        day.dt.month.eq(12) & day.dt.day.eq(31)
    ).astype(int)

    # Black Friday: Friday after the fourth Thursday of November.
    # Calculate directly from the target date so single-day predictions work.
    df["is_day_after_thanksgiving"] = (
        day.dt.month.eq(11)
        & day.dt.dayofweek.eq(4)
        & day.dt.day.between(23, 29)
    ).astype(int)

    df["is_xmas"] = (
        day.dt.month.eq(12) & day.dt.day.between(24, 26)
    ).astype(int)

    return df, {"special days": SPECIAL_DAYS}

# ---------------------------------------------------------------- calendar (Step 12)

# UW quarters, from washington.edu/students/reg (yearly calendars and the 2024-2033 summary)
UW_QUARTERS = [
    ("2022-01-03", "2022-03-18"), ("2022-03-28", "2022-06-10"), ("2022-09-28", "2022-12-16"),
    ("2023-01-03", "2023-03-17"), ("2023-03-27", "2023-06-09"), ("2023-09-27", "2023-12-15"),
    ("2024-01-03", "2024-03-15"), ("2024-03-25", "2024-06-07"), ("2024-09-25", "2024-12-13"),
    ("2025-01-06", "2025-03-21"), ("2025-03-31", "2025-06-13"), ("2025-09-24", "2025-12-12"),
    ("2026-01-05", "2026-03-20"), ("2026-03-30", "2026-06-12"), ("2026-09-30", "2026-12-18"),
]


def add_calendar_candidates(df):
    """a. is_month_start   days 1-3 of the month (benefits and pay arrive; Ray 2016)
    b. uw_in_session    UW autumn, winter or spring quarter, first class day to last exam day
                        (summer quarter is small and left out)
    c. long_weekend     a weekend or holiday that is part of 3 or more days off in a row
    d. day_length       hours of daylight in Seattle (smoother than month)
    e. sps_break        Seattle Public Schools mid-winter break and spring break, Mon-Fri
                        (summer and winter break are left out: month and the holiday flags
                        already cover them)
    """
    df = df.copy()
    day = df[DATE_COL]

    df["is_month_start"] = day.dt.day.le(3).astype(int)

    in_session = pd.Series(False, index=df.index)
    for first, last in UW_QUARTERS:
        in_session |= day.between(first, last)
    df["uw_in_session"] = in_session.astype(int)

    # days off: Saturday, Sunday or a holiday (is_holiday from step4_features.py); one row
    # per day with a week of margin, so runs at the edges of the data are counted in full
    calendar = pd.DataFrame({"date": pd.date_range(day.min() - pd.Timedelta(days=7), day.max() + pd.Timedelta(days=7))})
    holidays = set(df.loc[df["is_holiday"] == 1, DATE_COL])
    calendar["off"] = calendar["date"].dt.dayofweek.ge(5) | calendar["date"].isin(holidays)
    run_id = calendar["off"].ne(calendar["off"].shift()).cumsum()          # number each run of days off / work days
    run_length = calendar.groupby(run_id)["off"].transform("size")
    long_weekend_days = set(calendar.loc[calendar["off"] & run_length.ge(3), "date"])
    df["long_weekend"] = day.isin(long_weekend_days).astype(int)

    # daylight hours from the sun's declination at Seattle's latitude (47.6 N)
    declination = np.radians(23.44) * np.sin(2 * np.pi * (284 + day.dt.dayofyear) / 365)
    df["day_length"] = 24 / np.pi * np.arccos(-np.tan(np.radians(47.6)) * np.tan(declination))

    # SPS breaks follow fixed rules in the district calendars (checked against the approved
    # 2021-22, 2022-23, 2024-25 and 2025-26 calendars on seattleschools.org):
    #   mid-winter break = the week of Presidents Day (3rd Monday of February)
    #   spring break     = the week starting the 2nd Monday of April
    sps_days = set()
    for year in range(day.dt.year.min(), day.dt.year.max() + 1):
        february_mondays = pd.date_range(f"{year}-02-01", f"{year}-02-28", freq="W-MON")
        april_mondays = pd.date_range(f"{year}-04-01", f"{year}-04-30", freq="W-MON")
        sps_days |= set(pd.date_range(february_mondays[2], periods=5)) | set(pd.date_range(april_mondays[1], periods=5))
    df["sps_break"] = day.isin(sps_days).astype(int)

    groups = {
        "month start": ["is_month_start"],
        "UW in session": ["uw_in_session"],
        "long weekend": ["long_weekend"],
        "day length": ["day_length"],
        "SPS break": ["sps_break"],
    }
    return df, groups


# ---------------------------------------------------------------- history (Step 11)

def add_lag_candidates(df):
    """Extra history features from our own call data, each from shifts at least MIN_LAG back
    (masked outage shifts count as missing, like the existing lags):
    a. citywide        citywide_ratio and citywide_lag3_ratio
    b. neighbor_lag3   mean lag3 of the 3 nearest stations
    c. trend           avg7 / avg28 (are calls going up or down lately?)
    d. std28           standard deviation of the 28 shifts before lag3
    e. type lags       medical_lag3, fire_lag3, non_emergency_lag3
    f. lag364          calls 364 shifts earlier (same weekday, one year before)
    """
    counts = pd.read_csv("data/station_shift_counts.csv", parse_dates=[DATE_COL])
    counts["Masked"] = counts["Masked"].astype(str).str.lower().isin(["true", "1"])
    stations = _stations()

    def wide(column):
        """One row per shift, one column per station; outage shifts count as unknown (like step4_features.py)."""
        known = counts[column].where(~counts["Masked"])
        return counts.assign(known=known).pivot(index=DATE_COL, columns=STATION_COL, values="known").sort_index()

    def window_mean(table, first, length, min_count):
        """Mean over the `length` shifts starting `first` shifts back (per column)."""
        return table.shift(first).rolling(length, min_periods=min_count).mean()

    calls_wide = wide(TARGET)
    new = {}   # feature name -> wide table (shift x station)

    # a. citywide: the mean calls per station over all stations with known calls, so a station
    #    in outage does not pull the total down; ratios make it fair for big and small stations
    city = calls_wide.mean(axis=1).to_frame("city")
    city_avg28 = window_mean(city, MIN_LAG, 28, 14)["city"]
    city_ratio = window_mean(city, MIN_LAG, 7, 4)["city"] / city_avg28     # citywide avg7 / avg28
    city_lag3_ratio = city["city"].shift(MIN_LAG) / city_avg28            # citywide lag3 / avg28
    new["citywide_ratio"] = pd.DataFrame({s: city_ratio for s in calls_wide.columns})
    new["citywide_lag3_ratio"] = pd.DataFrame({s: city_lag3_ratio for s in calls_wide.columns})

    # b. neighbor_lag3: mean lag3 of the 3 nearest stations (distance on a flat map, scaled for latitude)
    lag3_wide = calls_wide.shift(MIN_LAG)
    neighbors = {}
    for s in calls_wide.columns:
        dist = np.hypot(stations["lat"] - stations.loc[s, "lat"],
                        (stations["lon"] - stations.loc[s, "lon"]) * np.cos(np.radians(47.6)))
        neighbors[s] = [n for n in dist.drop(s).nsmallest(3).index if n in calls_wide.columns]
    new["neighbor_lag3"] = pd.DataFrame({s: lag3_wide[neighbors[s]].mean(axis=1) for s in calls_wide.columns})

    # d. std28: spread of the 28 shifts that avg28 uses
    new["std28"] = calls_wide.shift(MIN_LAG).rolling(28, min_periods=14).std()

    # e. type lags: calls of each type 3 shifts earlier
    for kind in ["medical", "fire", "non_emergency"]:
        new[f"{kind}_lag3"] = wide(kind).shift(MIN_LAG)

    # f. lag364: same weekday, one year earlier
    new["lag364"] = calls_wide.shift(364)

    # wide tables -> long columns, joined to the modeling rows by station and shift
    for name, table in new.items():
        long = table.stack().rename(name).reset_index()
        long.columns = [DATE_COL, STATION_COL, name]
        df = df.merge(long, on=[DATE_COL, STATION_COL], how="left")
    # c. trend, from the existing columns
    df["trend"] = df["avg7"] / df["avg28"]

    groups = {
        "citywide": ["citywide_ratio", "citywide_lag3_ratio"],
        "neighbor": ["neighbor_lag3"],
        "trend": ["trend"],
        "std28": ["std28"],
        "type lags": ["medical_lag3", "fire_lag3", "non_emergency_lag3"],
        "lag364": ["lag364"],
    }
    return df, groups


# ---------------------------------------------------------------- events (Steps 13-14)

VENUES = pd.DataFrame({"venue": ["Lumen Field", "T-Mobile Park", "Climate Pledge Arena", "Husky Stadium"],
                       "lat": [47.5952, 47.5914, 47.6221, 47.6503],
                       "lon": [-122.3316, -122.3325, -122.3540, -122.3016]}).set_index("venue")
CHOOSE_END = pd.Timestamp("2025-01-01")     # event-station pairs are chosen on 2022-2024 only


def load_events():
    """One table of events from data/events.csv (home games, scripts/fetch_events.py) and
    data/concerts.csv (scripts/fetch_concerts.py): kind (team, or stadium / arena concert),
    venue and ShiftDate."""
    games = pd.read_csv("data/events.csv", parse_dates=["ShiftDate"])
    concerts = pd.read_csv("data/concerts.csv", parse_dates=["ShiftDate"])
    concerts["team"] = concerts["size"] + " concert"
    events = pd.concat([games[["team", "venue", "ShiftDate"]], concerts[["team", "venue", "ShiftDate"]]])
    return events.rename(columns={"team": "kind"}).drop_duplicates()


def event_days(events, kind, venue):
    return set(events.loc[(events["kind"] == kind) & (events["venue"] == venue), "ShiftDate"])


def nearest_stations(venue, k=2):
    """The k stations closest to a venue (flat-map distance, scaled for latitude)."""
    stations = _stations()
    lat, lon = VENUES.loc[venue, ["lat", "lon"]]
    dist = np.hypot(stations["lat"] - lat, (stations["lon"] - lon) * np.cos(np.radians(lat)))
    return list(dist.nsmallest(k).index)


def calls_by_station(df):
    """One row per day, one column per station (missing shifts are NaN)."""
    return df.pivot(index=DATE_COL, columns=STATION_COL, values=TARGET).asfreq("D")


def event_effect(calls, station, days, before):
    """Mean extra calls on event shifts vs same-weekday shifts within +-4 weeks without that
    kind of event, using shifts before `before`. Returns (events, usual calls, extra calls,
    extra %, t)."""
    s = calls[station][calls.index < before].dropna()
    days = set(days)
    diffs, usual = [], []
    for d0 in sorted(days & set(s.index)):
        around = [d0 + pd.Timedelta(weeks=k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        around = [d for d in around if d in s.index and d not in days]
        if len(around) >= 3:
            diffs.append(s[d0] - s[around].mean())
            usual.append(s[around].mean())
    diffs = np.array(diffs)
    if len(diffs) < 2:
        return len(diffs), np.nan, np.nan, np.nan, np.nan
    t = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))
    return len(diffs), np.mean(usual), diffs.mean(), 100 * diffs.mean() / np.mean(usual), t


def choose_event_pairs(df, events):
    """Effect of each event kind at the 2 stations nearest its venue, on 2022-2024 only.
    A pair is kept if the effect is clear: at least 10 events, +10% or more, t >= 2.
    Returns (effects table, list of (kind, venue, station))."""
    calls = calls_by_station(df)
    rows = []
    for (kind, venue), group in events.groupby(["kind", "venue"]):
        for station in nearest_stations(venue):
            n, usual, extra, pct, t = event_effect(calls, station, group["ShiftDate"], CHOOSE_END)
            rows.append({"kind": kind, "venue": venue, "station": station, "events": n,
                         "usual_calls": usual, "extra_calls": extra, "extra_pct": pct, "t": t})
    effects = pd.DataFrame(rows)
    chosen = effects[(effects["events"] >= 10) & (effects["extra_pct"] >= 10) & (effects["t"] >= 2)]
    return effects, list(zip(chosen["kind"], chosen["venue"], chosen["station"]))


def add_event_candidates(df, events, pairs):
    """a. city_*         one city-wide flag per event kind (the same value at every station)
    b. event_nearby   1 at a chosen station on the shifts of its event kind (pairs from
                      choose_event_pairs)
    """
    df = df.copy()
    day = df[DATE_COL]
    city_flags = []
    for kind in sorted(events["kind"].unique()):
        column = "city_" + kind.lower().replace(" ", "_")
        df[column] = day.isin(set(events.loc[events["kind"] == kind, "ShiftDate"])).astype(int)
        city_flags.append(column)
    df["event_nearby"] = 0
    for kind, venue, station in pairs:
        df.loc[day.isin(event_days(events, kind, venue)) & df[STATION_COL].eq(station), "event_nearby"] = 1
    return df, {"city event flags": city_flags, "event_nearby": ["event_nearby"]}


# ---------------------------------------------------------------- all

def add_candidates(df):
    """All candidate groups (special days, calendar, history, events) at once. Returns (df, groups)."""
    df, groups = add_special_days(df)
    df, calendar_groups = add_calendar_candidates(df)
    groups.update(calendar_groups)
    df, lag_groups = add_lag_candidates(df)
    events = load_events()
    _, pairs = choose_event_pairs(df, events)
    df, event_groups = add_event_candidates(df, events, pairs)
    return df, {**groups, **lag_groups, **event_groups}
