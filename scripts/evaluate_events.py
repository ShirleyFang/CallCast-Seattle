# What it does
# Tests whether special days and city events improve the forecast, before
# adding them to the shared modeling dataset. The features are built here on
# the fly, so data/modeling_dataset.csv and the teammates' model are not changed.
# Uses the same rolling CV as scripts/evaluate_models.py (4 quarterly folds
# over 2025). The 2026 test set is not used here.
#
# Two parts:
#   Part 1  overall: does knowing "there is an event in the city today" help?
#   Part 2  per station: do events raise calls at the stations next to the venue,
#           and does a station-level event feature help?
# The station-event pairs in part 2 are chosen on 2022-2024 only, so the 2025
# folds that score them were not used to choose them.
#
# Run it from the repository root: uv run python scripts/evaluate_events.py
#

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from cv_utils import load_dataset, build_features, iter_folds, STATION_COL, DATE_COL, TARGET

# TODO
# [x] 1. Load data/modeling_dataset.csv, data/events.csv (home games, made by
#        scripts/fetch_events.py) and data/concerts.csv (made by scripts/fetch_concerts.py)

df = load_dataset()
games = pd.read_csv("data/events.csv", parse_dates=["ShiftDate"])
concerts = pd.read_csv("data/concerts.csv", parse_dates=["ShiftDate"])
stations = pd.read_csv("data/sfd_stations.csv").set_index("station")

# one table of events: kind (team, or stadium / arena concert), venue and shift
concerts["team"] = concerts["size"] + " concert"
events = pd.concat([games[["team", "venue", "ShiftDate"]], concerts[["team", "venue", "ShiftDate"]]])
events = events.rename(columns={"team": "kind"}).drop_duplicates()
print(events.groupby(["kind", events["ShiftDate"].dt.year]).size().unstack(fill_value=0).to_string())

# [x] 2. Special-day features (calendar only, known in advance; see evaluate_calendar_features.py):
#        is_nye, is_day_after_thanksgiving, is_xmas (Dec 24-26)

day = df[DATE_COL]
thanksgiving = day.dt.month.eq(11) & day.dt.dayofweek.eq(3) & day.dt.day.between(22, 28)
df["is_nye"] = (day.dt.month.eq(12) & day.dt.day.eq(31)).astype(int)
df["is_day_after_thanksgiving"] = day.isin(set(day[thanksgiving] + pd.Timedelta(days=1))).astype(int)
df["is_xmas"] = (day.dt.month.eq(12) & day.dt.day.between(24, 26)).astype(int)
SPECIAL = ["is_nye", "is_day_after_thanksgiving", "is_xmas"]

# [x] 3. Part 1 features: one city-wide flag per event kind (the same value at every station)

KINDS = sorted(events["kind"].unique())
CITY_FLAGS = []
for kind in KINDS:
    column = "city_" + kind.lower().replace(" ", "_")
    df[column] = day.isin(set(events.loc[events["kind"] == kind, "ShiftDate"])).astype(int)
    CITY_FLAGS.append(column)

# [x] 4. Part 2: event effect at the 2 stations nearest each venue, on 2022-2024 only
#        effect = calls on event shifts minus the mean of the same weekday +-4 weeks
#        without that kind of event (the same comparison as the earlier analysis)

VENUES = pd.DataFrame({"venue": ["Lumen Field", "T-Mobile Park", "Climate Pledge Arena", "Husky Stadium"],
                       "lat": [47.5952, 47.5914, 47.6221, 47.6503],
                       "lon": [-122.3316, -122.3325, -122.3540, -122.3016]}).set_index("venue")


def nearest_stations(venue, k=2):
    """The k stations closest to a venue (flat-map distance, scaled for latitude)."""
    lat, lon = VENUES.loc[venue, ["lat", "lon"]]
    dist = np.hypot(stations["lat"] - lat, (stations["lon"] - lon) * np.cos(np.radians(lat)))
    return list(dist.nsmallest(k).index)


calls = df.pivot(index=DATE_COL, columns=STATION_COL, values=TARGET).asfreq("D")
CHOOSE_END = pd.Timestamp("2025-01-01")     # choose the pairs on 2022-2024 only


def event_effect(station, days, before):
    """Mean extra calls on event shifts vs same-weekday shifts within +-4 weeks, using shifts before `before`."""
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


rows = []
for (kind, venue), group in events.groupby(["kind", "venue"]):
    for station in nearest_stations(venue):
        n, usual, extra, pct, t = event_effect(station, group["ShiftDate"], CHOOSE_END)
        rows.append({"kind": kind, "venue": venue, "station": station, "events": n,
                     "usual_calls": usual, "extra_calls": extra, "extra_pct": pct, "t": t})
effects = pd.DataFrame(rows)

# a pair is kept if the effect is clear on 2022-2024: at least 10 events, +10% or more, t >= 2
chosen = effects[(effects["events"] >= 10) & (effects["extra_pct"] >= 10) & (effects["t"] >= 2)]
PAIRS = list(zip(chosen["kind"], chosen["venue"], chosen["station"]))

# station-level flag: 1 at a chosen station on the shifts of its event kind
df["event_nearby"] = 0
for kind, venue, station in PAIRS:
    days = set(events.loc[(events["kind"] == kind) & (events["venue"] == venue), "ShiftDate"])
    df.loc[day.isin(days) & df[STATION_COL].eq(station), "event_nearby"] = 1

# [x] 5. Versions to compare (all with the same GBM settings as evaluate_models.py):
#        a. station x weekday mean (baseline)
#        b. current GBM (no new features)
#        c. + special days
#        d. + special days + city-wide event flags (part 1)
#        e. + special days + event_nearby as a GBM feature (part 2)
#        f. + special days, then add the average extra calls of each chosen pair
#           (learned from each fold's training shifts only) to the GBM forecast (part 2)

NEW = SPECIAL + CITY_FLAGS + ["event_nearby"]
X_all, _ = build_features(df, extra_exclude=["year"])
CURRENT = [c for c in X_all.columns if c not in NEW]


def fit_gbm(columns, train, validation):
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=200, random_state=0)
    model.fit(X_all.loc[train.index, columns], train[TARGET])
    return model.predict(X_all.loc[validation.index, columns])


def gbm(columns):
    return lambda train, validation: fit_gbm(columns, train, validation)


def gbm_plus_adjustment(train, validation):
    """GBM with special days, plus each chosen pair's mean extra calls from the training shifts."""
    pred = fit_gbm(CURRENT + SPECIAL, train, validation)
    # the +-4 week neighbours of a training event must be training shifts too
    cutoff = train[DATE_COL].max() - pd.Timedelta(days=28)
    for kind, venue, station in PAIRS:
        days = set(events.loc[(events["kind"] == kind) & (events["venue"] == venue), "ShiftDate"])
        extra = event_effect(station, days, cutoff)[2]
        rows_hit = (validation[DATE_COL].isin(days) & validation[STATION_COL].eq(station)).to_numpy()
        pred[rows_hit] += extra
    return pred


def station_weekday_mean(train, validation):
    """Mean calls for each station on each weekday, computed on training rows only."""
    means = train.groupby([STATION_COL, "weekday"])[TARGET].mean()
    overall = train[TARGET].mean()
    return np.array([means.get(k, overall) for k in zip(validation[STATION_COL], validation["weekday"])])


BASE = "+ special days"
VERSIONS = {
    "station x weekday": station_weekday_mean,
    "current GBM": gbm(CURRENT),
    BASE: gbm(CURRENT + SPECIAL),
    "+ special days + city event flags": gbm(CURRENT + SPECIAL + CITY_FLAGS),
    "+ special days + event_nearby": gbm(CURRENT + SPECIAL + ["event_nearby"]),
    "+ special days + event adjustment": gbm_plus_adjustment,
}

# [x] 6. Rolling CV: fit every version on each fold's training shifts, predict the validation quarter

parts = []
for fold_start, train, validation in iter_folds(df):
    fold = validation[[DATE_COL, STATION_COL, TARGET, "event_nearby"]].assign(fold=fold_start)
    for name, fit_predict in VERSIONS.items():
        fold[name] = fit_predict(train, validation)
    parts.append(fold)
predictions = pd.concat(parts)
names = list(VERSIONS)

# [x] 7. Scores:
#        - overall station MAE and citywide MAE, and in how many folds citywide MAE beats "+ special days"
#        - error and bias (forecast minus actual) on the chosen event-station shifts

station_err = predictions[names].sub(predictions[TARGET], axis=0).abs()
city = predictions.groupby(["fold", DATE_COL])[[TARGET] + names].sum()
city_err = city[names].sub(city[TARGET], axis=0).abs()
fold_city = city_err.groupby(level="fold").mean().T

overall = pd.DataFrame({"station_MAE": station_err.mean(), "citywide_MAE": city_err.mean()})
overall["folds_better_city"] = fold_city.lt(fold_city.loc[BASE], axis=1).sum(axis=1)

hit = predictions[predictions["event_nearby"] == 1]
on_events = pd.DataFrame({"MAE": hit[names].sub(hit[TARGET], axis=0).abs().mean(),
                          "bias": hit[names].sub(hit[TARGET], axis=0).mean()})

# [x] 8. Print the results; decide with the team before adding the features to step4_features.py
pd.set_option("display.width", 200)
print("\nPart 2: effect at the 2 nearest stations, 2022-2024 only:")
print(effects.sort_values("t", ascending=False).round(2).to_string(index=False))
print("\nChosen pairs (>= 10 events, >= +10%, t >= 2):")
print(chosen[["kind", "venue", "station", "events", "extra_calls", "extra_pct", "t"]].round(2).to_string(index=False))
print("\nOverall, 2025 rolling CV (4 folds):")
print(overall.round(3).to_string())
print("\nCitywide MAE per fold:")
print(fold_city.round(2).to_string())
print(f"\nOn the chosen event-station shifts in 2025 ({len(hit)} rows):")
print(on_events.round(3).to_string())


"""
Result

Part 2 effect, 2022-2024 (chosen pairs: >= 10 events, >= +10%, t >= 2):
kind              station  events  extra calls  extra %  t
Husky football    SFD 22   19      +4.4         +111%    5.95
Seahawks          SFD 10   27      +3.8         +11%     2.19
stadium concert   SFD 10   17      +4.1         +11%     2.11
Other sports and arena concerts at the nearest stations: +0-4%, not chosen.

2025 rolling CV (4 folds):
version                              station MAE  citywide MAE  on the 19 event-station shifts
                                                                 MAE     bias
current GBM                          2.409        18.95         7.27    -4.94
+ special days                       2.406        18.73         7.13    -4.82
+ special days + city event flags    2.409        18.92         7.26    -4.93
+ special days + event_nearby        2.406        18.86         6.75    -2.52
+ special days + event adjustment    2.405        18.75         6.17    -0.83

Part 1: a city-wide "event today" flag does not help (citywide MAE worse in
all 4 folds): events only change calls next to the venue.
Part 2: adding the extra calls learned from the training shifts at the 3
chosen pairs cuts the error on those shifts by 13% and almost removes the
under-forecast (bias -4.8 -> -0.8), with overall MAE unchanged (only 19 of
~12,000 rows). Using event_nearby as a GBM feature helps less: the trees
rarely split on a flag that is 1 in so few rows.
"""
