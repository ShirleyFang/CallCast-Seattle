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
from cv_utils import load_dataset, build_features, iter_folds, metrics, STATION_COL, DATE_COL, TARGET
from feature_sets import (baseline_columns, SPECIAL_DAYS, add_special_days, load_events, event_days,
                          calls_by_station, event_effect, choose_event_pairs, add_event_candidates)

# [x] 1. Load data/modeling_dataset.csv and the events (feature_sets.load_events: data/events.csv,
#        home games made by scripts/fetch_events.py, and data/concerts.csv, made by
#        scripts/fetch_concerts.py)

df = load_dataset()
events = load_events()
print(events.groupby(["kind", events["ShiftDate"].dt.year]).size().unstack(fill_value=0).to_string())

# [x] 2. Special-day features (calendar only, known in advance; see evaluate_calendar_features.py):
#        is_nye, is_day_after_thanksgiving, is_xmas (Dec 24-26), added in memory
#        (feature_sets.add_special_days); not part of the baseline. The event versions
#        build on them, as they are proposed for the final model.

df, _ = add_special_days(df)
day = df[DATE_COL]
SPECIAL = SPECIAL_DAYS

# [x] 3. Part 2 analysis: event effect at the 2 stations nearest each venue, on 2022-2024 only
#        (feature_sets.choose_event_pairs). effect = calls on event shifts minus the mean of the
#        same weekday +-4 weeks without that kind of event. A pair is kept if the effect is
#        clear: at least 10 events, +10% or more, t >= 2.

effects, PAIRS = choose_event_pairs(df, events)
chosen = effects.set_index(["kind", "venue", "station"]).loc[PAIRS].reset_index()
calls = calls_by_station(df)

# [x] 4. Candidate features, added in memory (feature_sets.add_event_candidates):
#        Part 1  city_*        one city-wide flag per event kind (the same value at every station)
#        Part 2  event_nearby  1 at a chosen station on the shifts of its event kind

df, EVENT_GROUPS = add_event_candidates(df, events, PAIRS)
CITY_FLAGS = EVENT_GROUPS["city event flags"]

# [x] 5. Versions to compare (all with the same GBM settings as evaluate_models.py):
#        a. station x weekday mean (baseline)
#        b. current GBM (no new features)
#        c. + special days
#        d. + special days + city-wide event flags (part 1)
#        e. + special days + event_nearby as a GBM feature (part 2)
#        f. + special days, then add the average extra calls of each chosen pair
#           (learned from each fold's training shifts only) to the GBM forecast (part 2)
#        g. as f, but the amount added is the GBM's own mean residual (actual minus
#           forecast) on the pair's training event shifts, from out-of-fold predictions.
#           The raw uplift in f may count part of the effect twice if the GBM already
#           predicts some of it (e.g. through weekday, month or lag7).

X_all, _ = build_features(df, extra_exclude=["year"])
CURRENT = baseline_columns(X_all)   # feature_sets.BASELINE_NUMERIC + station and weekday dummies


def fit_gbm(columns, train, validation):
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=200, random_state=0)
    model.fit(X_all.loc[train.index, columns], train[TARGET])
    return model.predict(X_all.loc[validation.index, columns])


def gbm(columns):
    return lambda train, validation: fit_gbm(columns, train, validation)


def pair_days(kind, venue):
    return event_days(events, kind, venue)


# amount added per fold and pair, by method (printed at the end)
adjustments = []


def gbm_plus_adjustment(train, validation):
    """GBM with special days, plus each chosen pair's mean extra calls from the training shifts."""
    pred = fit_gbm(CURRENT + SPECIAL, train, validation)
    # the +-4 week neighbours of a training event must be training shifts too
    cutoff = train[DATE_COL].max() - pd.Timedelta(days=28)
    for kind, venue, station in PAIRS:
        days = pair_days(kind, venue)
        extra = event_effect(calls, station, days, cutoff)[2]
        rows_hit = (validation[DATE_COL].isin(days) & validation[STATION_COL].eq(station)).to_numpy()
        pred[rows_hit] += extra
        adjustments.append({"fold": validation[DATE_COL].min().date(), "pair": f"{kind} @ {station}",
                            "method": "uplift (f)", "added": extra})
    return pred


def out_of_fold(train, rows):
    """GBM forecasts for the given training rows, each from a model that did not see its year."""
    year = train[DATE_COL].dt.year
    pred = pd.Series(np.nan, index=rows)
    for y in sorted(year.loc[rows].unique()):
        held_out = rows[year.loc[rows] == y]
        pred[held_out] = fit_gbm(CURRENT + SPECIAL, train[year != y], train.loc[held_out])
    return pred


def gbm_plus_residual_adjustment(train, validation):
    """GBM with special days, plus each pair's mean out-of-fold residual on its training event shifts."""
    pred = fit_gbm(CURRENT + SPECIAL, train, validation)
    hits = {pair: train.index[train[DATE_COL].isin(pair_days(*pair[:2])) & train[STATION_COL].eq(pair[2])]
            for pair in PAIRS}
    oof = out_of_fold(train, pd.Index(sorted(set().union(*hits.values()))))
    for (kind, venue, station), rows in hits.items():
        extra = (train.loc[rows, TARGET] - oof[rows]).mean()
        rows_hit = (validation[DATE_COL].isin(pair_days(kind, venue)) & validation[STATION_COL].eq(station)).to_numpy()
        pred[rows_hit] += extra
        adjustments.append({"fold": validation[DATE_COL].min().date(), "pair": f"{kind} @ {station}",
                            "method": "residual (g)", "added": extra})
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
    "+ special days + residual adjustment": gbm_plus_residual_adjustment,
}

# which chosen pair (if any) each row belongs to; the two SFD 10 pairs never share a night
df["pair"] = ""
for kind, venue, station in PAIRS:
    df.loc[day.isin(pair_days(kind, venue)) & df[STATION_COL].eq(station), "pair"] = f"{kind} @ {station}"

# [x] 6. Rolling CV: fit every version on each fold's training shifts, predict the validation quarter

parts = []
for fold_start, train, validation in iter_folds(df):
    fold = validation[[DATE_COL, STATION_COL, TARGET, "event_nearby", "pair"]].assign(fold=fold_start)
    for name, fit_predict in VERSIONS.items():
        fold[name] = fit_predict(train, validation)
    parts.append(fold)
predictions = pd.concat(parts)
names = list(VERSIONS)

# [x] 7. Scores:
#        - station MAE and citywide MAE (mean over the folds), and in how many folds citywide MAE beats "+ special days"
#        - error and bias (forecast minus actual) on the chosen event-station shifts

# cv_utils.metrics() per fold, then the mean over the folds (as in evaluate_models.py)
per_fold = pd.DataFrame([{"version": name, "fold": fold, **metrics(part, part[name].to_numpy())}
                         for fold, part in predictions.groupby("fold") for name in names])
overall = per_fold.groupby("version", sort=False)[["station_MAE", "station_RMSE", "citywide_MAE"]].mean()
fold_city = per_fold.pivot(index="version", columns="fold", values="citywide_MAE").loc[names]
overall["folds_better_city"] = fold_city.lt(fold_city.loc[BASE], axis=1).sum(axis=1)

hit = predictions[predictions["event_nearby"] == 1]
on_events = pd.DataFrame({"MAE": hit[names].sub(hit[TARGET], axis=0).abs().mean(),
                          "bias": hit[names].sub(hit[TARGET], axis=0).mean()})

# bias per pair, for the versions that target the event shifts
ADJUSTED = [BASE, "+ special days + event adjustment", "+ special days + residual adjustment"]
by_pair = hit[ADJUSTED].sub(hit[TARGET], axis=0).groupby(hit["pair"]).mean()
by_pair.insert(0, "shifts", hit.groupby("pair").size())
added = pd.DataFrame(adjustments).pivot_table(index=["pair", "fold"], columns="method", values="added")

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
print("\nBias (forecast minus actual) per pair on its 2025 event shifts:")
print(by_pair.round(2).to_string())
print("\nCalls added per event shift, by fold (learned from that fold's training shifts):")
print(added.round(2).to_string())


"""
Result

Part 2 effect, 2022-2024 (chosen pairs: >= 10 events, >= +10%, t >= 2):
kind              station  events  extra calls  extra %  t
Husky football    SFD 22   19      +4.4         +111%    5.95
Seahawks          SFD 10   27      +3.8         +11%     2.19
stadium concert   SFD 10   17      +4.1         +11%     2.11
Other sports and arena concerts at the nearest stations: +0-4%, not chosen.

2025 rolling CV (4 folds; station and citywide MAE = mean over the folds, as in evaluate_models.py):
version                              station MAE  citywide MAE  on the 19 event-station shifts
                                                                 MAE     bias
current GBM                          2.409        18.95         7.27    -4.94
+ special days                       2.406        18.73         7.13    -4.82
+ special days + city event flags    2.409        18.92         7.26    -4.93
+ special days + event_nearby        2.405        18.85         6.75    -2.52
+ special days + event adjustment    2.405        18.75         6.17    -0.83
+ special days + residual adjustment 2.405        18.74         6.17    -1.03

Bias per pair on its 2025 event shifts (+ special days / uplift f / residual g):
Husky football @ SFD 22   (7 shifts)   -2.18 / +2.07 / +1.56
Seahawks @ SFD 10         (8 shifts)   -5.71 / -1.69 / -1.51
stadium concert @ SFD 10  (4 shifts)   -7.66 / -4.16 / -4.59
Calls added per shift, mean over folds (uplift f / residual g):
Husky 4.33 / 3.74, Seahawks 3.97 / 4.24, stadium concert 3.98 / 3.74.

Part 1: a city-wide "event today" flag does not help (citywide MAE worse in
all 4 folds): events only change calls next to the venue.
Part 2: adding the extra calls learned from the training shifts at the 3
chosen pairs cuts the error on those shifts by 13% and almost removes the
under-forecast (bias -4.8 -> -0.8), with overall MAE unchanged (only 19 of
~12,000 rows). Using event_nearby as a GBM feature helps less: the trees
rarely split on a flag that is 1 in so few rows.
Uplift vs residual: the two give the same MAE on the event shifts (6.17).
At Husky football the GBM already predicts part of the effect, so the raw
uplift over-adjusts (bias +2.1); the residual adds ~0.6 fewer calls (bias
+1.6). At SFD 10 both still under-forecast. With 4-8 shifts per pair in
2025, the difference between the two methods is within noise.
"""
