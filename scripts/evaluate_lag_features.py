# What it does
# Tests whether extra history features, built from our own call data, improve
# the forecast. The features are built here on the fly, so
# data/modeling_dataset.csv and the teammates' model are not changed.
# Uses the same rolling CV as scripts/evaluate_models.py (4 quarterly folds
# over 2025). The 2026 test set is not used here.
#
# Every new feature only uses shifts at least MIN_LAG (3) shifts before the
# forecast shift, the same rule as the existing lag3 / avg28 (no leakage).
#
# Run it from the repository root: uv run python scripts/evaluate_lag_features.py
#

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from cv_utils import (load_dataset, build_features, iter_folds,
                      STATION_COL, DATE_COL, TARGET, MIN_LAG)

# TODO
# [x] 1. Load data/modeling_dataset.csv (cv_utils.load_dataset), data/station_shift_counts.csv
#        (calls by type: fire, medical, non_emergency, other) and data/sfd_stations.csv (lat, lon)

df = load_dataset()
counts = pd.read_csv("data/station_shift_counts.csv", parse_dates=[DATE_COL])
counts["Masked"] = counts["Masked"].astype(str).str.lower().isin(["true", "1"])
stations = pd.read_csv("data/sfd_stations.csv").set_index("station")


def wide(column):
    """One row per shift, one column per station; outage shifts count as unknown (like step4_features.py)."""
    known = counts[column].where(~counts["Masked"])
    return counts.assign(known=known).pivot(index=DATE_COL, columns=STATION_COL, values="known").sort_index()


calls_wide = wide(TARGET)

# [x] 2. New features (masked outage shifts count as missing, like the existing lags):
#        a. citywide           citywide_ratio and citywide_lag3_ratio (see below)
#        b. neighbor_lag3      mean lag3 of the 3 nearest stations
#        c. trend              avg7 / avg28 (are calls going up or down lately?)
#        d. std28              standard deviation of the 28 shifts before lag3
#        e. type lags          medical_lag3, fire_lag3, non_emergency_lag3
#        f. lag364             calls 364 shifts earlier (same weekday, one year before)


def window_mean(table, first, length, min_count):
    """Mean over the `length` shifts starting `first` shifts back (per column)."""
    return table.shift(first).rolling(length, min_periods=min_count).mean()


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

# c. trend: avg7 / avg28 of this station (computed after the merge, from the existing columns)

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
df["trend"] = df["avg7"] / df["avg28"]

GROUPS = {
    "citywide": ["citywide_ratio", "citywide_lag3_ratio"],
    "neighbor": ["neighbor_lag3"],
    "trend": ["trend"],
    "std28": ["std28"],
    "type lags": ["medical_lag3", "fire_lag3", "non_emergency_lag3"],
    "lag364": ["lag364"],
}
ALL_NEW = [c for cols in GROUPS.values() for c in cols]
print("missing values in the new features (%):")
print((df[ALL_NEW].isna().mean() * 100).round(1).to_string())

# [x] 3. Versions to compare (same GBM settings as evaluate_models.py):
#        - station x weekday mean (baseline)
#        - current GBM (no new features)
#        - current GBM + each group a-f on its own
#        - current GBM + all groups that helped

# current features, same as evaluate_models.py (no year); the new columns are added per version
X_all, _ = build_features(df, extra_exclude=["year"])
CURRENT = [c for c in X_all.columns if c not in ALL_NEW]


def gbm(columns):
    """Poisson GBM on the given feature columns."""
    def fit_predict(train, validation):
        model = HistGradientBoostingRegressor(loss="poisson", max_iter=200, random_state=0)
        model.fit(X_all.loc[train.index, columns], train[TARGET])
        return model.predict(X_all.loc[validation.index, columns])
    return fit_predict


def station_weekday_mean(train, validation):
    """Mean calls for each station on each weekday, computed on training rows only."""
    means = train.groupby([STATION_COL, "weekday"])[TARGET].mean()
    overall = train[TARGET].mean()
    return np.array([means.get(k, overall) for k in zip(validation[STATION_COL], validation["weekday"])])


VERSIONS = {"station x weekday": station_weekday_mean, "current GBM": gbm(CURRENT)}
for group, cols in GROUPS.items():
    VERSIONS[f"+ {group}"] = gbm(CURRENT + cols)

# [x] 4. Rolling CV: fit every version on each fold's training shifts, predict the validation quarter


def run_cv(versions):
    """One row per validation station-shift, one column per version."""
    parts = []
    for fold_start, train, validation in iter_folds(df):
        fold = validation[[DATE_COL, STATION_COL, TARGET]].assign(fold=fold_start)
        for name, fit_predict in versions.items():
            fold[name] = fit_predict(train, validation)
        parts.append(fold)
    return pd.concat(parts)


predictions = run_cv(VERSIONS)

# [x] 5. Scores:
#        - overall station MAE and citywide MAE, and the change vs the current GBM
#        - change per fold (a feature should help in most folds, not just one)
#        - station MAE at the stations farthest from the noise floor (SFD 10, 9, 17)


def scores(pred, names):
    """Station MAE and citywide MAE (sum the stations per shift), overall and per fold."""
    station = pred[names].sub(pred[TARGET], axis=0).abs()
    city = pred.groupby(["fold", DATE_COL])[[TARGET] + names].sum()
    city_err = city[names].sub(city[TARGET], axis=0).abs()
    overall = pd.DataFrame({"station_MAE": station.mean(), "citywide_MAE": city_err.mean()})
    per_fold_station = station.groupby(pred["fold"]).mean().T
    per_fold_city = city_err.groupby(level="fold").mean().T
    return overall, per_fold_station, per_fold_city


BASE = "current GBM"


def add_changes(overall, fold_city):
    """Change vs the current GBM, and in how many folds the citywide MAE is lower."""
    overall["station_change_%"] = 100 * (overall["station_MAE"] / overall.loc[BASE, "station_MAE"] - 1)
    overall["citywide_change_%"] = 100 * (overall["citywide_MAE"] / overall.loc[BASE, "citywide_MAE"] - 1)
    overall["folds_better_city"] = fold_city.lt(fold_city.loc[BASE], axis=1).sum(axis=1)
    return overall


names = list(VERSIONS)
overall, fold_station, fold_city = scores(predictions, names)
overall = add_changes(overall, fold_city)

# a group helps if it lowers citywide MAE in at least 3 of 4 folds and does not make station MAE worse
helped = [g for g in GROUPS
          if overall.loc[f"+ {g}", "folds_better_city"] >= 3
          and overall.loc[f"+ {g}", "station_MAE"] <= overall.loc[BASE, "station_MAE"]]

# the groups that helped, together
if len(helped) > 1:
    name = "+ " + " + ".join(helped)
    cols = [c for g in helped for c in GROUPS[g]]
    predictions[name] = run_cv({name: gbm(CURRENT + cols)})[name].to_numpy()
    names.append(name)
    overall, fold_station, fold_city = scores(predictions, names)
    overall = add_changes(overall, fold_city)

# station MAE at the stations farthest from the noise floor (see evaluate_by_station.py)
hard = predictions[predictions[STATION_COL].isin(["SFD 10", "SFD 9", "SFD 17"])]
hard_mae = hard[names[1:]].sub(hard[TARGET], axis=0).abs().groupby(hard[STATION_COL]).mean()

# [x] 6. Print the results; decide with the team before adding the features to step4_features.py
pd.set_option("display.width", 200)
print("\nOverall, 2025 rolling CV (4 folds):")
print(overall.round(3).to_string())
print("\nCitywide MAE per fold:")
print(fold_city.round(2).to_string())
print("\nStation MAE per fold:")
print(fold_station.round(3).to_string())
print("\nStation MAE at the hardest stations:")
print(hard_mae.T.round(3).to_string())
print("\nGroups that helped (citywide better in >= 3 of 4 folds, station MAE not worse):", helped or "none")


"""
Result (2025 rolling CV, 4 folds):

version            station MAE  citywide MAE  citywide folds better
current GBM        2.409        18.95         -
+ citywide         2.414        19.40         1 of 4
+ neighbor         2.411        19.17         0 of 4
+ trend            2.413        19.22         0 of 4
+ std28            2.415        19.48         0 of 4
+ type lags        2.414        19.19         1 of 4
+ lag364           2.408        19.15         1 of 4

No group helped: station MAE changes by less than 0.3%, and citywide MAE is
1-3% worse. The existing lags (lag3, lag7, avg7, avg28, same_weekday_avg4)
already hold the information in these features; extra columns only add noise.
None of them is added to step4_features.py.
"""
