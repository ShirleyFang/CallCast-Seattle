# What it does
# Tests whether extra history features, built from our own call data, improve
# the forecast. The candidates are added in memory by scripts/feature_sets.py, so
# data/modeling_dataset.csv and the teammates' model are not changed.
# Uses the same rolling CV and scoring as scripts/evaluate_models.py (4 quarterly folds
# over 2025, metrics per fold, then the mean over the folds). The 2026 test set is not used here.
#
# Every new feature only uses shifts at least MIN_LAG (3) shifts before the
# forecast shift, the same rule as the existing lag3 / avg28 (no leakage).
#
# Run it from the repository root: uv run python scripts/evaluate_lag_features.py
#

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from cv_utils import load_dataset, build_features, iter_folds, metrics, STATION_COL, DATE_COL, TARGET
from feature_sets import baseline_columns, add_lag_candidates

# [x] 1. Load data/modeling_dataset.csv (cv_utils.load_dataset)

df = load_dataset()

# [x] 2. New features, added in memory (feature_sets.add_lag_candidates; masked outage
#        shifts count as missing, like the existing lags):
#        a. citywide           citywide_ratio and citywide_lag3_ratio
#        b. neighbor_lag3      mean lag3 of the 3 nearest stations
#        c. trend              avg7 / avg28 (are calls going up or down lately?)
#        d. std28              standard deviation of the 28 shifts before lag3
#        e. type lags          medical_lag3, fire_lag3, non_emergency_lag3
#        f. lag364             calls 364 shifts earlier (same weekday, one year before)

df, GROUPS = add_lag_candidates(df)
ALL_NEW = [c for cols in GROUPS.values() for c in cols]
print("missing values in the new features (%):")
print((df[ALL_NEW].isna().mean() * 100).round(1).to_string())

# [x] 3. Versions to compare (same GBM settings as evaluate_models.py):
#        - station x weekday mean
#        - current GBM = baseline model (no new features)
#        - current GBM + each group a-f on its own
#        - current GBM + all groups that helped
#        Run before the special days were accepted, so the groups are tested on the
#        baseline model alone.

# baseline model: feature_sets.BASELINE_NUMERIC + station and weekday dummies (no year);
# the new columns are added per version
X_all, _ = build_features(df, extra_exclude=["year"])
CURRENT = baseline_columns(X_all)


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


BASE = "current GBM"
VERSIONS = {"station x weekday": station_weekday_mean, BASE: gbm(CURRENT)}
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
#        - station MAE and citywide MAE (mean over the folds), and the change vs the current GBM
#        - change per fold (a feature should help in most folds, not just one)
#        - station MAE at the stations farthest from the noise floor (SFD 10, 9, 17)


def scores(pred, names):
    """cv_utils.metrics() per fold, then the mean over the folds (as in evaluate_models.py),
    and the change vs BASE. Returns (overall, per_fold)."""
    per_fold = pd.DataFrame([{"version": name, "fold": fold, **metrics(part, part[name].to_numpy())}
                             for fold, part in pred.groupby("fold") for name in names])
    overall = per_fold.groupby("version", sort=False)[["station_MAE", "station_RMSE", "citywide_MAE"]].mean()
    for col in ["station_MAE", "citywide_MAE"]:
        overall[col.replace("MAE", "change_%")] = 100 * (overall[col] / overall.loc[BASE, col] - 1)
    city = per_fold.pivot(index="version", columns="fold", values="citywide_MAE")
    overall["folds_better_city"] = city.lt(city.loc[BASE], axis=1).sum(axis=1).reindex(overall.index)
    return overall, per_fold


def helps(name):
    """Selection rule: citywide MAE lower in at least 3 of the 4 folds, station MAE not worse."""
    return (overall.loc[name, "folds_better_city"] >= 3
            and overall.loc[name, "station_MAE"] <= overall.loc[BASE, "station_MAE"])


names = list(VERSIONS)
overall, per_fold = scores(predictions, names)
helped = [g for g in GROUPS if helps(f"+ {g}")]

# the groups that helped, together
if len(helped) > 1:
    name = "+ " + " + ".join(helped)
    cols = [c for g in helped for c in GROUPS[g]]
    predictions[name] = run_cv({name: gbm(CURRENT + cols)})[name].to_numpy()
    names.append(name)
    overall, per_fold = scores(predictions, names)

fold_city = per_fold.pivot(index="version", columns="fold", values="citywide_MAE").loc[names]
fold_station = per_fold.pivot(index="version", columns="fold", values="station_MAE").loc[names]

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
Result (2025 rolling CV, 4 folds; MAE = mean over the folds, as in evaluate_models.py):

version            station MAE  citywide MAE  citywide folds better
current GBM        2.409        18.95         -
+ citywide         2.414        19.40         1 of 4
+ neighbor         2.411        19.17         0 of 4
+ trend            2.413        19.21         0 of 4
+ std28            2.415        19.48         0 of 4
+ type lags        2.414        19.19         1 of 4
+ lag364           2.408        19.14         1 of 4

No group helped: station MAE changes by less than 0.3%, and citywide MAE is
1-3% worse. The existing lags (lag3, lag7, avg7, avg28, same_weekday_avg4)
already hold the information in these features; extra columns only add noise.
None of them is added to step4_features.py.
"""
