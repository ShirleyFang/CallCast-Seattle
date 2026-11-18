# What it does
# Tests whether calendar features improve the forecast: special days (New
# Year's Eve, the day after Thanksgiving(Black Friday), Christmas), the start of the month,
# the UW and Seattle Public Schools calendars, long weekends and day length. All of them are known
# long before the forecast shift (no leakage). The candidates are added in memory by
# scripts/feature_sets.py, so data/modeling_dataset.csv and the teammates' model are not changed.
# Uses the same rolling CV and scoring as scripts/evaluate_models.py (4 quarterly folds
# over 2025, metrics per fold, then the mean over the folds). The 2026 test set is not used here.
#
# Run it from the repository root: uv run python scripts/evaluate_calendar_features.py
#

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from cv_utils import load_dataset, build_features, iter_folds, metrics, STATION_COL, DATE_COL, TARGET
from feature_sets import baseline_columns, SPECIAL_DAYS, add_special_days, add_calendar_candidates

# Load data/modeling_dataset.csv (cv_utils.load_dataset)

df = load_dataset()

# Special days (found in the error analysis), added in memory (feature_sets.add_special_days):
# is_nye, is_day_after_thanksgiving, is_xmas (Dec 24-26). They are not part of the
# baseline; this script gives the evidence for proposing them for the final model.

df, _ = add_special_days(df)
SPECIAL = SPECIAL_DAYS

# New calendar features, added in memory (feature_sets.add_calendar_candidates):
#     a. is_month_start   days 1-3 of the month
#     b. uw_in_session    UW autumn, winter or spring quarter
#     c. long_weekend     a weekend or holiday that is part of 3 or more days off in a row
#     d. day_length       hours of daylight in Seattle
#     e. sps_break        Seattle Public Schools mid-winter break and spring break

df, GROUPS = add_calendar_candidates(df)

ALL_NEW = SPECIAL + [c for cols in GROUPS.values() for c in cols]
print("share of shifts flagged:")
print(df[ALL_NEW].drop(columns="day_length").mean().round(3).to_string())

# raw check on the training years only (2022-2024): SPS break weekdays vs the other
# weekdays of February and April, citywide calls per shift
city = df[df[DATE_COL] < "2025-01-01"].groupby(DATE_COL).agg(calls=(TARGET, "sum"), sps=("sps_break", "first"))
city = city[city.index.month.isin([2, 4]) & (city.index.dayofweek < 5)]
print(f"\ncitywide calls, Feb + Apr weekdays 2022-2024: SPS break {city.loc[city.sps == 1, 'calls'].mean():.1f}, "
      f"school days {city.loc[city.sps == 0, 'calls'].mean():.1f}")

# Versions to compare (same GBM settings as evaluate_models.py):
#     - station x weekday mean
#     - current GBM = baseline model (no new features)
#     - current GBM + special days
#     - current GBM + special days + each group a-e on its own
#     - current GBM + special days + all groups that helped

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


BASE = "+ special days"
VERSIONS = {"station x weekday": station_weekday_mean,
            "current GBM": gbm(CURRENT),
            BASE: gbm(CURRENT + SPECIAL)}
for group, cols in GROUPS.items():
    VERSIONS[f"+ special days + {group}"] = gbm(CURRENT + SPECIAL + cols)

# Rolling CV: fit every version on each fold's training shifts, predict the validation quarter


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

# Scores:
#     - station MAE and citywide MAE (mean over the folds), and the change vs "+ special days"
#     - change per fold (a feature should help in most folds, not just one)
#     - UW in session: station MAE at SFD 17 (U District) and SFD 22
#     The "+ special days" row is the comparison row, so its changes and folds_better_city are 0.


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
helped = [g for g in GROUPS if helps(f"+ special days + {g}")]

# the groups that helped, together
if len(helped) > 1:
    name = "+ special days + " + " + ".join(helped)
    cols = [c for g in helped for c in GROUPS[g]]
    predictions[name] = run_cv({name: gbm(CURRENT + SPECIAL + cols)})[name].to_numpy()
    names.append(name)
    overall, per_fold = scores(predictions, names)

fold_city = per_fold.pivot(index="version", columns="fold", values="citywide_MAE").loc[names]

# the UW calendar should matter most next to campus
uw = predictions[predictions[STATION_COL].isin(["SFD 17", "SFD 22"])]
uw_mae = uw[names[1:]].sub(uw[TARGET], axis=0).abs().groupby(uw[STATION_COL]).mean()

# Print the results; the model evaluation decides which candidates enter the final model
pd.set_option("display.width", 200)
print("\nOverall, 2025 rolling CV (4 folds):")
print(overall.round(3).to_string())
print("\nCitywide MAE per fold:")
print(fold_city.round(2).to_string())
print("\nStation MAE next to UW (SFD 17, SFD 22):")
print(uw_mae.T.round(3).to_string())
# special days only change a few shifts (all in Q4), so judge them on those shifts, not on folds
city = predictions.groupby(DATE_COL)[[TARGET, "current GBM", BASE]].sum()
err = city[["current GBM", BASE]].sub(city[TARGET], axis=0).abs()
is_special = df.groupby(DATE_COL)[SPECIAL].max().max(axis=1).reindex(city.index).astype(bool)
print(f"\nSpecial days vs current GBM, citywide MAE:"
      f"\n  on the {is_special.sum()} special-day shifts in 2025: {err.loc[is_special, 'current GBM'].mean():.1f} -> "
      f"{err.loc[is_special, BASE].mean():.1f} (better on {(err.loc[is_special, BASE] < err.loc[is_special, 'current GBM']).sum()} of {is_special.sum()})"
      f"\n  on the other shifts: {err.loc[~is_special, 'current GBM'].mean():.2f} -> {err.loc[~is_special, BASE].mean():.2f}")

print("Other groups that pass the first screen on top of special days "
      "(citywide better in >= 3 of 4 folds, station MAE not worse):", helped or "none")



"""
Result (2025 rolling CV, 4 folds; MAE = mean over the folds, as in evaluate_models.py):

version                          station MAE  citywide MAE  citywide folds better
current GBM                      2.409        18.95         -
+ special days                   2.406        18.73         (see below)
+ special days + month start     2.407        18.83         1 of 4 vs + special days
+ special days + UW in session   2.410        18.94         1 of 4
+ special days + long weekend    2.411        19.11         1 of 4
+ special days + day length      2.410        19.03         1 of 4
+ special days + SPS break       2.404        18.78         2 of 4

Special days help. All of them fall in Q4, so the fold count says nothing about
them (Q1-Q3 changes are noise); they are judged on the special-day shifts:
citywide MAE on the 5 special-day shifts of 2025 44.0 -> 26.3, better on all 5;
other shifts 18.60 -> 18.63. Over 2004-2025, calls were higher on New Year's Eve
in 22 of 22 years and lower on the day after Thanksgiving in 20 of 22.
None of the other calendar features adds anything on top: month and weekday
already cover day length and long weekends, and the UW calendar does not
lower the error even at SFD 17 and SFD 22 next to campus. On SPS break
weekdays, citywide calls are about the same as on school days (293.6 vs 296.8
per shift, Feb + Apr 2022-2024): more injuries in parks may be offset by
fewer injuries at school and families leaving town. Only the special days are
worth proposing for the final model (feature_sets.add_special_days).
"""
