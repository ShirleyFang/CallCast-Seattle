# What it does
# Tests whether calendar features improve the forecast: special days (New
# Year's Eve, the day after Thanksgiving, Christmas), the start of the month,
# the UW and Seattle Public Schools calendars, long weekends and day length. All of them are known
# long before the forecast shift (no leakage). The features are built here on
# the fly, so data/modeling_dataset.csv and the teammates' model are not changed.
# Uses the same rolling CV as scripts/evaluate_models.py (4 quarterly folds
# over 2025). The 2026 test set is not used here.
#
# Run it from the repository root: uv run python scripts/evaluate_calendar_features.py
#

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from cv_utils import load_dataset, build_features, iter_folds, STATION_COL, DATE_COL, TARGET

# TODO
# [x] 1. Load data/modeling_dataset.csv (cv_utils.load_dataset)

df = load_dataset()
day = df[DATE_COL]

# [x] 2. Special days (found in the error analysis; earlier CV: citywide MAE 18.95 -> 18.73)
#        is_nye, is_day_after_thanksgiving, is_xmas (Dec 24-26)

thanksgiving = day.dt.month.eq(11) & day.dt.dayofweek.eq(3) & day.dt.day.between(22, 28)
df["is_nye"] = (day.dt.month.eq(12) & day.dt.day.eq(31)).astype(int)
df["is_day_after_thanksgiving"] = day.isin(set(day[thanksgiving] + pd.Timedelta(days=1))).astype(int)
df["is_xmas"] = (day.dt.month.eq(12) & day.dt.day.between(24, 26)).astype(int)

# [x] 3. New calendar features
#        a. is_month_start   days 1-3 of the month (benefits and pay arrive; Ray 2016)
#        b. uw_in_session    UW autumn, winter or spring quarter, first class day to last exam day
#                            (summer quarter is small and left out)
#        c. long_weekend     a weekend or holiday that is part of 3 or more days off in a row
#        d. day_length       hours of daylight in Seattle (smoother than month)
#        e. sps_break        Seattle Public Schools mid-winter break and spring break, Mon-Fri
#                            (summer and winter break are left out: month and the holiday
#                            flags already cover them)

df["is_month_start"] = day.dt.day.le(3).astype(int)

# UW quarters, from washington.edu/students/reg (yearly calendars and the 2024-2033 summary)
UW_QUARTERS = [
    ("2022-01-03", "2022-03-18"), ("2022-03-28", "2022-06-10"), ("2022-09-28", "2022-12-16"),
    ("2023-01-03", "2023-03-17"), ("2023-03-27", "2023-06-09"), ("2023-09-27", "2023-12-15"),
    ("2024-01-03", "2024-03-15"), ("2024-03-25", "2024-06-07"), ("2024-09-25", "2024-12-13"),
    ("2025-01-06", "2025-03-21"), ("2025-03-31", "2025-06-13"), ("2025-09-24", "2025-12-12"),
    ("2026-01-05", "2026-03-20"), ("2026-03-30", "2026-06-12"), ("2026-09-30", "2026-12-18"),
]
in_session = pd.Series(False, index=df.index)
for first, last in UW_QUARTERS:
    in_session |= day.between(first, last)
df["uw_in_session"] = in_session.astype(int)

# days off: Saturday, Sunday or a holiday (is_holiday from step4_features.py)
calendar = pd.DataFrame({"date": pd.date_range(day.min() - pd.Timedelta(days=7), day.max() + pd.Timedelta(days=7))})
holidays = set(df.loc[df["is_holiday"] == 1, DATE_COL])
calendar["off"] = calendar["date"].dt.dayofweek.ge(5) | calendar["date"].isin(holidays)
run_id = calendar["off"].ne(calendar["off"].shift()).cumsum()          # number each run of days off / work days
run_length = calendar.groupby(run_id)["off"].transform("size")
long_weekend_days = set(calendar.loc[calendar["off"] & run_length.ge(3), "date"])
df["long_weekend"] = day.isin(long_weekend_days).astype(int)

# daylight hours from the sun's declination at Seattle's latitude (47.6 N)
declination = np.radians(23.44) * np.sin(2 * np.pi * (284 + day.dt.dayofyear) / 365)
latitude = np.radians(47.6)
df["day_length"] = 24 / np.pi * np.arccos(-np.tan(latitude) * np.tan(declination))

# SPS breaks follow fixed rules in the district calendars (checked against the approved
# 2021-22, 2022-23, 2024-25 and 2025-26 calendars on seattleschools.org):
#   mid-winter break = the week of Presidents Day (3rd Monday of February)
#   spring break     = the week starting the 2nd Monday of April
def week_from(monday):
    return pd.date_range(monday, periods=5)


sps_days = set()
for year in range(day.dt.year.min(), day.dt.year.max() + 1):
    february_mondays = pd.date_range(f"{year}-02-01", f"{year}-02-28", freq="W-MON")
    april_mondays = pd.date_range(f"{year}-04-01", f"{year}-04-30", freq="W-MON")
    sps_days |= set(week_from(february_mondays[2])) | set(week_from(april_mondays[1]))
df["sps_break"] = day.isin(sps_days).astype(int)

SPECIAL = ["is_nye", "is_day_after_thanksgiving", "is_xmas"]
GROUPS = {
    "month start": ["is_month_start"],
    "UW in session": ["uw_in_session"],
    "long weekend": ["long_weekend"],
    "day length": ["day_length"],
    "SPS break": ["sps_break"],
}
ALL_NEW = SPECIAL + [c for cols in GROUPS.values() for c in cols]
print("share of shifts flagged:")
print(df[ALL_NEW].drop(columns="day_length").mean().round(3).to_string())

# raw check on the training years only (2022-2024): SPS break weekdays vs the other
# weekdays of February and April, citywide calls per shift
city = df[df[DATE_COL] < "2025-01-01"].groupby(DATE_COL).agg(calls=(TARGET, "sum"), sps=("sps_break", "first"))
city = city[city.index.month.isin([2, 4]) & (city.index.dayofweek < 5)]
print(f"\ncitywide calls, Feb + Apr weekdays 2022-2024: SPS break {city.loc[city.sps == 1, 'calls'].mean():.1f}, "
      f"school days {city.loc[city.sps == 0, 'calls'].mean():.1f}")

# [x] 4. Versions to compare (same GBM settings as evaluate_models.py):
#        - station x weekday mean (baseline)
#        - current GBM (no new features)
#        - current GBM + special days
#        - current GBM + special days + each group a-d on its own
#        - current GBM + special days + all groups that helped

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


BASE = "+ special days"
VERSIONS = {"station x weekday": station_weekday_mean,
            "current GBM": gbm(CURRENT),
            BASE: gbm(CURRENT + SPECIAL)}
for group, cols in GROUPS.items():
    VERSIONS[f"+ special days + {group}"] = gbm(CURRENT + SPECIAL + cols)

# [x] 5. Rolling CV: fit every version on each fold's training shifts, predict the validation quarter


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

# [x] 6. Scores:
#        - overall station MAE and citywide MAE, and the change vs "+ special days"
#        - change per fold (a feature should help in most folds, not just one)
#        - UW in session: station MAE at SFD 17 (U District) and SFD 22


def scores(pred, names):
    """Station MAE and citywide MAE (sum the stations per shift), overall and per fold."""
    station = pred[names].sub(pred[TARGET], axis=0).abs()
    city = pred.groupby(["fold", DATE_COL])[[TARGET] + names].sum()
    city_err = city[names].sub(city[TARGET], axis=0).abs()
    overall = pd.DataFrame({"station_MAE": station.mean(), "citywide_MAE": city_err.mean()})
    per_fold_city = city_err.groupby(level="fold").mean().T
    return overall, per_fold_city


def add_changes(overall, fold_city):
    """Change vs "+ special days", and in how many folds the citywide MAE is lower."""
    overall["station_change_%"] = 100 * (overall["station_MAE"] / overall.loc[BASE, "station_MAE"] - 1)
    overall["citywide_change_%"] = 100 * (overall["citywide_MAE"] / overall.loc[BASE, "citywide_MAE"] - 1)
    overall["folds_better_city"] = fold_city.lt(fold_city.loc[BASE], axis=1).sum(axis=1)
    return overall


names = list(VERSIONS)
overall, fold_city = scores(predictions, names)
overall = add_changes(overall, fold_city)

# a group helps if it lowers citywide MAE in at least 3 of 4 folds and does not make station MAE worse
helped = [g for g in GROUPS
          if overall.loc[f"+ special days + {g}", "folds_better_city"] >= 3
          and overall.loc[f"+ special days + {g}", "station_MAE"] <= overall.loc[BASE, "station_MAE"]]

# the groups that helped, together
if len(helped) > 1:
    name = "+ special days + " + " + ".join(helped)
    cols = [c for g in helped for c in GROUPS[g]]
    predictions[name] = run_cv({name: gbm(CURRENT + SPECIAL + cols)})[name].to_numpy()
    names.append(name)
    overall, fold_city = scores(predictions, names)
    overall = add_changes(overall, fold_city)

# the UW calendar should matter most next to campus
uw = predictions[predictions[STATION_COL].isin(["SFD 17", "SFD 22"])]
uw_mae = uw[names[1:]].sub(uw[TARGET], axis=0).abs().groupby(uw[STATION_COL]).mean()

# [x] 7. Print the results; decide with the team before adding the features to step4_features.py
pd.set_option("display.width", 200)
print("\nOverall, 2025 rolling CV (4 folds):")
print(overall.round(3).to_string())
print("\nCitywide MAE per fold:")
print(fold_city.round(2).to_string())
print("\nStation MAE next to UW (SFD 17, SFD 22):")
print(uw_mae.T.round(3).to_string())
print("\nGroups that helped on top of special days "
      "(citywide better in >= 3 of 4 folds, station MAE not worse):", helped or "none")


"""
Result (2025 rolling CV, 4 folds):

version                          station MAE  citywide MAE  citywide folds better
current GBM                      2.409        18.95         -
+ special days                   2.406        18.73         3 of 4 vs current GBM
+ special days + month start     2.407        18.83         1 of 4 vs + special days
+ special days + UW in session   2.410        18.94         1 of 4
+ special days + long weekend    2.411        19.11         1 of 4
+ special days + day length      2.410        19.03         1 of 4
+ special days + SPS break       2.404        18.78         2 of 4

Special days help (most of the gain is in the Oct-Dec fold: 20.82 -> 19.83).
None of the other calendar features adds anything on top: month and weekday
already cover day length and long weekends, and the UW calendar does not
lower the error even at SFD 17 and SFD 22 next to campus. On SPS break
weekdays, citywide calls are about the same as on school days (293.6 vs 296.8
per shift, Feb + Apr 2022-2024): more injuries in parks may be offset by
fewer injuries at school and families leaving town. Only the special days are
worth proposing for step4_features.py.
"""
