# What it does
# Compares models station by station, to answer: does a different model work
# better at different stations (neighborhoods)? Uses the same rolling CV as
# scripts/evaluate_models.py (4 quarterly folds over 2025). The 2026 test set
# is not used here.
#
# Run it from the repository root: uv run python scripts/evaluate_by_station.py
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson
from cv_utils import load_dataset, build_features, iter_folds, STATION_COL, DATE_COL, TARGET
# TODO
# [x] 1. Load data/modeling_dataset.csv with cv_utils.load_dataset; get the list of 34 stations

# scripts run from the repository root, so the default path data/modeling_dataset.csv is used
df = load_dataset()
stations = sorted(df[STATION_COL].unique())

print(f"{len(df)} rows, {len(stations)} stations, "
      f"{df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}")
print("mean calls per shift by station:")
print(df.groupby(STATION_COL)[TARGET].mean().sort_values(ascending=False).round(1).to_string())

# Baselines:       station x weekday mean, lag3, avg28
def station_weekday_mean(train, validation):
    """Mean calls for each station on each weekday, computed on training rows only."""
    means = train.groupby(["Station", "weekday"])["calls"].mean()
    overall = train["calls"].mean()
    return np.array([means.get(k, overall) for k in zip(validation["Station"], validation["weekday"])])


def station_mean(train, validation):
    """Fallback for missing lags: each station's training mean."""
    return validation[STATION_COL].map(train.groupby(STATION_COL)[TARGET].mean()).to_numpy()


def lag3(train, validation):
    """Calls 3 shifts earlier (the latest known shift); station mean when unknown."""
    return validation["lag3"].fillna(pd.Series(station_mean(train, validation), index=validation.index)).to_numpy()


def avg28(train, validation):
    """Mean of the 28 shifts before lag3; station mean when unknown."""
    return validation["avg28"].fillna(pd.Series(station_mean(train, validation), index=validation.index)).to_numpy()


# [x] 2. Models to compare
#        Global models:   Poisson GBM (current model), random forest
#                         (one model trained on all stations)
#        Per-station:     Poisson GBM and a Poisson regression (GLM), one model per station,
#                         trained on that station's rows only

# same features as scripts/evaluate_models.py: numeric columns plus station and weekday dummies, no year
X, numeric = build_features(df, extra_exclude=["year"])

def global_gbm(train, validation):
    """The current model: one Poisson GBM for all stations (same settings as evaluate_models.py)."""
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=200, random_state=0)
    return model.fit(X.loc[train.index], train[TARGET]).predict(X.loc[validation.index])


def global_rf(train, validation):
    """One random forest for all stations; missing lags filled with -1 (as in evaluate_models.py)."""
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=0)
    return model.fit(X.loc[train.index].fillna(-1), train[TARGET]).predict(X.loc[validation.index].fillna(-1))


def per_station(make_model, fill_missing, features):
    """Fit one model per station on that station's rows only, then predict its validation rows."""
    def fit_predict(train, validation):
        pred = pd.Series(np.nan, index=validation.index)
        for station, va in validation.groupby(STATION_COL):
            tr = train[train[STATION_COL] == station]
            xtr, xva = features.loc[tr.index], features.loc[va.index]
            if fill_missing:   # the GLM cannot use missing values: fill with this station's training mean
                fill = tr[TARGET].mean()
                xtr, xva = xtr.fillna(fill), xva.fillna(fill)
            pred[va.index] = make_model().fit(xtr, tr[TARGET]).predict(xva)
        return pred.to_numpy()
    return fit_predict


# inside one station the station dummies are constant, so the per-station models use the rest
X_station = X[[c for c in X.columns if not c.startswith("st_")]]
# trees can split month 1-12 into seasons, but a linear model would fit one slope from January
# to December, so the GLM gets month as dummies (like weekday)
X_station_glm = pd.concat([X_station.drop(columns="month"),
                           pd.get_dummies(df["month"], prefix="m", dtype=float)], axis=1)

per_station_gbm = per_station(lambda: HistGradientBoostingRegressor(loss="poisson", max_iter=200, random_state=0),
                              fill_missing=False, features=X_station)
per_station_glm = per_station(lambda: make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=1000)),
                              fill_missing=True, features=X_station_glm)

MODELS = {
    "station x weekday": station_weekday_mean,
    "lag3": lag3,
    "avg28": avg28,
    "global GBM": global_gbm,
    "global RF": global_rf,
    "per-station GBM": per_station_gbm,
    "per-station GLM": per_station_glm,
}
BASELINE = "station x weekday"
# [x] 3. Rolling CV: for each fold (cv_utils.iter_folds), fit every model on the training
#        shifts and predict the validation quarter; keep all predictions in one table

# one row per station-shift in the 2025 validation quarters, one column per model
parts = []
for fold_start, train, validation in iter_folds(df):
    fold = validation[[DATE_COL, STATION_COL, TARGET]].assign(fold=fold_start)
    for name, fit_predict in MODELS.items():
        fold[name] = fit_predict(train, validation)
    parts.append(fold)
    print(f"fold {fold_start}: trained on {len(train)} rows, validated on {len(validation)} rows")
predictions = pd.concat(parts)

# every validation row should have a prediction from every model
assert predictions[list(MODELS)].notna().all().all()
print(f"\n{len(predictions)} validation rows ({predictions[DATE_COL].nunique()} shifts x {predictions[STATION_COL].nunique()} stations)")
# [x] 4. Per-station scores
#        - MAE for each model
#        - gain over the station x weekday baseline (%), so large and small stations compare fairly
#        - Poisson noise floor (expected MAE if calls were pure Poisson noise) and the gap to it
#        - best model per station
MODEL_NAMES = list(MODELS)
abs_error = predictions[MODEL_NAMES].sub(predictions[TARGET], axis=0).abs()

by_station = abs_error.groupby(predictions[STATION_COL]).mean()
by_station.insert(0, "mean_calls", predictions.groupby(STATION_COL)[TARGET].mean())

# Poisson noise floor: if calls were Poisson around the GBM forecast lam, the expected
# absolute error is E|Y - lam| = 2 * lam * P(Y = floor(lam)). No model can beat this on average.
lam = predictions["global GBM"].to_numpy()
floor = pd.Series(2 * lam * poisson.pmf(np.floor(lam), lam), index=predictions.index)
by_station["noise_floor"] = floor.groupby(predictions[STATION_COL]).mean()

by_station["best_model"] = by_station[MODEL_NAMES].idxmin(axis=1)
by_station["gbm_gain_vs_baseline_pct"] = 100 * (1 - by_station["global GBM"] / by_station[BASELINE])
by_station["gbm_gap_to_floor_pct"] = 100 * (by_station["global GBM"] / by_station["noise_floor"] - 1)
# how much the best model beats the GBM at this station (0 when the GBM is the best)
by_station["best_vs_gbm_pct"] = 100 * (1 - by_station[MODEL_NAMES].min(axis=1) / by_station["global GBM"])


def citywide_mae(rows, model):
    """Sum the station forecasts per shift, then take the MAE against the citywide total."""
    city = rows.groupby(DATE_COL)[[TARGET, model]].sum()
    return (city[model] - city[TARGET]).abs().mean()


overall = pd.DataFrame({"station_MAE": abs_error.mean(),
                        "citywide_MAE": [citywide_mae(predictions, m) for m in MODEL_NAMES]})

# [x] 5. Selection check: pick the best model per station on folds 1-3, then score fold 4.
#        Compare with "GBM for every station" (station MAE and citywide MAE)
# If per-station selection really helps, the models picked on folds 1-3 should still win on fold 4.
last_fold = predictions["fold"] == predictions["fold"].max()
picked = (abs_error[~last_fold].groupby(predictions.loc[~last_fold, STATION_COL]).mean().idxmin(axis=1))

fold4 = predictions[last_fold].copy()
fold4["picked per station"] = [row[picked[row[STATION_COL]]] for _, row in fold4.iterrows()]
selection = pd.DataFrame({
    model: {"station_MAE": (fold4[model] - fold4[TARGET]).abs().mean(),
            "citywide_MAE": citywide_mae(fold4, model)}
    for model in ["global GBM", "picked per station"]}).T

# [x] 6. Print the per-station table and the selection check
pd.set_option("display.width", 200)
print("\nOverall, 2025 rolling CV (4 folds):")
print(overall.round(3).to_string())
print("\nStation MAE by model (busiest station first):")
print(by_station.sort_values("mean_calls", ascending=False).round(2).to_string())
print("\nBest model count:", by_station["best_model"].value_counts().to_dict())
print(f"Stations where the best model beats the GBM by more than 2%: "
      f"{(by_station['best_vs_gbm_pct'] > 2).sum()} of {len(by_station)}")
print("\nSelection check: models picked per station on folds 1-3, scored on fold 4:")
print(selection.round(3).to_string())
print("Picked models:", picked.value_counts().to_dict())

# [x] 7. Save data/station_errors.csv (one row per station) for the app's analysis tab
station_info = pd.read_csv("data/sfd_stations.csv").set_index("station")
out = by_station.join(station_info[["address", "lat", "lon"]])
out.index.name = STATION_COL
out.round(3).reset_index().to_csv("data/station_errors.csv", index=False)
print(f"\nsaved data/station_errors.csv ({len(out)} stations)")

# [x] 8. Add this step to the pipeline table in README.md


"""
Result: 

models	Station MAE	Citywide MAE
global GBM	2.409	18.95
per-station GLM	2.431	19.98
station * weekday	2.448	20.71
global RF	2.457	20.03
per-station GBM	2.718	21.40
"""