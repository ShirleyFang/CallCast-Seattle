"""Step 6: Rolling-CV evaluation of baselines and first-pass models.

Models: training mean, station mean, station x weekday mean (baselines),
Poisson gradient boosting and random forest (first-pass models, no tuning).
The 2026 test set is never used here.

Run from the repository root:
    uv run python scripts/evaluate_models.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from cv_utils import DATE_COL, STATION_COL, TARGET, build_features, iter_folds, load_dataset, metrics

MODELS = ["training mean", "station mean", "station x weekday mean", "Poisson GBM", "Random forest"]

df = load_dataset()
X, numeric = build_features(df, extra_exclude=["year"])
print(f"{len(X)} rows, {X.shape[1]} features")
print("numeric feature columns used:", numeric)


def fit_predict(name, tr, va):
    """Return predictions for the validation rows of one fold."""
    overall = tr[TARGET].mean()
    if name == "training mean":
        return np.full(len(va), overall)
    if name == "station mean":
        means = tr.groupby(STATION_COL)[TARGET].mean()
        return va[STATION_COL].map(means).fillna(overall).to_numpy()
    if name == "station x weekday mean":
        means = tr.groupby([STATION_COL, "weekday"])[TARGET].mean()
        return np.array([means.get(k, overall) for k in zip(va[STATION_COL], va["weekday"])])
    xtr, xva = X.loc[tr.index], X.loc[va.index]
    if name == "Poisson GBM":  # handles missing values natively
        model = HistGradientBoostingRegressor(loss="poisson", max_iter=200, random_state=0)
    else:  # random forest: fill missing lag values with -1
        model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=0)
        xtr, xva = xtr.fillna(-1), xva.fillna(-1)
    return model.fit(xtr, tr[TARGET]).predict(xva)


rows = []
for start, tr, va in iter_folds(df):
    for name in MODELS:
        rows.append({"fold": start, "model": name, **metrics(va, fit_predict(name, tr, va))})

results = pd.DataFrame(rows)
print("\nPer fold:")
print(results.round(2).to_string(index=False))
print("\nMean over the 4 folds:")
print(results.groupby("model", sort=False)[["station_MAE", "station_RMSE", "citywide_MAE"]].mean().round(2))
