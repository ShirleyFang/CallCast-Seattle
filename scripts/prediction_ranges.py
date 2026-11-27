# Measures how far the Poisson GBM's forecasts were from the real calls in 2025 and saves the
# "likely range" the app shows to data/prediction_ranges.csv. The range is the 10th to 90th
# percentile of (actual - forecast), so about 4 in 5 shifts land inside it. It uses the same
# four rolling folds as scripts/evaluate_models.py and never trains or scores on 2026.
# A station gets one range per band of forecast size (a forecast of 40 misses by more than one
# of 5). The citywide total, the 34 forecasts added up per shift, gets its own.
#
# Run it from the repository root: uv run python scripts/prediction_ranges.py

import joblib
import numpy as np
import pandas as pd
from cv_utils import DATE_COL, TARGET, build_features, iter_folds, load_dataset
from feature_sets import add_special_days
from sklearn.base import clone

LOW_Q, HIGH_Q = 0.10, 0.90  # the range holds the middle 80% of misses
# bands of forecast size: up to 5, 5 to 10, 10 to 20, 20 to 40, over 40
BAND_TOPS = [5, 10, 20, 40, 1000]
OUT_PATH = "data/prediction_ranges.csv"


def band_of(forecast):
    return np.searchsorted(BAND_TOPS, forecast, side="left").clip(
        max=len(BAND_TOPS) - 1
    )


def make_table(rows):
    """Range per forecast band and for the city, from rows that have a forecast and a miss."""
    band, out, bottom = band_of(rows["pred"]), [], 0
    for i, top in enumerate(BAND_TOPS):
        miss = rows.loc[band == i, "miss"]
        out.append(
            {
                "level": "station",
                "predicted_from": bottom,
                "predicted_to": top,
                "low": miss.quantile(LOW_Q),
                "high": miss.quantile(HIGH_Q),
                "shifts": len(miss),
            }
        )
        bottom = top
    city = rows.groupby(DATE_COL)[[TARGET, "pred"]].sum()
    city_miss = city[TARGET] - city["pred"]
    out.append(
        {
            "level": "citywide",
            "predicted_from": np.nan,
            "predicted_to": np.nan,
            "low": city_miss.quantile(LOW_Q),
            "high": city_miss.quantile(HIGH_Q),
            "shifts": len(city),
        }
    )
    return pd.DataFrame(out)


def coverage(rows, table):
    """Share of station shifts and of citywide shifts whose real calls fall inside the range."""
    bands = table[table["level"] == "station"]
    band = band_of(rows["pred"])
    inside = (rows["miss"] >= bands["low"].to_numpy()[band]) & (
        rows["miss"] <= bands["high"].to_numpy()[band]
    )
    city = rows.groupby(DATE_COL)[[TARGET, "pred"]].sum()
    city_miss = city[TARGET] - city["pred"]
    c = table[table["level"] == "citywide"].iloc[0]
    return (
        inside.mean(),
        ((city_miss >= c["low"]) & (city_miss <= c["high"])).mean(),
        inside.groupby(band).mean(),
    )


# forecast every 2025 quarter with a model that only saw earlier shifts
df = load_dataset()
# The saved model decides the settings and the inputs, so a retrain never leaves the ranges behind.
bundle = joblib.load("models/poisson_gbm.joblib")
df, _ = add_special_days(df)
X, _ = build_features(df, extra_exclude=["year"])
X = X[bundle["columns"]]
parts = []
for fold, (_start, train, valid) in enumerate(iter_folds(df)):
    # same settings as the saved model, fitted again on the earlier shifts only
    model = clone(bundle["model"])
    model.fit(X.loc[train.index], train[TARGET])
    parts.append(valid.assign(fold=fold, pred=model.predict(X.loc[valid.index])))
rows = pd.concat(parts)
rows["miss"] = rows[TARGET] - rows["pred"]
assert rows[DATE_COL].max() < pd.Timestamp("2026-01-01"), "2026 must not be used"
print(
    f"{len(rows)} station shifts, {rows[DATE_COL].nunique()} shifts, "
    f"{rows[DATE_COL].min().date()} to {rows[DATE_COL].max().date()}"
)

# the table the app uses comes from all four folds
table = make_table(rows)
saved = table.drop(columns="shifts").round({"low": 1, "high": 1})
saved.to_csv(OUT_PATH, index=False)
pd.set_option("display.width", 200)
print("\nRange (actual minus forecast), all four folds:")
print(table.round(2).to_string(index=False))
station_all, city_all, by_band = coverage(rows, saved)
print(
    f"\nCoverage on the same shifts: stations {station_all:.1%}, citywide {city_all:.1%}"
)

# honest check: ranges from folds 1 to 3, counted on fold 4 (shifts the ranges never saw)
past, last = rows[rows["fold"] < 3], rows[rows["fold"] == 3]
station_out, city_out, by_band_out = coverage(
    last, make_table(past).round({"low": 1, "high": 1})
)
print(
    f"Coverage on fold 4 with ranges from folds 1 to 3: stations {station_out:.1%}, citywide {city_out:.1%}"
)
print(
    "by band (fold 4):",
    ", ".join(f"{BAND_TOPS[i]}: {v:.0%}" for i, v in by_band_out.items()),
)
print(f"\nsaved {OUT_PATH}")

# Result with the retuned model, 2025 rolling CV (actual minus forecast, 10th to 90th percentile;
# station shifts in brackets):
#   forecast up to 5: -2.4 to +2.7 (4,898)    5 to 10: -3.5 to +3.8 (3,830)
#   10 to 20: -4.7 to +4.8 (2,587)    20 to 40: -7.3 to +8.2 (917)    over 40: -8.9 to +9.9 (178)
#   citywide: -29.6 to +32.0 (365 shifts)
# Coverage on the shifts the ranges came from: stations 79.7%, citywide 79.7%.
# Out of sample (ranges from folds 1 to 3, counted on fold 4): stations 78.6%, citywide 71.7%.
# The citywide range runs narrow in the last quarter (92 shifts), so read its "about 4 in 5" as
# about 7 in 10. It describes a forecast made 3 shifts ahead.
