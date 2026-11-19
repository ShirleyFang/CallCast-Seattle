"""Compare the existing Poisson GBM with/without special-day features.

Run from the CallCast-Seattle repository root after placing this file in scripts/:
    uv run python scripts/tune_model.py

Uses only the four 2025 rolling-CV folds. The 2026 test set is not read for
model selection. No production model is overwritten.
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from cv_utils import DATE_COL, TARGET, build_features, iter_folds, load_dataset, metrics
from feature_sets import SPECIAL_DAYS, add_special_days, baseline_columns


def main():
    df = load_dataset()
    df = df.loc[df[DATE_COL] < pd.Timestamp("2026-01-01")].copy()
    df, _ = add_special_days(df)

    X, _ = build_features(df, extra_exclude=["year"])
    baseline = baseline_columns(X)
    feature_sets = {
        "baseline": baseline,
        "baseline + special days": baseline + SPECIAL_DAYS,
    }

    results = []
    for label, columns in feature_sets.items():
        fold_scores = []
        print(f"\n{label} ({len(columns)} features)", flush=True)

        for start, train, val in iter_folds(df):
            model = HistGradientBoostingRegressor(
                loss="poisson", max_iter=200, random_state=0
            )
            model.fit(X.loc[train.index, columns], train[TARGET])
            pred = model.predict(X.loc[val.index, columns])
            score = metrics(val, pred)
            fold_scores.append(score)
            print(
                f"  {start}: station MAE={score['station_MAE']:.3f}, "
                f"citywide MAE={score['citywide_MAE']:.3f}",
                flush=True,
            )

        mean_score = pd.DataFrame(fold_scores).mean().to_dict()
        results.append({"features": label, **mean_score})

    print("\nMean over four 2025 folds (lower is better):")
    print(pd.DataFrame(results).round(3).to_string(index=False))
    print("\nThe 2026 test set was not evaluated; no model was saved.")


if __name__ == "__main__":
    main()
