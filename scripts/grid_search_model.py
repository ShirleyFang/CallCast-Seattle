"""Small, reproducible 2025 rolling-CV grid search for CallCast-Seattle.

Copy this file to scripts/grid_search_model.py, then run from the repo root:
    uv run python scripts/grid_search_model.py

2026 data is excluded entirely. This script does not overwrite the deployed model.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterGrid

from cv_utils import DATE_COL, TARGET, build_features, iter_folds, load_dataset, metrics
from feature_sets import SPECIAL_DAYS, add_special_days, baseline_columns


GRID = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [100, 200],
    "max_leaf_nodes": [15, 31],
    "l2_regularization": [0.0, 1.0],
}
OUTPUT = Path("results/tuning_cv.csv")


def main():
    data = load_dataset()
    # Preserve the untouched final holdout. Neither 2026 features nor labels
    # enter any fit or model-selection computation.
    data = data.loc[data[DATE_COL] < pd.Timestamp("2026-01-01")].copy()
    data, _ = add_special_days(data)

    X, _ = build_features(data, extra_exclude=["year"])
    base = baseline_columns(X)
    feature_groups = {
        "baseline": base,
        "baseline + special days": base + SPECIAL_DAYS,
    }

    folds = list(iter_folds(data))
    combinations = list(ParameterGrid(GRID))
    total = len(feature_groups) * len(combinations)
    print(f"Running {total} configurations × {len(folds)} folds ({total * len(folds)} fits)", flush=True)

    rows = []
    for group, columns in feature_groups.items():
        for params in combinations:
            fold_scores = []
            result = {"features": group, **params}

            for start, train, val in folds:
                model = HistGradientBoostingRegressor(
                    loss="poisson", random_state=0, **params
                )
                model.fit(X.loc[train.index, columns], train[TARGET])
                prediction = model.predict(X.loc[val.index, columns])
                score = metrics(val, prediction)
                fold_scores.append(score)
                result[f"citywide_{start}"] = score["citywide_MAE"]

            result.update(pd.DataFrame(fold_scores).mean().to_dict())
            rows.append(result)
            print(
                f"{len(rows):2d}/{total} {group:24s} {params} "
                f"→ citywide={result['citywide_MAE']:.3f} "
                f"station={result['station_MAE']:.3f}",
                flush=True,
            )

    table = pd.DataFrame(rows).sort_values(
        ["citywide_MAE", "station_MAE"], ascending=True
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)

    print("\nBest 5 configurations across both feature groups:")
    cols = ["features", *GRID, "station_MAE", "station_RMSE", "citywide_MAE"]
    print(table[cols].head(5).round(3).to_string(index=False))

    print("\nBest configuration within each feature group:")
    winners = table.drop_duplicates("features")
    print(winners[cols].round(3).to_string(index=False))

    print(f"\nFull results saved to {OUTPUT}")
    print("2026 test untouched; production model unchanged.")


if __name__ == "__main__":
    main()
