"""Small, reproducible 2025 rolling-CV grid search for CallCast-Seattle.

Copy this file to scripts/grid_search_model.py, then run from the repo root:
    uv run python scripts/grid_search_model.py

2026 data is excluded entirely. This script does not overwrite the deployed model.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import ParameterGrid

from cv_utils import DATE_COL, TARGET, build_features, iter_folds, load_dataset, metrics
from feature_sets import SPECIAL_DAYS, add_special_days, baseline_columns


COARSE_GRID = {
    "learning_rate": [0.1],
    "max_leaf_nodes": [7, 15, 31, 63],
    "min_samples_leaf": [20, 50, 100, 200],
    "l2_regularization": [0.0],
}

ITERATION_CAP = 300

COARSE_FILE = Path("results/tuning_coarse.csv")
FINE_FILE = Path("results/tuning_fine.csv")


def fine_grid(group):
    """Search near the best configuration from the coarse search."""

    if not COARSE_FILE.is_file():
        raise FileNotFoundError(
            f"Run --stage coarse first to create {COARSE_FILE}"
        )

    previous = pd.read_csv(COARSE_FILE)
    previous = previous.loc[previous["features"] == group]

    if previous.empty:
        raise ValueError(f"No coarse results for {group}")

    best = previous.sort_values(
        ["citywide_MAE", "station_MAE"]
    ).iloc[0]

    leaves = int(best["max_leaf_nodes"])
    minimum = int(best["min_samples_leaf"])

    # Fine-tune max_leaf_nodes
    if leaves <= 7:
        nearby_leaves = [5, 7, 9, 11, 13]
    elif leaves <= 15:
        nearby_leaves = [7, 9, 11, 13, 15]
    elif leaves <= 31:
        nearby_leaves = [15, 19, 23, 27, 31]
    else:
        nearby_leaves = [31, 39, 47, 55, 63]

    # Fine-tune min_samples_leaf
    nearby_minimum = sorted({
        max(10, minimum // 2),
        minimum,
        minimum + minimum // 2
    })

    print(
        f"Coarse winner: leaves={leaves}, "
        f"min_samples_leaf={minimum}"
    )

    return {
        "max_leaf_nodes": nearby_leaves,
        "min_samples_leaf": nearby_minimum,
        "learning_rate": [0.05, 0.1],
        "l2_regularization": [0.0],
    }


def evaluate_config(X, columns, folds, params, seeds):
    scores = []
    iterations = []

    for seed in seeds:
        for start, train, valid in folds:

            model = HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=ITERATION_CAP,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=seed,
                **params,
            )

            model.fit(
                X.loc[train.index, columns],
                train[TARGET]
            )

            pred = model.predict(X.loc[valid.index, columns])

            scores.append({
                "seed": seed,
                "fold": start,
                **metrics(valid, pred)
            })

            iterations.append(model.n_iter_)

    scores = pd.DataFrame(scores)

    result = scores[
        ["station_MAE", "station_RMSE", "citywide_MAE"]
    ].mean().to_dict()

    # Citywide MAE for each validation fold
    for fold, fold_rows in scores.groupby("fold", sort=False):
        result[f"citywide_{fold}"] = fold_rows["citywide_MAE"].mean()

    # Variation across random seeds
    per_seed = scores.groupby("seed")["citywide_MAE"].mean()

    sd = per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan

    result["citywide_sd_seeds"] = sd
    result["citywide_se_seeds"] = (
        sd / np.sqrt(len(per_seed))
        if len(per_seed) > 1 else np.nan
    )

    # Actual number of boosting iterations
    result["n_iter_mean"] = np.mean(iterations)
    result["n_iter_max"] = max(iterations)

    result["hit_iteration_cap_pct"] = (
        100 * np.mean(np.array(iterations) >= ITERATION_CAP)
    )

    return result


def choose_simple(table):
    """Prefer simpler models when scores are within one standard error."""

    # Find the configuration with the lowest citywide MAE
    best = table.loc[table["citywide_MAE"].idxmin()]

    tolerance = best["citywide_se_seeds"]

    if pd.isna(tolerance):
        tolerance = 0.0

    # Configurations close enough to the best score
    candidates = table.loc[
        table["citywide_MAE"] <= best["citywide_MAE"] + tolerance
    ]

    # Prefer fewer leaves and more samples per leaf
    simple = candidates.sort_values(
        ["max_leaf_nodes", "min_samples_leaf", "citywide_MAE"],
        ascending=[True, False, True]
    ).iloc[0]

    return best, simple, tolerance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["coarse", "fine"], default="coarse"
    )
    parser.add_argument(
        "--features",
        choices=["special", "baseline", "both"],
        default="special"
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (e.g. 0,1,2)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the grid size without fitting"
    )

    args = parser.parse_args()

    # 1. Choose random seeds
    seeds = (
        [int(v.strip()) for v in args.seeds.split(",")]
        if args.seeds
        else ([0] if args.stage == "coarse" else [0, 1, 2])
    )

    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")

    # 2. Choose feature groups
    groups = (
        ["baseline", "baseline + special days"]
        if args.features == "both"
        else [
            "baseline"
            if args.features == "baseline"
            else "baseline + special days"
        ]
    )

    # 3. Choose coarse or fine parameter grid
    grids = {
        group: (
            COARSE_GRID
            if args.stage == "coarse"
            else fine_grid(group)
        )
        for group in groups
    }

    total = sum(
        len(list(ParameterGrid(grid)))
        for grid in grids.values()
    )

    print(
        f"{args.stage}: {total} configurations × "
        f"{len(seeds)} seeds × 4 folds = "
        f"{total * len(seeds) * 4} fits "
        f"(seeds={seeds})",
        flush=True
    )

    if args.dry_run:
        return

    # 4. Load data and exclude the 2026 test set
    data = load_dataset()

    data = data.loc[
        data[DATE_COL] < pd.Timestamp("2026-01-01")
    ].copy()

    data, _ = add_special_days(data)

    # 5. Build model features
    X, _ = build_features(data, extra_exclude=["year"])

    base = baseline_columns(X)

    feature_groups = {
        "baseline": base,
        "baseline + special days": base + SPECIAL_DAYS,
    }

    # 6. Prepare the four rolling CV folds
    folds = list(iter_folds(data))

    if len(folds) != 4 or any(
        train.empty or val.empty
        for _, train, val in folds
    ):
        raise ValueError(
            "Expected four nonempty 2025 rolling-CV folds"
        )

    # 7. Run grid search
    rows = []

    for group, grid in grids.items():
        for params in ParameterGrid(grid):

            scores = evaluate_config(
                X,
                feature_groups[group],
                folds,
                params,
                seeds
            )

            rows.append({
                "features": group,
                **params,
                "seeds": len(seeds),
                **scores
            })

            print(
                f"{len(rows):3d}/{total} "
                f"{group:24s} {params} "
                f"→ citywide={scores['citywide_MAE']:.3f} "
                f"station={scores['station_MAE']:.3f} "
                f"mean n_iter_={scores['n_iter_mean']:.1f}",
                flush=True
            )

    # 8. Save results
    table = pd.DataFrame(rows).sort_values(
        ["features", "citywide_MAE", "station_MAE"]
    )

    output = (
        COARSE_FILE
        if args.stage == "coarse"
        else FINE_FILE
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    # 9. Select the best and simplest configurations
    print(
        "\nBest citywide MAE and simpler model "
        "within one seed-SE:"
    )

    for group in groups:
        best, simple, tolerance = choose_simple(
            table.loc[table["features"] == group]
        )

        for label, row in [
            ("best", best),
            ("simple", simple)
        ]:
            print(
                f"  {group} {label}: "
                f"citywide={row['citywide_MAE']:.3f}, "
                f"station={row['station_MAE']:.3f}, "
                f"leaves={int(row['max_leaf_nodes'])}, "
                f"min_samples_leaf={int(row['min_samples_leaf'])}, "
                f"learning_rate={row['learning_rate']:.3g}, "
                f"n_iter_={row['n_iter_mean']:.1f} "
                f"(one-SE={tolerance:.3f})"
            )

    print(f"\nFull results saved to {output}")
    print("2026 test untouched; production model unchanged.")

if __name__ == "__main__":
    main()
