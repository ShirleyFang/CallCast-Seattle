"""Train the first-pass Poisson GBM and export artifacts for Ryan's Streamlit app.

Run from the repository root:
    uv run python scripts/train_model.py

Outputs:
    models/poisson_gbm.joblib
    model_metadata.json

This script deliberately does NOT evaluate on the held-out 2026 test set.
"""
import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from cv_utils import DATE_COL, STATION_COL, TARGET, build_features, load_dataset

ROOT = Path(__file__).resolve().parents[1]  # file belongs in scripts/
TRAIN_END = "2026-01-01"  # exclusive: train on 2022–2025
EXCLUDE = ["year"]
MODEL_PATH = ROOT / "models" / "poisson_gbm.joblib"
METADATA_PATH = ROOT / "model_metadata.json"


def main():
    df = load_dataset(path=ROOT / "data" / "modeling_dataset.csv")
    train = df[df[DATE_COL] < TRAIN_END].copy()
    if train.empty:
        raise ValueError("No training rows before TRAIN_END")

    # Fit preprocessing schema using training rows only. One-hot categories/column
    # order are stored so inference can align a single station or an entire date.
    X_train, numeric = build_features(train, extra_exclude=EXCLUDE)
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=200, random_state=0
    )
    model.fit(X_train, train[TARGET])

    usual = train.groupby([STATION_COL, "weekday"])[TARGET].mean().to_dict()
    bundle = {
        "model": model,
        "columns": X_train.columns.tolist(),
        "numeric": list(numeric),
        "train_end": TRAIN_END,
        "usual": usual,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    # Ryan's app_data.py checks these *raw* CSV columns before calling predict_shift.
    # Keep Station, because predict_shift needs it for one-hot encoding and output.
    raw_feature_columns = [STATION_COL, "weekday", *numeric]
    if len(raw_feature_columns) != len(set(raw_feature_columns)):
        raise ValueError("Duplicate raw feature columns")
    metadata = {
        "model_name": "Poisson GBM (first pass)",
        "trained_through": train[DATE_COL].max().date().isoformat(),
        "feature_columns": raw_feature_columns,
        "accepts_missing": True,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Trained on {len(train):,} rows through {metadata['trained_through']}")
    print(f"Model features: {len(X_train.columns)}; raw inputs: {len(raw_feature_columns)}")
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {METADATA_PATH}")


if __name__ == "__main__":
    main()
