from __future__ import annotations

import json
from pathlib import Path

import joblib

from feature_engineering import add_feature_engineering
from preprocess import clean_data, load_data, split_features_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "rainfall_model.pkl"


def main():
    model = joblib.load(MODEL_PATH)
    df = add_feature_engineering(clean_data(load_data()))
    features, target = split_features_target(df, target_column="31st")

    sample_predictions = model.predict(features.head(5))
    output = {
        "sample_predictions": sample_predictions.tolist(),
        "actual_values": target.head(5).tolist(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
