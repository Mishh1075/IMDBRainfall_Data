from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_engineering import add_feature_engineering
from preprocess import clean_data, load_data, split_features_target


DATA_PATH = PROJECT_ROOT / "data" / "Indian Rainfall Dataset District-wise Daily Measurements.csv"
MODEL_PATH = PROJECT_ROOT / "rainfall_model.pkl"


def main():
    model = joblib.load(MODEL_PATH)
    df = add_feature_engineering(clean_data(load_data(DATA_PATH)))
    features, target = split_features_target(df, target_column="31st")

    sample_predictions = model.predict(features.head(5))
    output = {
        "sample_predictions": sample_predictions.tolist(),
        "actual_values": target.head(5).tolist(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
