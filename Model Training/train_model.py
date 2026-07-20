from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import add_feature_engineering
from preprocess import clean_data, load_data, split_features_target


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "rainfall_model.pkl"
METRICS_PATH = MODELS_DIR / "rainfall_model_metrics.json"


def build_preprocessor(feature_frame):
    categorical_features = ["state", "district", "Season"]
    numeric_features = [
        column
        for column in feature_frame.columns
        if column not in categorical_features
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2_score(y_test, predictions),
    }


def main():
    df = load_data()
    df = clean_data(df)
    df = add_feature_engineering(df)

    features, target = split_features_target(df, target_column="31st")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    preprocessor = build_preprocessor(x_train)

    candidate_models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    best_name = None
    best_mae = float("inf")
    best_model = None

    for name, estimator in candidate_models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test)
        results[name] = metrics
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_name = name
            best_model = pipeline

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_model": best_name,
                "results": results,
            },
            handle,
            indent=2,
        )

    print(f"Saved best model: {best_name}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
