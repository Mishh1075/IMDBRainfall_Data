from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Indian Rainfall Dataset District-wise Daily Measurements.csv"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the rainfall dataset using the expected semicolon separator."""

    return pd.read_csv(path, sep=";")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the minimal cleaning steps used in the project."""

    cleaned = df.drop_duplicates().copy()
    cleaned = cleaned.fillna(0)
    return cleaned


def split_features_target(df: pd.DataFrame, target_column: str = "31st"):
    """Split the dataset into feature matrix and target vector."""

    features = df.drop(columns=[target_column])
    target = df[target_column]
    return features, target
