from __future__ import annotations

from typing import Iterable

import pandas as pd


DAY_COLUMNS: list[str] = [
    "1st",
    "2nd",
    "3rd",
    "4th",
    "5th",
    "6th",
    "7th",
    "8th",
    "9th",
    "10th",
    "11th",
    "12th",
    "13th",
    "14th",
    "15th",
    "16th",
    "17th",
    "18th",
    "19th",
    "20th",
    "21st",
    "22nd",
    "23rd",
    "24th",
    "25th",
    "26th",
    "27th",
    "28th",
    "29th",
    "30th",
]


def season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Summer"
    if month in [6, 7, 8, 9]:
        return "Monsoon"
    return "PostMonsoon"


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small set of useful features for month-level rainfall prediction."""

    engineered = df.copy()
    engineered["month"] = pd.to_numeric(engineered["month"], errors="coerce").fillna(0).astype(int)
    engineered["Season"] = engineered["month"].apply(season)
    engineered["TotalRainfall"] = engineered[DAY_COLUMNS].sum(axis=1)
    engineered["AverageRainfall"] = engineered[DAY_COLUMNS].mean(axis=1)
    return engineered


def get_day_columns(columns: Iterable[str]) -> list[str]:
    return [column for column in DAY_COLUMNS if column in columns]
