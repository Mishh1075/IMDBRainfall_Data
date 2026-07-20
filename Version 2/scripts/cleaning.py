import pandas as pd

# Load the dataset
df = pd.read_csv(
    "data/raw/Indian Rainfall Dataset District-wise Daily Measurements.csv",
    sep=";"
)

print("=" * 50)
print("First 5 Rows")
print(df.head())

print("\n" + "=" * 50)
print("Dataset Shape:")
print(df.shape)

print("\n" + "=" * 50)
print("Column Names:")
print(df.columns.tolist())

print("\n" + "=" * 50)
print("Data Types:")
print(df.dtypes)

print("\n" + "=" * 50)
print("Missing Values:")
print(df.isnull().sum())

import os

os.makedirs("data/cleaned", exist_ok=True)

df.to_csv(
    "data/cleaned/cleaned_weather.csv",
    index=False
)

print("Cleaned dataset saved successfully!")