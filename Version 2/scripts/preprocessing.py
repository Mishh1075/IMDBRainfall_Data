import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/cleaned/cleaned_weather.csv")

print(df.head())

print("Missing Values:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

categorical_cols = df.select_dtypes(include="object").columns

df[categorical_cols] = df[categorical_cols].fillna("Unknown")

print(df.dtypes)

rainfall_cols = [
    "1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th",
    "11th","12th","13th","14th","15th","16th","17th","18th","19th",
    "20th","21st","22nd","23rd","24th","25th","26th","27th","28th",
    "29th","30th","31st"
]

scaler = MinMaxScaler()

df[rainfall_cols] = scaler.fit_transform(df[rainfall_cols])

print(df[rainfall_cols].head())

# Save the preprocessed dataset
df.to_csv("data/processed/preprocessed_weather.csv", index=False)

print("\nPreprocessed dataset saved successfully!")
print("Location: data/processed/preprocessed_weather.csv")

# Display dataset information
print("\nFinal Dataset Shape:", df.shape)
print("\nFirst 5 rows of preprocessed dataset:")
print(df.head())