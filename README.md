# Rainfall Prediction

This project predicts rainfall on the 31st day of the month from district-level daily rainfall data.

## Project Layout

- `data/` contains the CSV dataset.
- `notebooks/` contains the analysis and training notebook.
- `src/` contains reusable preprocessing, feature engineering, training, and evaluation code.
- `models/` stores the trained model artifact.

## Approach

The pipeline keeps preprocessing lightweight:

1. Load the dataset and remove duplicates.
2. Fill missing values.
3. Add seasonal and rainfall summary features.
4. Use one-hot encoding for categorical features.
5. Train and compare multiple regression models.
6. Save the best model.

The target is the `31st` day rainfall value.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train_model.py
```

Evaluate a saved model:

```bash
python src/evaluate.py
```
