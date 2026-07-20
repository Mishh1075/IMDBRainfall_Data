# Version 4 - Model Training

This folder contains the training-focused version of the rainfall prediction work.

## Contents

- `train_model.py` trains and compares regression models, then saves the best one.
- `evaluate.py` loads the saved model and prints sample predictions.
- `rainfall_prediction.ipynb` is the notebook version of the training workflow.
- `rainfall_model.pkl` is the saved trained model.
- `rainfall_model_metrics.json` stores the evaluation metrics.
- `requirements.txt` lists the Python dependencies used for this version.

## Target

The model predicts rainfall on the `31st` day using the other available features.

## Run

From this folder, run:

```bash
python3 train_model.py
python3 evaluate.py
cat rainfall_model_metrics.json
```

If you want to retrain after changing hyperparameters, rerun `train_model.py` and then inspect the updated metrics file.
