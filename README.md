# Autism Prediction

Binary classifier that predicts ASD (`Class/ASD`) from the AQ-10 screening questionnaire and demographic features.

## Data

`train.csv` — one row per respondent.

- `A1_Score`–`A10_Score`: AQ-10 screening question answers (0/1)
- `age`, `gender`, `ethnicity`, `jaundice`, `austim`, `contry_of_res`, `used_app_before`, `relation`: demographics/history
- `result`: AQ-10 total score
- `Class/ASD`: target (1 = ASD, 0 = no ASD)

`ID` and `age_desc` are dropped before training (no predictive signal).

## Approach

See `Autism Prediction.ipynb`:

1. Clean data, label-encode categorical columns (saved to `encoders.pkl`)
2. Train/test split (80/20), SMOTE oversampling on the training set to address class imbalance
3. Compare `DecisionTreeClassifier`, `RandomForestClassifier`, `XGBClassifier` via cross-validation
4. Hyperparameter search (`RandomizedSearchCV`) per model, keep the best
5. Best model: `RandomForestClassifier(bootstrap=False, max_depth=20, n_estimators=50)` — **81.9% test accuracy**
6. Saved to `best_model.pkl`

## Usage

```python
import pickle
import pandas as pd

model = pickle.load(open("best_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# encode categorical columns with the matching fitted LabelEncoder before predict()
model.predict(X)
```

## Requirements

`pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `seaborn`
