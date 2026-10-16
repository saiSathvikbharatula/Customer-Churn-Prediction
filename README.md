# Customer Churn Prediction

A machine learning pipeline that predicts telecom customer churn using classification models.

## Project Structure
```
churn_prediction/
├── data/
│   └── telecom_churn.csv        # Dataset (5,000 customers)
├── models/
│   └── best_model.joblib        # Saved preprocessing + model pipeline
├── reports/
│   ├── figures/
│   │   └── eda_charts.png       # EDA visualizations
│   ├── metrics.json             # Model comparison results
│   └── predictions.csv          # Prediction output
├── src/churn/
│   ├── config.py                # Shared paths and column definitions
│   ├── data.py                  # Synthetic dataset generator
│   ├── eda.py                   # Exploratory data analysis charts
│   ├── pipeline.py              # Preprocessing and candidate models
│   ├── train.py                 # Model training and selection
│   └── predict.py               # Inference / scoring
├── tests/
│   └── test_pipeline.py         # Smoke tests
├── pyproject.toml
├── requirements.txt             # Pinned, tested versions
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Run
```bash
# 1. Generate dataset (already included)
python -m churn.data

# 2. EDA
python -m churn.eda

# 3. Train models
python -m churn.train

# 4. Score customers
python -m churn.predict --input data/telecom_churn.csv
# Options: --output <path>  --threshold <0-1>
```

## Tests
```bash
pytest
```

## Approach
- **Preprocessing:** numeric features are standardized and categorical features are one-hot encoded inside a scikit-learn `Pipeline`, so training and inference always use the same transformations.
- **Validation:** stratified 80/20 train/test split plus 5-fold cross-validation on the training set.
- **Class imbalance:** about 19% of customers churn, so Logistic Regression and Random Forest use balanced class weights.
- **Model selection:** the model with the highest test ROC-AUC is saved.

## Models Compared
- Logistic Regression
- Random Forest
- Gradient Boosting

## Features
- Tenure, Monthly Charges, Total Charges
- Contract Type, Internet Service, Payment Method
- Support Calls, Number of Products
- Online Security, Tech Support, Paperless Billing
- Senior Citizen, Partner, Dependents

## Results
Held-out test set (1,000 customers):

| Model | CV ROC-AUC | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.829 | 0.763 | 0.550 | **0.844** |
| Random Forest | 0.808 | 0.822 | 0.564 | 0.839 |
| Gradient Boosting | 0.813 | 0.855 | 0.465 | 0.843 |

Logistic Regression was selected. It catches 78% of churners (recall) at 43% precision.
The strongest predictors are contract type, tenure, support calls and online security.

> The dataset is synthetic. See `src/churn/data.py` for the churn logic it encodes.

## Tech Stack
Python · scikit-learn · pandas · matplotlib
