"""Shared paths and column definitions for the churn pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "telecom_churn.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.csv"

RANDOM_STATE = 42
ID_COL = "customer_id"
TARGET = "churn"

NUMERIC_COLS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "support_calls",
    "senior_citizen",
]
CATEGORICAL_COLS = [
    "contract_type",
    "payment_method",
    "internet_service",
    "online_security",
    "tech_support",
    "paperless_billing",
    "dependents",
    "partner",
]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS
