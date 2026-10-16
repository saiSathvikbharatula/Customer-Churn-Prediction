"""Preprocessing and candidate model definitions."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_COLS, NUMERIC_COLS, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ])


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
