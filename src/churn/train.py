"""
Customer Churn Prediction - Training Pipeline
Author: Sathvik

Compares three classifiers with cross-validation, evaluates them on a held-out
test set, and saves the best one (by ROC-AUC) as a complete preprocessing +
model pipeline.
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from churn.config import DATA_PATH, FEATURES, METRICS_PATH, MODEL_PATH, RANDOM_STATE, TARGET
from churn.pipeline import build_models, build_pipeline


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }


def top_features(pipeline: Pipeline, n: int = 5) -> pd.Series:
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    else:
        scores = abs(model.coef_[0])
    return pd.Series(scores, index=names).sort_values(ascending=False).head(n)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows | Churn rate: {df[TARGET].mean():.2%}\n")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, pipelines = {}, {}
    for name, model in build_models().items():
        pipeline = build_pipeline(model)
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        pipeline.fit(X_train, y_train)

        results[name] = {"cv_roc_auc": round(cv_auc.mean(), 4), **evaluate(pipeline, X_test, y_test)}
        pipelines[name] = pipeline

        m = results[name]
        print(f"-- {name} --")
        print(f"  CV ROC-AUC : {m['cv_roc_auc']} +/- {cv_auc.std():.4f}")
        print(f"  Accuracy   : {m['accuracy']}")
        print(f"  F1 Score   : {m['f1']}")
        print(f"  ROC-AUC    : {m['roc_auc']}\n")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best = pipelines[best_name]
    print(f"Best Model: {best_name} (ROC-AUC: {results[best_name]['roc_auc']})\n")
    print(classification_report(y_test, best.predict(X_test), target_names=["No Churn", "Churn"]))
    print("Top 5 Features:\n", top_features(best).to_string(), "\n")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": best_name, "pipeline": best, "features": FEATURES}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({"best_model": best_name, "results": results}, indent=2))
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
