"""
Customer Churn Prediction - Inference Script
Usage: python -m churn.predict --input data/telecom_churn.csv [--output reports/predictions.csv]
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from churn.config import MODEL_PATH, PREDICTIONS_PATH

THRESHOLD = 0.5


def load_model(path: Path | None = None) -> dict:
    path = path or MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python -m churn.train` first.")
    return joblib.load(path)


def predict(df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    bundle = load_model()
    missing = set(bundle["features"]) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    # The saved pipeline applies the same encoding and scaling used in training
    probs = bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1]

    out = df.copy()
    out["churn_probability"] = probs.round(4)
    out["churn_prediction"] = (probs >= threshold).astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers for churn risk.")
    parser.add_argument("--input", required=True, type=Path, help="Path to input CSV")
    parser.add_argument("--output", type=Path, default=PREDICTIONS_PATH,
                        help="Where to write predictions")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Probability cutoff for a churn prediction")
    args = parser.parse_args()

    result = predict(pd.read_csv(args.input), args.threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Churn predicted for {result['churn_prediction'].sum()} / {len(result)} customers")


if __name__ == "__main__":
    main()
