"""Generate a synthetic telecom churn dataset."""

import numpy as np
import pandas as pd

from churn.config import DATA_PATH, RANDOM_STATE

N_CUSTOMERS = 5000


def generate(n: int = N_CUSTOMERS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.uniform(20, 120, n), 2)
    df = pd.DataFrame({
        "customer_id": [f"CUST{i:05d}" for i in range(1, n + 1)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        # Total charges accumulate over tenure, with some month-to-month variation
        "total_charges": np.round(tenure * monthly * rng.uniform(0.9, 1.1, n), 2),
        "num_products": rng.integers(1, 6, n),
        "support_calls": rng.poisson(2.5, n).clip(0, 10),
        "contract_type": rng.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.5, 0.3, 0.2]),
        "payment_method": rng.choice(["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"], n),
        "internet_service": rng.choice(["DSL", "Fiber Optic", "No"], n, p=[0.4, 0.4, 0.2]),
        "online_security": rng.choice(["Yes", "No"], n),
        "tech_support": rng.choice(["Yes", "No"], n),
        "paperless_billing": rng.choice(["Yes", "No"], n),
        "senior_citizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "dependents": rng.choice(["Yes", "No"], n),
        "partner": rng.choice(["Yes", "No"], n),
    })

    # Churn log-odds: month-to-month, short tenure, frequent support calls,
    # fiber, high bills and electronic checks increase risk; add-ons reduce it.
    contract_effect = df["contract_type"].map({"Month-to-Month": 1.2, "One Year": -0.3, "Two Year": -1.5})
    logit = (
        -1.6
        + contract_effect
        - 0.04 * df["tenure_months"]
        + 0.35 * df["support_calls"]
        + 0.6 * (df["internet_service"] == "Fiber Optic")
        + 0.015 * (df["monthly_charges"] - 70)
        + 0.5 * (df["payment_method"] == "Electronic Check")
        - 0.5 * (df["online_security"] == "Yes")
        - 0.4 * (df["tech_support"] == "Yes")
        + 0.4 * df["senior_citizen"]
        - 0.2 * (df["partner"] == "Yes")
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    df["churn"] = (rng.random(n) < churn_prob).astype(int)
    return df


def main() -> None:
    data = generate()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH}: {len(data)} rows, churn rate = {data['churn'].mean():.2%}")


if __name__ == "__main__":
    main()
