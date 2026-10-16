"""
Customer Churn Prediction - EDA & Visualization
Generates charts saved to reports/figures/
"""

import matplotlib

matplotlib.use("Agg")  # render to file without needing a display

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from churn.config import DATA_PATH, FIGURES_DIR, TARGET

STAY_COLOR = "#4CAF50"
CHURN_COLOR = "#F44336"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Customer Churn Analysis", fontsize=16, fontweight="bold")

    # ── Plot 1: Churn Distribution ────────────────────────────────────────────
    churn_counts = df[TARGET].value_counts().sort_index()
    axes[0, 0].pie(
        churn_counts, labels=["No Churn", "Churn"],
        autopct="%1.1f%%", colors=[STAY_COLOR, CHURN_COLOR],
        startangle=90, textprops={"fontsize": 12},
    )
    axes[0, 0].set_title("Overall Churn Rate")

    # ── Plot 2: Churn by Contract Type ────────────────────────────────────────
    ct = df.groupby("contract_type")[TARGET].mean().sort_values(ascending=False)
    bars = axes[0, 1].bar(ct.index, ct.values, color=[CHURN_COLOR, "#FF9800", STAY_COLOR])
    axes[0, 1].set_title("Churn Rate by Contract Type")
    axes[0, 1].set_ylabel("Churn Rate")
    axes[0, 1].set_ylim(0, ct.max() * 1.2)
    for bar, val in zip(bars, ct.values):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.1%}", ha="center", fontsize=10)

    # ── Plot 3: Tenure vs Churn ───────────────────────────────────────────────
    axes[1, 0].hist(df.loc[df[TARGET] == 0, "tenure_months"], bins=30, alpha=0.6,
                    color=STAY_COLOR, label="No Churn")
    axes[1, 0].hist(df.loc[df[TARGET] == 1, "tenure_months"], bins=30, alpha=0.6,
                    color=CHURN_COLOR, label="Churn")
    axes[1, 0].set_title("Tenure Distribution by Churn")
    axes[1, 0].set_xlabel("Tenure (months)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()

    # ── Plot 4: Monthly Charges vs Support Calls ──────────────────────────────
    sample = df.sample(500, random_state=42)
    colors = sample[TARGET].map({0: STAY_COLOR, 1: CHURN_COLOR})
    axes[1, 1].scatter(sample["monthly_charges"], sample["support_calls"],
                       c=colors, alpha=0.5, s=20)
    axes[1, 1].set_title("Monthly Charges vs Support Calls")
    axes[1, 1].set_xlabel("Monthly Charges ($)")
    axes[1, 1].set_ylabel("Support Calls")
    axes[1, 1].legend(handles=[
        mpatches.Patch(color=STAY_COLOR, label="No Churn"),
        mpatches.Patch(color=CHURN_COLOR, label="Churn"),
    ])

    plt.tight_layout()
    out_path = FIGURES_DIR / "eda_charts.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"EDA charts saved to {out_path}")


if __name__ == "__main__":
    main()
