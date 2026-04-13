import os

import numpy as np
import pandas as pd


PERSONAS = {
    "stable_salaried": {
        "income_mean": 60000,
        "income_std": 3000,
        "growth_rate": 0.003,
        "shock_prob": 0.03,
        "rent_ratio": 0.30,
        "food_ratio": 0.15,
        "travel_ratio": 0.05,
        "utilities_ratio": 0.05,
        "other_ratio": 0.10,
    },
    "gig_volatile": {
        "income_mean": 45000,
        "income_std": 12000,
        "growth_rate": 0.001,
        "shock_prob": 0.08,
        "rent_ratio": 0.28,
        "food_ratio": 0.18,
        "travel_ratio": 0.08,
        "utilities_ratio": 0.06,
        "other_ratio": 0.12,
    },
    "high_earner_lifestyle": {
        "income_mean": 120000,
        "income_std": 10000,
        "growth_rate": 0.005,
        "shock_prob": 0.02,
        "rent_ratio": 0.30,
        "food_ratio": 0.12,
        "travel_ratio": 0.15,
        "utilities_ratio": 0.04,
        "other_ratio": 0.20,
    },
    "conservative_saver": {
        "income_mean": 50000,
        "income_std": 2000,
        "growth_rate": 0.004,
        "shock_prob": 0.01,
        "rent_ratio": 0.25,
        "food_ratio": 0.12,
        "travel_ratio": 0.03,
        "utilities_ratio": 0.04,
        "other_ratio": 0.06,
    },
    "financially_struggling": {
        "income_mean": 25000,
        "income_std": 5000,
        "growth_rate": 0.001,
        "shock_prob": 0.10,
        "rent_ratio": 0.40,
        "food_ratio": 0.25,
        "travel_ratio": 0.05,
        "utilities_ratio": 0.08,
        "other_ratio": 0.15,
    },
}

CITY_TIERS = ["metro", "tier2", "tier3"]
CITY_TIER_WEIGHTS = [0.40, 0.35, 0.25]
USERS_PER_PERSONA = 400
MONTHS = 24

# Small, persona-aware probability of an overspend month.
OVERSPEND_PROB = {
    "stable_salaried": 0.03,
    "gig_volatile": 0.10,
    "high_earner_lifestyle": 0.06,
    "conservative_saver": 0.01,
    "financially_struggling": 0.18,
}


def _positive(value: float) -> float:
    return max(value, 0.0)


def generate_data(seed=42):
    np.random.seed(seed)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("[data_generation] Starting...")

    rows = []
    user_id = 1

    for segment, params in PERSONAS.items():
        for _ in range(USERS_PER_PERSONA):
            base_income = np.random.normal(params["income_mean"], params["income_std"])
            base_income = max(base_income, 1000)
            city_tier = np.random.choice(CITY_TIERS, p=CITY_TIER_WEIGHTS)

            for month in range(1, MONTHS + 1):
                income = base_income * (1 + params["growth_rate"] * month) + np.random.normal(0, params["income_std"])
                income = max(income, 1000)

                if np.random.random() < params["shock_prob"]:
                    income *= np.random.uniform(0.5, 0.8)
                    income = max(income, 1000)

                rent = _positive(params["rent_ratio"] * income * (1 + np.random.normal(0, 0.05)))
                food = _positive(params["food_ratio"] * income * (1 + np.random.normal(0, 0.05)))
                travel = _positive(params["travel_ratio"] * income * (1 + np.random.normal(0, 0.05)))
                utilities = _positive(params["utilities_ratio"] * income * (1 + np.random.normal(0, 0.05)))
                other = _positive(params["other_ratio"] * income * (1 + np.random.normal(0, 0.05)))

                # Overspend months capture emergency/impulse spending patterns.
                # This ensures some users exceed income and severe overspend is learnable.
                if np.random.random() < OVERSPEND_PROB[segment]:
                    overspend_multiplier = np.random.uniform(1.15, 1.45)
                    rent *= overspend_multiplier
                    food *= overspend_multiplier
                    travel *= overspend_multiplier
                    utilities *= overspend_multiplier
                    other *= overspend_multiplier

                total_expense = rent + food + travel + utilities + other
                savings = income - total_expense
                expense_ratio = total_expense / income

                rows.append(
                    {
                        "user_id": user_id,
                        "segment": segment,
                        "month": month,
                        "income": round(income, 2),
                        "rent": round(rent, 2),
                        "food": round(food, 2),
                        "travel": round(travel, 2),
                        "utilities": round(utilities, 2),
                        "other": round(other, 2),
                        "total_expense": round(total_expense, 2),
                        "savings": round(savings, 2),
                        "expense_ratio": round(expense_ratio, 6),
                        "city_tier": city_tier,
                    }
                )

            user_id += 1

    df = pd.DataFrame(rows)
    full_columns = [
        "user_id",
        "segment",
        "month",
        "income",
        "rent",
        "food",
        "travel",
        "utilities",
        "other",
        "total_expense",
        "savings",
        "expense_ratio",
        "city_tier",
    ]
    clean_columns = [
        "user_id",
        "month",
        "income",
        "rent",
        "food",
        "travel",
        "utilities",
        "other",
        "total_expense",
        "savings",
        "expense_ratio",
        "city_tier",
    ]

    df = df[full_columns]
    full_path = "data/raw/finance_data_full.csv"
    clean_path = "data/raw/finance_data.csv"

    df.to_csv(full_path, index=False)
    df[clean_columns].to_csv(clean_path, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Unique users: {df['user_id'].nunique()}")
    print(df.groupby("segment")["savings"].mean())
    print("[data_generation] Done. Saved to data/raw/")

    return df


if __name__ == "__main__":
    generate_data(seed=42)
