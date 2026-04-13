import os

import numpy as np
import pandas as pd


CITY_TIER_MAP = {"metro": 3, "tier2": 2, "tier3": 1}


def _max_negative_streak(values: np.ndarray) -> int:
    streak = 0
    max_streak = 0
    for value in values:
        if value < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def engineer_features():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    print("[feature_engineering] Starting...")

    raw = pd.read_csv("data/raw/finance_data.csv")
    feature_rows = []

    for user_id, group in raw.groupby("user_id"):
        g = group.sort_values("month").reset_index(drop=True)

        income_values = g["income"].to_numpy(dtype=float)
        expense_values = g["total_expense"].to_numpy(dtype=float)
        savings_values = g["savings"].to_numpy(dtype=float)
        expense_ratio_values = g["expense_ratio"].to_numpy(dtype=float)
        months = np.arange(len(g), dtype=float)

        avg_income = float(np.mean(income_values))
        income_volatility = float(np.std(income_values) / (avg_income + 1e-9))
        income_growth_rate = float(np.polyfit(months, income_values, 1)[0])

        expense_ratio_mean = float(np.mean(expense_values / (income_values + 1e-9)))
        expense_volatility = float(np.std(expense_ratio_values) / (np.mean(expense_ratio_values) + 1e-9))

        savings_ratio = savings_values / (income_values + 1e-9)
        savings_ratio_mean = float(np.mean(savings_ratio))
        savings_volatility = float(np.std(savings_ratio) / (abs(np.mean(savings_ratio)) + 1e-9))

        neg_savings_freq = float(np.mean(savings_values < 0))
        severe_overspend_freq = float(np.mean(expense_values > income_values * 1.20))
        max_neg_savings_streak = int(_max_negative_streak(savings_values))
        low_savings_freq = float(np.mean(savings_ratio < 0.05))

        monthly_mean_expense = float(np.mean(expense_values))
        irregular_mask = expense_values > (monthly_mean_expense * 1.15)
        irregular_freq = float(np.mean(irregular_mask))
        avg_irregular_amt = float(np.mean(expense_values[irregular_mask])) if np.any(irregular_mask) else 0.0

        city_tier = g["city_tier"].iloc[0]
        city_tier_code = int(CITY_TIER_MAP[city_tier])

        feature_rows.append(
            {
                "user_id": int(user_id),
                "avg_income": avg_income,
                "income_volatility": income_volatility,
                "income_growth_rate": income_growth_rate,
                "expense_ratio_mean": expense_ratio_mean,
                "expense_volatility": expense_volatility,
                "savings_ratio_mean": savings_ratio_mean,
                "savings_volatility": savings_volatility,
                "neg_savings_freq": neg_savings_freq,
                "severe_overspend_freq": severe_overspend_freq,
                "max_neg_savings_streak": max_neg_savings_streak,
                "low_savings_freq": low_savings_freq,
                "irregular_freq": irregular_freq,
                "avg_irregular_amt": avg_irregular_amt,
                "city_tier_code": city_tier_code,
            }
        )

    df = pd.DataFrame(feature_rows)
    output_columns = [
        "user_id",
        "avg_income",
        "income_volatility",
        "income_growth_rate",
        "expense_ratio_mean",
        "expense_volatility",
        "savings_ratio_mean",
        "savings_volatility",
        "neg_savings_freq",
        "severe_overspend_freq",
        "max_neg_savings_streak",
        "low_savings_freq",
        "irregular_freq",
        "avg_irregular_amt",
        "city_tier_code",
    ]
    df = df[output_columns]

    assert df.isnull().sum().sum() == 0, "NULL values found"
    assert len(df) == 2000, "Wrong user count"
    print("Null check: PASSED")
    print("User count: PASSED")
    print(df.describe())

    output_path = "data/processed/features.csv"
    df.to_csv(output_path, index=False)
    print("[feature_engineering] Done. Saved to data/processed/")

    return df


if __name__ == "__main__":
    engineer_features()
