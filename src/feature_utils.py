import numpy as np


def compute_features_from_lists(monthly_income_list, monthly_expense_list):
    income = np.array(monthly_income_list)
    expense = np.array(monthly_expense_list)
    savings = income - expense

    expense_ratio = expense / income
    savings_ratio = savings / income
    x = np.arange(len(income))

    mean_expense = np.mean(expense)
    spike_months = expense > mean_expense * 1.15
    irregular_freq = float(np.mean(spike_months))
    avg_irregular_amt = float(np.mean(expense[spike_months])) if spike_months.any() else 0.0

    neg = (savings < 0).astype(int)
    max_streak = 0
    current = 0
    for v in neg:
        if v == 1:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0

    severe_overspend_freq = float(np.mean(expense > income * 1.20))

    return {
        "avg_income": float(np.mean(income)),
        "income_volatility": float(np.std(income) / (np.mean(income) + 1e-9)),
        "income_growth_rate": float(np.polyfit(x, income, 1)[0]),
        "expense_ratio_mean": float(np.mean(expense_ratio)),
        "expense_volatility": float(np.std(expense_ratio) / (np.mean(expense_ratio) + 1e-9)),
        "irregular_freq": irregular_freq,
        "avg_irregular_amt": avg_irregular_amt,
        "savings_volatility": float(np.std(savings_ratio) / (abs(np.mean(savings_ratio)) + 1e-9)),
        "neg_savings_freq": float(np.mean(savings < 0)),
        "severe_overspend_freq": severe_overspend_freq,
        "max_neg_savings_streak": int(max_streak),
        "city_tier_code": 2.0,
    }