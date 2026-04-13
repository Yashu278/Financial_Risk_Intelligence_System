import numpy as np

from src.predict import predict_risk
from src.forecasting import forecast_user_realtime
from src.monte_carlo import run_monte_carlo, plot_monte_carlo, summarize_monte_carlo


def analyze_user(monthly_income_list, monthly_expense_list,
                 investment_amount, annual_return=0.08, volatility=0.15, years=5):

    # Input validation
    if len(monthly_income_list) < 6:
        raise ValueError("Minimum 6 months of data required.")
    if len(monthly_income_list) != len(monthly_expense_list):
        raise ValueError("Income and expense lists must be same length.")
    if any(i <= 0 for i in monthly_income_list):
        raise ValueError("Income values must be positive.")
    if any(e < 0 for e in monthly_expense_list):
        raise ValueError("Expense values cannot be negative.")
    if investment_amount <= 0:
        raise ValueError("Investment amount must be positive.")

    # Step 1: Risk classification
    try:
        income  = np.array(monthly_income_list)
        expense = np.array(monthly_expense_list)
        savings = income - expense

        expense_ratio = expense / income
        savings_ratio = savings / income
        x = np.arange(len(income))
        expense_diff = np.diff(expense)
        expense_change = expense_diff / (expense[:-1] + 1e-9)
        irregular_mask = np.abs(expense_change) > 0.20
        avg_irregular_amt = float(np.mean(np.abs(expense_diff)[irregular_mask])) if np.any(irregular_mask) else 0.0

        max_streak = 0
        current_streak = 0
        for val in savings:
            if val < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        input_dict = {
            'avg_income':             float(np.mean(income)),
            'income_volatility':      float(np.std(income) / np.mean(income)),
            'income_growth_rate':     float(np.clip(
                np.polyfit(x, income, 1)[0] / (np.mean(income) + 1e-9),
                -0.5, 0.5
            )),
            'expense_ratio_mean':     float(np.mean(expense_ratio)),
            'expense_volatility':     float(np.clip(
                np.std(expense_ratio) / (np.mean(expense_ratio) + 1e-9),
                0.0, 1.0
            )),
            'irregular_freq':         float(np.mean(irregular_mask)) if len(irregular_mask) > 0 else 0.0,
            'avg_irregular_amt':      float(np.clip(avg_irregular_amt, 0, 100000)),
            'savings_ratio_mean':     float(np.mean(savings_ratio)),
            'savings_volatility':     float(np.clip(
                np.std(savings_ratio) / (abs(np.mean(savings_ratio)) + 1e-9),
                0.0, 2.0
            )),
            'neg_savings_freq':       float(np.mean(savings < 0)),
            'max_neg_savings_streak': float(np.clip(max_streak, 0, 24)),
            'city_tier_code':         2.0,
        }

        result = predict_risk(input_dict)
        if 'error' in result:
            risk_label = "Unknown"
        else:
            risk_label = result.get('risk_label', 'Unknown')
    except Exception as e:
        print(f"[ERROR] predict_risk failed: {e}")
        risk_label = "Unknown"

    # Step 2: Forecasting
    try:
        forecast = forecast_user_realtime(monthly_income_list, monthly_expense_list)
    except Exception as e:
        print(f"[ERROR] forecast failed: {e}")
        forecast = {}

    expense_trend = forecast.get('expense_trend', 'Forecast Error')
    savings_trend = forecast.get('savings_trend', 'Forecast Error')
    expense_fig   = forecast.get('expense_fig', None)
    savings_fig   = forecast.get('savings_fig', None)

    # Step 3: Monte Carlo
    try:
        mc_results = run_monte_carlo(investment_amount, annual_return, volatility, years)
        mc_summary = summarize_monte_carlo(mc_results)
        mc_fig     = plot_monte_carlo(mc_results)
    except Exception as e:
        print(f"[ERROR] monte_carlo failed: {e}")
        mc_results = {}
        mc_fig     = None
        mc_summary = f"Simulation error: {str(e)}"

    return {
        'risk_label':    risk_label,
        'expense_trend': expense_trend,
        'savings_trend': savings_trend,
        'expense_fig':   expense_fig,
        'savings_fig':   savings_fig,
        'mc_results':    mc_results,
        'mc_fig':        mc_fig,
        'mc_summary':    mc_summary
    }
