import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_utils import compute_features_from_lists
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
        input_dict = compute_features_from_lists(monthly_income_list, monthly_expense_list)

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
