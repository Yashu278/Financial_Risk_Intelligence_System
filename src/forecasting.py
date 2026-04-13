from __future__ import annotations

import os
import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def forecast_user(user_data_df, user_id):
    """
    Takes the full finance dataframe and a user_id.
    Returns a dict with expense_trend and savings_trend labels.
    """
    user_rows = user_data_df[user_data_df["user_id"] == user_id].sort_values("month")

    if len(user_rows) < 12:
        return {
            "user_id": user_id,
            "expense_trend": "Insufficient Data",
            "savings_trend": "Insufficient Data",
        }

    def get_trend(series_values, direction="expense"):
        try:
            series = pd.Series(
                series_values,
                index=pd.date_range(start="2024-01-01", periods=len(series_values), freq="MS"),
            )
            model = ARIMA(series, order=(2, 1, 2))
            result = model.fit(method_kwargs={"maxiter": 50})
            forecast = result.forecast(steps=6)
            slope = forecast.values[-1] - forecast.values[0]

            if direction == "expense":
                # Rising expenses = worse
                return "Deteriorating" if slope > 0 else ("Improving" if slope < 0 else "Stable")
            # Rising savings = better
            return "Improving" if slope > 0 else ("Deteriorating" if slope < 0 else "Stable")
        except Exception:
            return "Forecast Error"

    return {
        "user_id": user_id,
        "expense_trend": get_trend(user_rows["total_expense"].values, "expense"),
        "savings_trend": get_trend(user_rows["savings"].values, "savings"),
    }


def run_batch_forecasting(input_path="data/raw/finance_data.csv", output_path="data/processed/forecasts.csv"):
    raw_df = pd.read_csv(input_path)
    results = []

    for uid in raw_df["user_id"].unique():
        if uid % 100 == 0:
            print(f"Processed {uid} users...")
        results.append(forecast_user(raw_df, uid))

    forecasts_df = pd.DataFrame(results)

    # Calibrate savings trend using user-level savings ratio so output aligns
    # with intuitive financial profiles in downstream UX.
    user_level = (
        raw_df.groupby("user_id")
        .agg(avg_income=("income", "mean"), avg_savings=("savings", "mean"))
        .reset_index()
    )
    user_level["savings_ratio_mean"] = user_level["avg_savings"] / user_level["avg_income"]

    forecasts_df = forecasts_df.merge(user_level[["user_id", "savings_ratio_mean"]], on="user_id", how="left")
    forecasts_df.loc[
        (forecasts_df["savings_ratio_mean"] >= 0.20) & (forecasts_df["savings_trend"] == "Deteriorating"),
        "savings_trend",
    ] = "Stable"
    forecasts_df.loc[
        (forecasts_df["savings_ratio_mean"] <= 0.05) & (forecasts_df["savings_trend"].isin(["Improving", "Stable"])),
        "savings_trend",
    ] = "Deteriorating"
    forecasts_df = forecasts_df.drop(columns=["savings_ratio_mean"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecasts_df.to_csv(output_path, index=False)
    print(forecasts_df["expense_trend"].value_counts())
    print(f"Saved forecasts to {output_path}")
    return forecasts_df


def plot_user_forecast(user_id, raw_df):
    """Returns a matplotlib figure of historical + forecast expense for one user."""
    import matplotlib.pyplot as plt

    user_rows = raw_df[raw_df["user_id"] == user_id].sort_values("month")
    expense_series = pd.Series(
        user_rows["total_expense"].values,
        index=pd.date_range(start="2024-01-01", periods=len(user_rows), freq="MS"),
    )
    model = ARIMA(expense_series, order=(2, 1, 2))
    result = model.fit(method_kwargs={"maxiter": 50})
    forecast = result.forecast(steps=6)
    fcast_idx = pd.date_range(
        start=expense_series.index[-1] + pd.DateOffset(months=1),
        periods=6,
        freq="MS",
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(expense_series.index, expense_series.values, label="Historical Expense", color="blue")
    ax.plot(fcast_idx, forecast.values, label="Forecast (6 months)", color="red", linestyle="--")
    ax.set_title(f"Expense Forecast - User {user_id}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Expense")
    ax.legend()
    plt.tight_layout()
    return fig


def forecast_user_realtime(monthly_income_list, monthly_expense_list):
    """
    Takes raw income and expense lists from the UI.
    Returns trends and matplotlib figures for both expense and savings.
    """
    import matplotlib.pyplot as plt

    savings_list = [i - e for i, e in zip(monthly_income_list, monthly_expense_list)]

    def _forecast_series(series_values, direction, title):
        try:
            series = pd.Series(
                series_values,
                index=pd.date_range(start="2024-01-01", periods=len(series_values), freq="MS"),
            )
            model = ARIMA(series, order=(2, 1, 2))
            result = model.fit(method_kwargs={"maxiter": 50})
            forecast = result.forecast(steps=6)
            slope = forecast.values[-1] - forecast.values[0]

            if direction == "expense":
                trend = "Deteriorating" if slope > 0 else ("Improving" if slope < 0 else "Stable")
            else:
                trend = "Improving" if slope > 0 else ("Deteriorating" if slope < 0 else "Stable")

            fcast_idx = pd.date_range(
                start=series.index[-1] + pd.DateOffset(months=1),
                periods=6,
                freq="MS",
            )

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(series.index, series.values, label="Historical", color="blue")
            ax.plot(fcast_idx, forecast.values, label="Forecast (6 months)", color="red", linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("Month")
            ax.set_ylabel("Amount")
            ax.legend()
            plt.tight_layout()

            return trend, fig
        except Exception:
            return "Forecast Error", None

    expense_trend, expense_fig = _forecast_series(monthly_expense_list, "expense", "Expense Forecast")
    savings_trend, savings_fig = _forecast_series(savings_list, "savings", "Savings Forecast")

    return {
        "expense_trend": expense_trend,
        "savings_trend": savings_trend,
        "expense_fig": expense_fig,
        "savings_fig": savings_fig,
    }


if __name__ == "__main__":
    run_batch_forecasting()
