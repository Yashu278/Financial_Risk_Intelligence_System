from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from statsmodels.tsa.arima.model import ARIMA

from src.feature_utils import compute_features_from_lists
from src.fintalkbot import get_financial_advice
from src.monte_carlo import run_monte_carlo
from src.predict import predict_risk


app = FastAPI(title="FinTalk API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    monthly_income_list: list[float] = Field(..., min_length=6)
    monthly_expense_list: list[float] = Field(..., min_length=6)
    investment_amount: float = Field(..., gt=0)
    annual_return: float = 0.08
    volatility: float = Field(0.15, ge=0)
    years: int = Field(5, gt=0)


class MonthValuePoint(BaseModel):
    month: str
    value: float


class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int


class ForecastSeries(BaseModel):
    historical: list[MonthValuePoint]
    forecast: list[MonthValuePoint]


class ForecastPayload(BaseModel):
    expense: ForecastSeries
    savings: ForecastSeries


class MonteCarloPayload(BaseModel):
    summary: dict[str, float]
    histogram_bins: list[HistogramBin]


class AnalyzeResponse(BaseModel):
    risk_label: str
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    explanation: str | None = None
    expense_trend: Literal["Improving", "Stable", "Deteriorating", "Forecast Error"]
    savings_trend: Literal["Improving", "Stable", "Deteriorating", "Forecast Error"]
    forecast: ForecastPayload
    monte_carlo: MonteCarloPayload


class ChatProfile(BaseModel):
    risk_label: str
    expense_trend: str
    savings_trend: str
    avg_income: float


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    profile: ChatProfile
    provider: str = "None (Rule-based)"
    api_key: str = ""
    model: str = ""


class ChatResponse(BaseModel):
    response: str
    mode: str


def _validate_lists(income: list[float], expense: list[float]) -> None:
    if len(income) != len(expense):
        raise HTTPException(status_code=400, detail="Income and expense lists must be same length.")
    if any(i <= 0 for i in income):
        raise HTTPException(status_code=400, detail="Income values must be positive.")
    if any(e < 0 for e in expense):
        raise HTTPException(status_code=400, detail="Expense values cannot be negative.")


def _month_series(start_month: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start_month, periods=periods, freq="MS")


def _to_points(index: pd.DatetimeIndex, values: np.ndarray) -> list[MonthValuePoint]:
    return [MonthValuePoint(month=str(dt.date()), value=float(v)) for dt, v in zip(index, values)]


def _trend_from_slope(slope: float, direction: str) -> str:
    if direction == "expense":
        return "Deteriorating" if slope > 0 else ("Improving" if slope < 0 else "Stable")
    return "Improving" if slope > 0 else ("Deteriorating" if slope < 0 else "Stable")


def _forecast_series(values: list[float], direction: str, start_month: str = "2024-01-01") -> tuple[str, ForecastSeries]:
    try:
        series_index = _month_series(start_month, len(values))
        series = pd.Series(values, index=series_index)
        model = ARIMA(series, order=(2, 1, 2))
        result = model.fit(method_kwargs={"maxiter": 50})
        forecast_values = result.forecast(steps=6).values
        slope = float(forecast_values[-1] - forecast_values[0])
        trend = _trend_from_slope(slope, direction)

        forecast_index = pd.date_range(start=series_index[-1] + pd.DateOffset(months=1), periods=6, freq="MS")

        payload = ForecastSeries(
            historical=_to_points(series_index, np.array(values, dtype=float)),
            forecast=_to_points(forecast_index, forecast_values),
        )
        return trend, payload
    except Exception:
        series_index = _month_series(start_month, len(values))
        payload = ForecastSeries(
            historical=_to_points(series_index, np.array(values, dtype=float)),
            forecast=[],
        )
        return "Forecast Error", payload


def _histogram_bins(values: np.ndarray, bins: int = 50) -> list[HistogramBin]:
    counts, edges = np.histogram(values, bins=bins)
    result: list[HistogramBin] = []
    for idx, count in enumerate(counts):
        result.append(
            HistogramBin(
                bin_start=float(edges[idx]),
                bin_end=float(edges[idx + 1]),
                count=int(count),
            )
        )
    return result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    income = payload.monthly_income_list
    expense = payload.monthly_expense_list
    _validate_lists(income, expense)

    features = compute_features_from_lists(income, expense)
    risk_result = predict_risk(features)
    risk_label = risk_result.get("risk_label", "Unknown")

    savings = [i - e for i, e in zip(income, expense)]
    expense_trend, expense_series = _forecast_series(expense, "expense")
    savings_trend, savings_series = _forecast_series(savings, "savings")

    mc_results = run_monte_carlo(
        investment_amount=payload.investment_amount,
        annual_return=payload.annual_return,
        volatility=payload.volatility,
        years=payload.years,
    )
    all_values = np.array(mc_results["all_values"], dtype=float)
    histogram_bins = _histogram_bins(all_values)

    response = AnalyzeResponse(
        risk_label=risk_label,
        confidence=risk_result.get("confidence"),
        probabilities=risk_result.get("probabilities"),
        explanation=risk_result.get("explanation"),
        expense_trend=expense_trend,
        savings_trend=savings_trend,
        forecast=ForecastPayload(expense=expense_series, savings=savings_series),
        monte_carlo=MonteCarloPayload(
            summary={
                "investment_amount": float(mc_results["investment_amount"]),
                "best_case": float(mc_results["best_case"]),
                "worst_case": float(mc_results["worst_case"]),
                "average_case": float(mc_results["average_case"]),
                "prob_of_loss": float(mc_results["prob_of_loss"]),
            },
            histogram_bins=histogram_bins,
        ),
    )
    return response


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    response_text, _, _ = get_financial_advice(
        payload.question,
        payload.profile.model_dump(),
        payload.provider,
        payload.api_key,
    )

    mode = f"AI ({payload.provider})" if payload.api_key.strip() else "Rule-Based"
    return ChatResponse(response=response_text, mode=mode)
