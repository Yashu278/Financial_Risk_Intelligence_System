# Financial Risk Intelligence System

## Overview
This project predicts financial risk (Low / Medium / High) using behavioral financial data instead of raw monetary values.

## Features
- Synthetic financial data generation (person-based simulation)
- Behavioral feature engineering
- Risk scoring and labeling
- Machine learning models (Logistic Regression, Decision Tree, Random Forest)
- Risk prediction with explanation

## Project Structure
- src/ → core logic (data, features, model, prediction)
- requirements.txt → dependencies
- test.py → environment check

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Run model training:
python src/risk_model.py

3. Run prediction:
python src/predict.py

## Current Status
- Phase 1: Data pipeline ✅
- Phase 2: Risk modeling ✅
- Phase 3+: In progress