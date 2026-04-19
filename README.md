# 🧠 AI-Driven Financial Risk & Intelligence System

A behavioral finance analysis system that classifies financial risk, forecasts future trends, and simulates investment uncertainty — built with machine learning, time-series modeling, and Monte Carlo simulation.

> Not a trading bot. Not a stock predictor.  
> A system that understands **how people behave with money** — and whether that behavior is dangerous.

---

## 📌 Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1A | Synthetic Data Generation | ✅ Complete |
| Phase 1B | Behavioral Feature Engineering | ✅ Complete |
| Phase 2 | Risk Classification (ML Core) | ✅ Complete |
| Phase 3 | Forecasting Engine (ARIMA) | ✅ Complete |
| Phase 4 | Monte Carlo Simulation | ✅ Complete |
| Phase 5 | System Integration | ✅ Complete |
| Phase 6 | FinTalkBot Decision Layer | ✅ Complete |
| Phase 7 | Streamlit UI | ✅ Complete |

---

## 🗂️ Project Structure

```
Financial_Risk_Intelligence_System/
│
├── data/
│   ├── raw/
│   │   ├── finance_data.csv          # 48,000 monthly records (ML-safe, no segment)
│   │   └── finance_data_full.csv     # Same data with segment column (audit only)
│   └── processed/
│       ├── features.csv              # 2,000 user behavioral profiles
│       └── features_labeled.csv      # Same + risk_score + risk_label
│
├── models/
│   ├── risk_model.pkl                # Best trained classifier
│   ├── scaler.pkl                    # StandardScaler fitted on training data
│   ├── label_encoder.pkl             # LabelEncoder for High/Medium/Low
│   └── feature_cols.pkl             # Ordered feature list (prevents silent mismatch)
│
├── src/
│   ├── data_generation.py            # Phase 1A: Synthetic finance data engine
│   ├── feature_engineering.py        # Phase 1B: Monthly → behavioral profiles
│   ├── pipeline_contract.py          # Shared feature/artifact contract
│   ├── pipeline.py                   # Single rebuild entrypoint
│   ├── risk_model.py                 # Phase 2: Full ML training pipeline
│   └── predict.py                    # Phase 2: Inference with validation + explanation
│
├── knowledge_base/                   # Optional future RAG documents
├── app.py                            # Phase 7: Streamlit UI
├── requirements.txt
├── test.py
└── ENV_start.txt
```

Deprecated artifact:

```text
src/risk_labeling.ipynb  # historical prototype only, not part of the operational pipeline
```

---

## ⚙️ Setup

```bash
# Activate virtual environment
source ENVIRONMENT/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify environment
python test.py
# Expected output: Environment OK
```

---

## 🚀 How to Run

### Single supported rebuild path

```bash
python src/pipeline.py --clean
```

This is the only operational path. It rebuilds raw data, engineered features, labels, model artifacts, and validation outputs in one run.

Pipeline scope note:
- The pipeline includes batch forecasting output generation (`data/processed/forecasts.csv`) for full reproducibility.
- The Streamlit app still performs user-specific realtime forecasting at runtime for live inputs.

### Verification commands

```bash
python src/validate_phase2.py
python src/predict.py
python test.py
```

## 🚀 Phase 7 Demo Flow

Run the Streamlit app from the `1st_Draft` folder:

```bash
streamlit run app.py
```

Demo script:
1. Enter income `50000`, expenses `40000`, investment `100000`, return `8%`, volatility `15%`, and years `5`.
2. Click Analyze My Financial Profile.
3. Show the risk label, trends, charts, and Monte Carlo summary.
4. Switch to FinTalkBot (final decision layer) and ask: `How should I improve my savings?`
5. Show the profile-aware action guidance generated from risk + trends + simulation context.

If Tab 2 is opened before Tab 1, the app shows a warning and waits for analysis first.

### FinTalkBot API key usage (Dual-key)

FinTalkBot uses a dual-key model:

1. If a user enters a key in Tab 2, that key is used first.
2. If user key is empty, the app tries a system demo key from environment variables.
3. If neither is available, the app uses rule-based fallback mode.
4. The app does not write keys to repository files.
5. Keys are not shown in logs.
6. User-entered keys are session-local and cleared when session state is reset.

System demo key env variable names:

```text
SYSTEM_GEMINI_API_KEY
SYSTEM_OPENAI_API_KEY
SYSTEM_ANTHROPIC_API_KEY
SYSTEM_GROQ_API_KEY
```

Legacy compatible variable names (also supported):

```text
GEMINI_API_KEY
OPENAI_API_KEY
ANTHROPIC_API_KEY
GROQ_API_KEY
```

Keep `.env` local and never commit it.

---

## 📊 Phase 1 — Data Foundation

### 1A: Synthetic Data Generation (`data_generation.py`)

Simulates realistic personal finance history for **2,000 users** across **24 months** (48,000 rows total).

**5 user personas:**

| Segment | Income Level | Behavior |
|---------|-------------|----------|
| `stable_salaried` | Medium | Low volatility, consistent savings |
| `gig_volatile` | Variable | High income shocks, irregular expenses |
| `high_earner_lifestyle` | High | High spending relative to income |
| `conservative_saver` | Medium | Low expenses, high savings rate |
| `financially_struggling` | Low | Frequent negative savings |

**Design decisions:**
- Lognormal income distribution (realistic right-skew)
- City-tier cost multipliers (metro / tier2 / tier3)
- 3% random income shock probability per month
- Two output files: full (with `segment`) and ML-safe (without)

**Leakage protection:** `segment` is removed before any ML work. It exists only in `finance_data_full.csv` for auditing.

### 1B: Feature Engineering (`feature_engineering.py`)

Compresses 24 monthly rows → 1 behavioral profile per user.

**Features computed:**

| Category | Features |
|----------|----------|
| Income stability | `avg_income`, `income_volatility` (CV), `income_growth_rate` |
| Expense behavior | `expense_ratio_mean`, `expense_volatility`, `irregular_freq`, `avg_irregular_amt` |
| Savings behavior | `savings_volatility`, `neg_savings_freq`, `max_neg_savings_streak` |
| Derived stress | `severe_overspend_freq` (fraction of months with expense > 120% of income) |
| Context | `city_tier_code`, `age` |

---

## 🤖 Phase 2 — Risk Classification

## Risk Scoring

Risk scores are computed using a weighted combination of five behavioral
features:

    neg_savings_freq:      0.30
    expense_ratio_mean:    0.25
    income_volatility:     0.20
    savings_volatility:    0.15
    severe_overspend_freq: 0.10

Before computing the weighted sum, each component is normalized to [0, 1]
using MinMaxScaler fitted on the full feature dataset. This means the
risk score is dataset-relative — a user's score reflects their position
within the distribution of all 2,000 users, not an absolute domain threshold.

Risk labels (High / Medium / Low) are then assigned using 33rd and 66th
percentile quantile thresholds on the resulting risk scores, ensuring
balanced class distribution (~33% per class).

This approach is intentional: it produces stable, comparable scores
across the synthetic population and ensures no single extreme user
distorts the label boundaries.

Limitation: Because MinMaxScaler is dataset-dependent, the risk score
thresholds will shift if the underlying population changes significantly.
In a production system, these thresholds would be fixed after validation
on a representative dataset.

### Model Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | **99.25%** | **0.9925** ← best |
| Random Forest | 97.50% | 0.9752 |
| Decision Tree | 95.00% | 0.9504 |

**Why 99% accuracy is expected, not suspicious:**  
The model is trained to replicate a mathematically defined rule. High accuracy confirms the features cleanly capture that rule — it is not overfitting to noise.

**Top features (Random Forest importance):**

```
severe_overspend_freq   0.2914  ██████████████
expense_ratio_mean      0.2207  ███████████
savings_volatility      0.1601  ████████
neg_savings_freq        0.1007  █████
income_volatility       0.0934  ████
city_tier_code          0.0077       ← no location bias
```

### Prediction API

```python
from src.predict import predict_risk

result = predict_risk(
    avg_income=50000,
    income_volatility=0.35,
    income_growth_rate=-0.002,
    expense_ratio_mean=0.88,
    expense_volatility=0.40,
    irregular_freq=0.50,
    avg_irregular_amt=15000,
    savings_volatility=3.50,
    neg_savings_freq=0.60,
    severe_overspend_freq=0.45,
    max_neg_savings_streak=4,
    city_tier_code=3
)

# Returns:
# {
#   "risk_label": "High",
#   "confidence": 0.9987,
#   "probabilities": {"High": 0.9987, "Low": 0.0, "Medium": 0.0013},
#   "explanation": "🔴 HIGH RISK — Negative savings in 60% of months | ..."
# }
```

Input validation is enforced. Values outside realistic bounds raise a `ValueError` before the model is called.

---

## 🧪 Design Principles

**No data leakage** — StandardScaler is fit only on training data, then applied to test. Segment column excluded from all ML inputs.

**Statistically aligned features** — `severe_overspend_freq` is computed empirically as the fraction of months where expense exceeds 120% of income, and the same definition is used in training and runtime inference.

**Balanced labels** — Percentile thresholds guarantee ~33% per class. Fixed thresholds would create imbalance and corrupt evaluation metrics.

**Production-ready inference** — Feature order is enforced via `feature_cols.pkl`. Artifacts are verified on reload.

**Notebook status** — `src/risk_labeling.ipynb` is a deprecated historical prototype only. It is not part of the operational pipeline and should not be used to generate labels.

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
statsmodels
scipy
matplotlib
streamlit
joblib
```

Install with: `pip install -r requirements.txt`

---

## 👤 Author

**Yashdeep Saxena**  
AI-Driven Financial Risk & Intelligence System  
Major Project — 2026
