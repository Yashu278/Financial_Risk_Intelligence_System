"""
predict.py — Phase 2 Inference Layer  (v2 — with input validation)
AI-Driven Financial Risk & Intelligence System
"""

import os, numpy as np, joblib

MODELS_DIR = "models"
_MODEL = _SCALER = _ENCODER = _FEAT_COLS = None

# ─────────────────────────────────────────────
# Input bounds (from actual dataset statistics)
# Any value outside these is likely a data error
# ─────────────────────────────────────────────
INPUT_BOUNDS = {
    "avg_income"            : (5000,    500000),
    "income_volatility"     : (0.0,     1.0),
    "income_growth_rate"    : (-0.05,   0.05),
    "expense_ratio_mean"    : (0.0,     2.0),
    "expense_volatility"    : (0.0,     1.0),
    "severe_overspend_freq" : (0.0,     1.0),
    "irregular_freq"        : (0.0,     1.0),
    "avg_irregular_amt"     : (0.0,     100000),
    "savings_volatility"    : (0.0,     700.0),
    "neg_savings_freq"      : (0.0,     1.0),
    "max_neg_savings_streak": (0,       24),
    "city_tier_code"        : (1,       3),
    "age"                   : (18,      80),
}

def _load_artifacts():
    global _MODEL, _SCALER, _ENCODER, _FEAT_COLS
    if _MODEL is not None:
        return
    for f in ["risk_model.pkl","scaler.pkl","label_encoder.pkl","feature_cols.pkl"]:
        p = os.path.join(MODELS_DIR, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}\nRun python src/risk_model.py first.")
    _MODEL     = joblib.load(os.path.join(MODELS_DIR, "risk_model.pkl"))
    _SCALER    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    _ENCODER   = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    _FEAT_COLS = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))

def _validate_inputs(inputs: dict):
    """FIX: reject impossible values before they silently corrupt predictions."""
    errors = []
    for col, (lo, hi) in INPUT_BOUNDS.items():
        if col in inputs:
            val = inputs[col]
            if not (lo <= val <= hi):
                errors.append(f"  {col}={val} is outside valid range [{lo}, {hi}]")
    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(errors))

def predict_risk(
    avg_income, income_volatility, income_growth_rate,
    expense_ratio_mean, expense_volatility,
    irregular_freq, avg_irregular_amt,
    savings_volatility, neg_savings_freq,
    max_neg_savings_streak, city_tier_code, age,
    severe_overspend_freq=None,   # optional: auto-derived if not provided
) -> dict:
    """
    Predict financial risk for one user.
    Returns: risk_label, confidence, probabilities, explanation
    """
    from scipy import stats as scipy_stats

    _load_artifacts()

    # Auto-derive severe_overspend_freq if not supplied
    if severe_overspend_freq is None:
        sigma = max(expense_volatility, 1e-6)
        severe_overspend_freq = float(
            np.clip(1 - scipy_stats.norm.cdf(0.90, loc=expense_ratio_mean, scale=sigma), 0, 1)
        )

    inputs = {
        "avg_income"            : avg_income,
        "income_volatility"     : income_volatility,
        "income_growth_rate"    : income_growth_rate,
        "expense_ratio_mean"    : expense_ratio_mean,
        "expense_volatility"    : expense_volatility,
        "severe_overspend_freq" : severe_overspend_freq,
        "irregular_freq"        : irregular_freq,
        "avg_irregular_amt"     : avg_irregular_amt,
        "savings_volatility"    : savings_volatility,
        "neg_savings_freq"      : neg_savings_freq,
        "max_neg_savings_streak": max_neg_savings_streak,
        "city_tier_code"        : city_tier_code,
        "age"                   : age,
    }

    # Validate BEFORE prediction — catches impossible values
    _validate_inputs(inputs)

    # Build feature row in EXACT training order
    try:
        row = np.array([[inputs[col] for col in _FEAT_COLS]])
    except KeyError as e:
        raise ValueError(f"Missing feature: {e}. Expected: {_FEAT_COLS}")

    row_scaled = _SCALER.transform(row)
    pred_enc   = _MODEL.predict(row_scaled)[0]
    label      = _ENCODER.inverse_transform([pred_enc])[0]
    proba      = _MODEL.predict_proba(row_scaled)[0]
    classes    = _ENCODER.classes_
    prob_dict  = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    confidence = round(float(proba[pred_enc]), 4)

    return {
        "risk_label"   : label,
        "confidence"   : confidence,
        "probabilities": prob_dict,
        "explanation"  : _explain(label, inputs),
    }

def _explain(label, f) -> str:
    reasons = []
    if f["neg_savings_freq"] >= 0.30:
        reasons.append(f"Negative savings in {f['neg_savings_freq']*100:.0f}% of months")
    if f["expense_ratio_mean"] >= 0.75:
        reasons.append(f"Spending {f['expense_ratio_mean']*100:.0f}% of income on expenses")
    if f["income_volatility"] >= 0.25:
        reasons.append(f"High income instability (CV={f['income_volatility']:.2f})")
    if f["income_growth_rate"] < 0:
        reasons.append("Income trend is declining")
    if f["max_neg_savings_streak"] >= 3:
        reasons.append(f"Longest negative-savings run: {f['max_neg_savings_streak']} months")
    if not reasons:
        reasons = ["Financial profile stable across all key metrics"]
    prefix = {"High": "🔴 HIGH RISK — ", "Medium": "🟡 MEDIUM RISK — ", "Low": "🟢 LOW RISK — "}[label]
    return prefix + " | ".join(reasons)

# ─────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*62)
    print("  PREDICT.PY — SELF TEST (v2)")
    print("═"*62)

    cases = [
        ("High", dict(avg_income=28000, income_volatility=0.55,
            income_growth_rate=-0.005, expense_ratio_mean=0.92,
            expense_volatility=0.45, irregular_freq=0.70,
            avg_irregular_amt=18000, savings_volatility=5.0,
            neg_savings_freq=0.75, max_neg_savings_streak=6,
            city_tier_code=3, age=24)),

        ("Low",  dict(avg_income=120000, income_volatility=0.05,
            income_growth_rate=0.015, expense_ratio_mean=0.40,
            expense_volatility=0.08, irregular_freq=0.04,
            avg_irregular_amt=5000, savings_volatility=0.10,
            neg_savings_freq=0.0, max_neg_savings_streak=0,
            city_tier_code=2, age=42)),

        ("Medium", dict(avg_income=55000, income_volatility=0.18,
            income_growth_rate=0.003, expense_ratio_mean=0.65,
            expense_volatility=0.20, irregular_freq=0.20,
            avg_irregular_amt=10000, savings_volatility=0.90,
            neg_savings_freq=0.08, max_neg_savings_streak=1,
            city_tier_code=2, age=33)),
    ]

    # Validation test — should raise error
    print("\n[VALIDATION TEST] Passing expense_ratio_mean=5 (impossible value):")
    try:
        predict_risk(avg_income=50000, income_volatility=0.1,
            income_growth_rate=0.001, expense_ratio_mean=5.0,  # ← invalid
            expense_volatility=0.15, irregular_freq=0.05,
            avg_irregular_amt=8000, savings_volatility=0.5,
            neg_savings_freq=0.05, max_neg_savings_streak=1,
            city_tier_code=2, age=30)
    except ValueError as e:
        print(f"  ✅ Correctly caught: {e}")

    # Normal tests
    print()
    all_pass = True
    for expected, inputs in cases:
        result = predict_risk(**inputs)
        ok = result["risk_label"] == expected
        if not ok:
            all_pass = False
        print(f"  {'✅' if ok else '❌'} Expected={expected:<8} Got={result['risk_label']:<8} "
              f"conf={result['confidence']:.2%}")
        print(f"     {result['explanation']}\n")

    print("✅ All prediction tests passed!" if all_pass else "❌ Some tests failed — review model.")
    print("═"*62 + "\n")
