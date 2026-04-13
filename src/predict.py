import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.pipeline_contract import FEATURE_COLS_PATH, LABEL_ENCODER_PATH, MODEL_PATH, SCALER_PATH, load_feature_cols, required_input_fields

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
FEATURE_COLS = load_feature_cols(FEATURE_COLS_PATH)

REQUIRED_FIELDS = required_input_fields(FEATURE_COLS)

print(f"Model expects {len(FEATURE_COLS)} features from {FEATURE_COLS_PATH}.")
print(f"User must supply {len(REQUIRED_FIELDS)} fields (severe_overspend_freq is derived).")


def enforce_feature_contract(input_df):
    input_features = list(input_df.columns)

    if input_features != FEATURE_COLS:
        raise ValueError(
            "Feature order mismatch between input and trained artifact. "
            f"Expected: {FEATURE_COLS} | Got: {input_features}"
        )

    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(input_features):
        raise ValueError(
            "Model feature count mismatch. "
            f"model.n_features_in_={model.n_features_in_}, input_features={len(input_features)}"
        )

    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(input_features):
        raise ValueError(
            "Scaler feature count mismatch. "
            f"scaler.n_features_in_={scaler.n_features_in_}, input_features={len(input_features)}"
        )


def validate_input(input_dict):
    """
    Validates that all required fields are present and within bounds.
    Required fields are derived from feature_cols.pkl at load time,
    so this validation always matches the actual model schema.
    """
    errors = []

    # Define bounds for all possible fields
    ALL_BOUNDS = {
        "avg_income": (1000, 500000),
        "income_volatility": (0, 2.0),
        "income_growth_rate": (-0.5, 0.5),
        "expense_ratio_mean": (0, 2.0),
        "expense_volatility": (0, 1.0),
        "irregular_freq": (0, 1.0),
        "avg_irregular_amt": (0, 100000),
        "savings_volatility": (0, 2.0),
        "neg_savings_freq": (0, 1.0),
        "max_neg_savings_streak": (0, 24),
        "city_tier_code": (1, 3),
        "severe_overspend_freq": (0, 1.0),
    }

    # Only validate fields the model actually needs
    for field in REQUIRED_FIELDS:
        if field not in input_dict:
            errors.append(f"Missing required field: '{field}'")
            continue
        val = input_dict[field]
        if field in ALL_BOUNDS:
            low, high = ALL_BOUNDS[field]
            if not (low <= val <= high):
                errors.append(f"'{field}' = {val} is outside valid range [{low}, {high}]")

    return errors


def derive_features(input_dict):
    if "severe_overspend_freq" not in input_dict:
        mean = input_dict.get("expense_ratio_mean", 0.5)
        std = input_dict.get("expense_volatility", 0.1) + 1e-6
        input_dict["severe_overspend_freq"] = float(1 - norm.cdf(0.90, loc=mean, scale=std))
    return input_dict


def generate_explanation(input_dict, risk_label):
    neg_freq = input_dict.get("neg_savings_freq", 0)
    exp_ratio = input_dict.get("expense_ratio_mean", 0)
    volatility = input_dict.get("income_volatility", 0)
    overspend = input_dict.get("severe_overspend_freq", 0)

    if risk_label == "High":
        reasons = []
        if neg_freq > 0.4:
            reasons.append(f"negative savings in {neg_freq*100:.0f}% of months")
        if exp_ratio > 0.75:
            reasons.append(f"average expense ratio of {exp_ratio:.2f}")
        if overspend > 0.3:
            reasons.append(f"high probability of severe overspending ({overspend*100:.0f}%)")
        if volatility > 0.3:
            reasons.append(f"unstable income (volatility: {volatility:.2f})")
        if not reasons:
            reasons.append("combination of multiple risk factors")
        return "High financial risk detected due to: " + ", ".join(reasons) + "."

    elif risk_label == "Medium":
        return (
            f"Moderate financial risk. Expense ratio is {exp_ratio:.2f} "
            f"with savings turning negative in {neg_freq*100:.0f}% of months. "
            f"Manageable but requires attention."
        )

    else:
        return (
            f"Low financial risk. Expense ratio is {exp_ratio:.2f} "
            f"with stable savings pattern. "
            f"Negative savings in only {neg_freq*100:.0f}% of months."
        )


def predict_risk(input_dict):
    """
    Predict financial risk for a single user.

    Required input fields (derived from feature_cols.pkl at load time):
      These fields must be present in input_dict.
      Run: print(REQUIRED_FIELDS) to see the exact list for your model.

    Auto-derived field (do NOT supply manually):
      severe_overspend_freq — computed from expense_ratio_mean
                              and expense_volatility

    Returns:
      dict with keys:
        risk_label    : str  — "High", "Medium", or "Low"
        confidence    : float — probability of predicted class
        probabilities : dict  — probability for each class
        explanation   : str  — plain language reason
      OR on validation failure:
        {"error": [list of error strings]}
    """
    errors = validate_input(input_dict)
    if errors:
        return {"error": errors}

    input_dict = derive_features(input_dict)

    # Build input as DataFrame to enforce column order explicitly
    input_df = pd.DataFrame([input_dict])

    for col in FEATURE_COLS:
        if col not in input_df.columns:
            input_df[col] = 0.0

    # Enforce exact column order matching training
    input_df = input_df[FEATURE_COLS]

    enforce_feature_contract(input_df)

    # Scale using the DataFrame directly (preserves column names, no warning)
    input_scaled = scaler.transform(input_df)

    pred_encoded = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0]
    risk_label = le.inverse_transform([pred_encoded])[0]
    confidence = float(round(max(pred_proba), 4))

    probabilities = {cls: float(round(prob, 4)) for cls, prob in zip(le.classes_, pred_proba)}

    explanation = generate_explanation(input_dict, risk_label)

    return {
        "risk_label": risk_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "explanation": explanation,
    }


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("PHASE 2 — PREDICT.PY SELF-TEST")
    print("=" * 60)

    test1 = {
        "avg_income": 35000,
        "income_volatility": 0.12,
        "income_growth_rate": 0.03,
        "expense_ratio_mean": 0.62,
        "expense_volatility": 0.08,
        "irregular_freq": 0.10,
        "avg_irregular_amt": 2000,
        "savings_volatility": 0.10,
        "neg_savings_freq": 0.05,
        "max_neg_savings_streak": 1,
        "city_tier_code": 2,
    }

    test_medium = {
        "avg_income": 78011.12,
        "income_volatility": 0.2008,
        "income_growth_rate": 0.0170,
        "expense_ratio_mean": 0.7057,
        "expense_volatility": 0.1016,
        "irregular_freq": 0.1085,
        "avg_irregular_amt": 4944.69,
        "savings_volatility": 0.1371,
        "neg_savings_freq": 0.0947,
        "max_neg_savings_streak": 2,
        "city_tier_code": 1,
    }

    test2 = {
        "avg_income": 120000,
        "income_volatility": 0.05,
        "income_growth_rate": 0.08,
        "expense_ratio_mean": 0.35,
        "expense_volatility": 0.04,
        "irregular_freq": 0.04,
        "avg_irregular_amt": 3000,
        "savings_volatility": 0.05,
        "neg_savings_freq": 0.00,
        "max_neg_savings_streak": 0,
        "city_tier_code": 3,
    }

    test3 = {
        "avg_income": 18000,
        "income_volatility": 0.40,
        "income_growth_rate": -0.02,
        "expense_ratio_mean": 0.95,
        "expense_volatility": 0.15,
        "irregular_freq": 0.30,
        "avg_irregular_amt": 4000,
        "savings_volatility": 0.30,
        "neg_savings_freq": 1.00,
        "max_neg_savings_streak": 24,
        "city_tier_code": 1,
    }

    test4 = {
        "avg_income": 22000,
        "income_volatility": 0.20,
        "income_growth_rate": 0.00,
        "expense_ratio_mean": 0.99,
        "expense_volatility": 0.05,
        "irregular_freq": 0.15,
        "avg_irregular_amt": 1500,
        "savings_volatility": 0.08,
        "neg_savings_freq": 0.50,
        "max_neg_savings_streak": 6,
        "city_tier_code": 2,
    }

    test5 = {
        "avg_income": 30000,
        "income_volatility": 0.15,
        "income_growth_rate": 0.01,
        "expense_ratio_mean": 0.60,
        "expense_volatility": 0.10,
        "irregular_freq": 0.08,
        "avg_irregular_amt": 1800,
        "savings_volatility": 0.12,
        "neg_savings_freq": 0.10,
        "max_neg_savings_streak": 2,
        # Intentionally omitted: city_tier_code
    }

    test_cases = [
        ("TEST 1 — Normal user", test1, "Low"),
        ("TEST 1B — Moderate-risk profile", test_medium, "Medium"),
        ("TEST 2 — High income / low expense", test2, "Low"),
        ("TEST 3 — All negative savings", test3, "High"),
        ("TEST 4 — Zero savings", test4, "High"),
        ("TEST 5 — Missing field", test5, "ERROR"),
    ]

    all_passed = True

    for name, case, expected in test_cases:
        print(f"\n{name}")
        print("-" * 50)
        result = predict_risk(case)
        print(result)

        if expected == "ERROR":
            if "error" in result:
                print("✅ PASS — validation caught missing field correctly")
            else:
                print("❌ FAIL — should have returned error for missing field")
                all_passed = False

        elif expected is not None:
            if result.get("risk_label") == expected:
                print(f"✅ PASS — correctly predicted {expected}")
            else:
                print(f"❌ FAIL — expected {expected}, got {result.get('risk_label')}")
                all_passed = False
        else:
            if "risk_label" in result and result["risk_label"] in ["High", "Medium", "Low"]:
                print(f"✅ PASS — returned valid label: {result['risk_label']}")
            else:
                print("❌ FAIL — invalid or missing risk_label in result")
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED — Phase 2 is complete.")
    else:
        print("SOME TESTS FAILED — Fix issues before moving to Phase 3.")
    print("=" * 60)
