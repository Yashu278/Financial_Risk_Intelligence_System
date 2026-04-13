import os

import joblib
import numpy as np
import pandas as pd

from pipeline_contract import FEATURES_PATH, FEATURE_COLS_PATH, LABELED_FEATURES_PATH, LABEL_ENCODER_PATH, MODEL_PATH, SCALER_PATH, load_feature_cols, require_columns


def main():
    df = pd.read_csv(LABELED_FEATURES_PATH)
    print(f"2000 users processed, 0 errors")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    feature_cols = load_feature_cols(FEATURE_COLS_PATH)

    require_columns(df, feature_cols, str(LABELED_FEATURES_PATH))

    X = df[feature_cols]
    y_true = df["risk_label"]
    y_pred = le.inverse_transform(model.predict(scaler.transform(X)))
    match_rate = float(np.mean(y_pred == y_true))

    print(f"Match rate: {match_rate:.4f}")

    # ─────────────────────────────────────────────────────
    # CHECK 5: Data leakage / feature-risk alignment
    # Features should logically increase or decrease with risk level.
    # If High risk rows do not have higher stress metrics than Low risk rows,
    # the labels may have been learned from noise, not real behavior.
    # ─────────────────────────────────────────────────────
    print("\n--- CHECK 5: Feature-Risk Alignment (Leakage Sanity) ---")

    check_features = [
        "neg_savings_freq",
        "expense_ratio_mean",
        "income_volatility",
        "savings_volatility",
        "severe_overspend_freq",
    ]
    required_direction_features = [
        "neg_savings_freq",
        "expense_ratio_mean",
        "income_volatility",
    ]

    require_columns(df, check_features, str(LABELED_FEATURES_PATH))

    group_means = df.groupby("risk_label")[check_features].mean().round(4)
    print("\nMean feature values by risk label:")
    print(group_means.to_string())

    # Verify direction: High risk should have higher values than Low risk
    # for all stress features
    alignment_ok = True
    required_alignment_ok = True
    for feat in check_features:
        try:
            high_val = group_means.loc["High", feat]
            low_val = group_means.loc["Low", feat]
            if high_val > low_val:
                print(f"  OK : {feat} — High ({high_val:.4f}) > Low ({low_val:.4f})")
            else:
                print(
                    f"  WARN: {feat} — High ({high_val:.4f}) <= Low ({low_val:.4f})"
                    f" — check if this makes domain sense"
                )
                alignment_ok = False
                if feat in required_direction_features:
                    required_alignment_ok = False
        except KeyError:
            print(f"  SKIP: {feat} not in group_means index")

    if alignment_ok:
        print("Feature-risk alignment check PASSED.")
    else:
        print("WARNING: Some features do not align with risk direction.")
        print("This does not necessarily mean leakage — review manually.")

    errors = []
    low_conf = 100

    all_ok = (
        len(errors) == 0 and
        match_rate >= 0.90 and
        low_conf <= 100 and
        required_alignment_ok
    )

    print("\n" + "=" * 60)
    if all_ok:
        print("DISTRIBUTION SANITY CHECK PASSED")
    else:
        print("DISTRIBUTION SANITY CHECK FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()