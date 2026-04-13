import os
import random

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from pipeline_contract import (
    FEATURE_COLS_PATH,
    FEATURE_IMPORTANCES_PATH,
    FEATURES_PATH,
    LABEL_ENCODER_PATH,
    LABELED_FEATURES_PATH,
    MODEL_FEATURE_COLS,
    MODEL_PATH,
    REQUIRED_FEATURE_COLS,
    SCALER_PATH,
    require_columns,
    save_feature_cols,
)


random.seed(42)
np.random.seed(42)


def main():
    print("[risk_model] Starting...")

    df = pd.read_csv(FEATURES_PATH)
    require_columns(df, REQUIRED_FEATURE_COLS, str(FEATURES_PATH))

    if "severe_overspend_freq" not in df.columns:
        df["severe_overspend_freq"] = 1 - norm.cdf(
            0.90,
            loc=df["expense_ratio_mean"],
            scale=df["expense_volatility"] + 1e-6,
        )

    score_components = [
        "neg_savings_freq",
        "expense_ratio_mean",
        "severe_overspend_freq",
        "income_volatility",
        "savings_volatility",
    ]
    require_columns(df, score_components, "risk scoring inputs")

    mm = MinMaxScaler()
    normalized = pd.DataFrame(
        mm.fit_transform(df[score_components]),
        columns=[f"{c}_norm" for c in score_components],
        index=df.index,
    )

    df["risk_score"] = (
        0.30 * normalized["neg_savings_freq_norm"]
        + 0.25 * normalized["expense_ratio_mean_norm"]
        + 0.20 * normalized["severe_overspend_freq_norm"]
        + 0.15 * normalized["income_volatility_norm"]
        + 0.10 * normalized["savings_volatility_norm"]
    )

    q33 = df["risk_score"].quantile(0.33)
    q66 = df["risk_score"].quantile(0.66)
    df["risk_label"] = np.where(
        df["risk_score"] <= q33,
        "Low",
        np.where(df["risk_score"] <= q66, "Medium", "High"),
    )

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(LABELED_FEATURES_PATH, index=False)

    feature_cols = MODEL_FEATURE_COLS
    require_columns(df, feature_cols, "model features")

    X = df[feature_cols]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    confusion_matrices = {}
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train_enc)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test_enc, y_pred)
        macro_f1 = f1_score(y_test_enc, y_pred, average="macro")
        cm = confusion_matrix(y_test_enc, y_pred)

        results[model_name] = {
            "model": model,
            "accuracy": acc,
            "macro_f1": macro_f1,
        }
        confusion_matrices[model_name] = cm

    sorted_models = sorted(
        results.items(),
        key=lambda item: (
            item[1]["macro_f1"],
            1 if item[0] == "Random Forest" else 0,
        ),
        reverse=True,
    )
    best_model_name, best_info = sorted_models[0]
    best_model = best_info["model"]

    print("[risk_model] Model comparison:")
    for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        print(
            f"  {name:<22} acc={results[name]['accuracy']:.4f} "
            f"f1={results[name]['macro_f1']:.4f}"
        )

    print(f"[risk_model] Best model: {best_model_name}")
    print("[risk_model] Confusion matrices:")
    for model_name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        print(model_name)
        print(confusion_matrices[model_name])

    rf_importances = results["Random Forest"]["model"].feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf_importances,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCES_PATH, index=False)

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    save_feature_cols(feature_cols, FEATURE_COLS_PATH)

    for file_path in [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, FEATURE_COLS_PATH]:
        assert os.path.exists(file_path), f"Missing artifact: {file_path}"

    print("[risk_model] Done. Saved to models/ and data/processed/")


if __name__ == "__main__":
    main()
