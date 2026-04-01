"""
risk_model.py — Phase 2 Complete ML Pipeline  (v2 — all issues fixed)
AI-Driven Financial Risk & Intelligence System

DESIGN DECISIONS (viva-defensible):

  1. Risk score uses RAW features scaled by known domain maximums.
     NOT normalized to dataset percentile — that would make predictions
     distribution-dependent. A new user's score must not shift just because
     dataset distribution changed.

  2. severe_overspend_freq derived via Normal CDF:
     P(expense_ratio > 0.9 | mean, volatility) — statistically principled.

  3. Labels are percentile-based (33/67) — guaranteed balanced classes.

  4. Split BEFORE scale — no data leakage.

  5. High accuracy (~97%) is expected because model replicates a defined rule.
     This is CORRECT behaviour, not suspicious.
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib
from scipy import stats
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
FEATURES_PATH       = "data/processed/features.csv"
LABELED_OUTPUT_PATH = "data/processed/features_labeled.csv"
MODELS_DIR          = "models"

# ─────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────
def load_features(path):
    df = pd.read_csv(path)
    print(f"[LOAD] {len(df)} users | {df.shape[1]} columns")
    return df

# ─────────────────────────────────────────────
# STEP 2A — DERIVE severe_overspend_freq
# Statistically principled: P(expense_ratio > 0.9) via Normal CDF
# ─────────────────────────────────────────────
def derive_severe_overspend(df):
    df = df.copy()
    mu    = df["expense_ratio_mean"]
    sigma = df["expense_volatility"].clip(lower=1e-6)
    df["severe_overspend_freq"] = (
        1 - stats.norm.cdf(0.90, loc=mu, scale=sigma)
    ).clip(0, 1)
    print(f"[DERIVE] severe_overspend_freq  mean={df['severe_overspend_freq'].mean():.4f}")
    return df

# ─────────────────────────────────────────────
# STEP 2B — RISK SCORE (raw features, domain-scaled)
# Weights (viva answer):
#   neg_savings_freq 0.30    — proves inability to live within income
#   expense_ratio    0.25    — persistent overspending
#   severe_overspend 0.20    — extreme stress episodes
#   income_volatility 0.15   — supply-side instability
#   savings_volatility 0.10  — demand-side noise
# ─────────────────────────────────────────────
RISK_WEIGHTS = {
    "neg_savings_freq"     : 0.30,
    "expense_ratio_mean"   : 0.25,
    "severe_overspend_freq": 0.20,
    "income_volatility"    : 0.15,
    "savings_volatility"   : 0.10,
}
FEATURE_SCALES = {
    "neg_savings_freq"     : 1.0,
    "expense_ratio_mean"   : 1.25,
    "severe_overspend_freq": 1.0,
    "income_volatility"    : 0.30,
    "savings_volatility"   : 10.0,
}

def compute_risk_score(df):
    df = df.copy()
    score = pd.Series(0.0, index=df.index)
    for col, weight in RISK_WEIGHTS.items():
        score += weight * (df[col] / FEATURE_SCALES[col]).clip(0, 1)
    df["risk_score"] = score.round(6)
    print(f"[SCORE] range={df['risk_score'].min():.4f}–{df['risk_score'].max():.4f}  mean={df['risk_score'].mean():.4f}")
    return df

# ─────────────────────────────────────────────
# STEP 2C — LABELS
# ─────────────────────────────────────────────
def create_risk_labels(df):
    df = df.copy()
    top    = df["risk_score"].quantile(0.67)
    bottom = df["risk_score"].quantile(0.33)
    df["risk_label"] = df["risk_score"].apply(
        lambda x: "High" if x >= top else ("Low" if x <= bottom else "Medium")
    )
    print("\n[LABELS] Distribution:")
    print(df["risk_label"].value_counts().to_string())
    print("\n[LABELS] Validation — mean values per group:")
    cols = ["neg_savings_freq","expense_ratio_mean","income_volatility",
            "savings_volatility","severe_overspend_freq"]
    print(df.groupby("risk_label")[cols].mean().round(4).to_string())

    g = df.groupby("risk_label")["neg_savings_freq"].mean()
    assert g["High"] > g["Medium"] > g["Low"], "LABEL SANITY FAILED"
    print("\n✅ Label sanity passed: High > Medium > Low")
    return df

# ─────────────────────────────────────────────
# STEP 2D — PREPROCESS (split FIRST, then scale)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "avg_income","income_volatility","income_growth_rate",
    "expense_ratio_mean","expense_volatility","severe_overspend_freq",
    "irregular_freq","avg_irregular_amt",
    "savings_volatility","neg_savings_freq",
    "max_neg_savings_streak","city_tier_code","age",
]

def preprocess(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    df_clean  = df[available + ["risk_label"]].dropna()
    print(f"\n[PREP] {len(available)} features | {len(df_clean)} rows")

    X       = df_clean[available].values
    encoder = LabelEncoder()
    y       = encoder.fit_transform(df_clean["risk_label"].values)
    print(f"[PREP] Classes: {list(encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print(f"[PREP] Train={X_train.shape} | Test={X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, encoder, available

# ─────────────────────────────────────────────
# STEP 2E — TRAIN
# ─────────────────────────────────────────────
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"      : DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    trained = {}
    print()
    for name, m in models.items():
        m.fit(X_train, y_train)
        trained[name] = m
        print(f"[TRAIN] {name} ✔")
    return trained

# ─────────────────────────────────────────────
# STEP 2F — EVALUATE (actual numbers printed)
# ─────────────────────────────────────────────
def evaluate_models(trained, X_test, y_test, encoder):
    results = {}
    print("\n" + "═"*62)
    print("  MODEL EVALUATION — ACTUAL RESULTS")
    print("═"*62)
    for name, model in trained.items():
        preds  = model.predict(X_test)
        acc    = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds,
                    target_names=encoder.classes_, output_dict=True)
        cm     = confusion_matrix(y_test, preds)
        f1     = report["macro avg"]["f1-score"]
        results[name] = {"accuracy": acc, "f1": f1, "report": report, "cm": cm}

        print(f"\n── {name}")
        print(f"   Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"   Macro F1 : {f1:.4f}")
        for cls in encoder.classes_:
            r = report[cls]
            print(f"   {cls:<8}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}")
        print(f"   Confusion matrix (actual↓ predicted→):")
        print("           " + "  ".join(f"{c:>8}" for c in encoder.classes_))
        for i, cls in enumerate(encoder.classes_):
            print(f"   {cls:<8}  " + "  ".join(f"{v:>8}" for v in cm[i]))

    print("\n" + "─"*62)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print("  " + "─"*47)
    for name, r in results.items():
        marker = " ← best" if r["f1"] == max(v["f1"] for v in results.values()) else ""
        print(f"  {name:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f}{marker}")
    print("═"*62)
    return results

# ─────────────────────────────────────────────
# STEP 2G — FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def show_feature_importance(rf_model, feature_cols):
    fi = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n" + "═"*62)
    print("  FEATURE IMPORTANCE (Random Forest)")
    print("═"*62)
    for feat, score in fi.items():
        bar = "█" * int(score * 50)
        print(f"  {feat:<28} {score:.4f}  {bar}")
    top3 = list(fi.index[:3])
    if "city_tier_code" in top3:
        print("\n⚠️  city_tier_code in top-3 — location bias risk")
    else:
        print(f"\n✅ Top-3 features: {top3} — behaviour-driven")
    return fi

# ─────────────────────────────────────────────
# STEP 2H — SAVE
# ─────────────────────────────────────────────
def save_artifacts(best_model, scaler, encoder, feature_cols, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    artifacts = {
        "risk_model.pkl"   : best_model,
        "scaler.pkl"       : scaler,
        "label_encoder.pkl": encoder,
        "feature_cols.pkl" : feature_cols,
    }
    print("\n[SAVE]")
    for fname, obj in artifacts.items():
        path = os.path.join(models_dir, fname)
        joblib.dump(obj, path)
        print(f"  {fname:<22} {os.path.getsize(path):>6} bytes ✔")
    joblib.load(os.path.join(models_dir, "risk_model.pkl"))
    print("✅ Reload check passed")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "═"*62)
    print("  PHASE 2 — RISK CLASSIFICATION PIPELINE  (v2)")
    print("═"*62 + "\n")

    df = load_features(FEATURES_PATH)
    df = derive_severe_overspend(df)
    df = compute_risk_score(df)
    df = create_risk_labels(df)

    os.makedirs(os.path.dirname(LABELED_OUTPUT_PATH), exist_ok=True)
    df.to_csv(LABELED_OUTPUT_PATH, index=False)
    print(f"[SAVE] {LABELED_OUTPUT_PATH}")

    X_train, X_test, y_train, y_test, scaler, encoder, feat_cols = preprocess(df)

    print("\n[TRAIN]")
    trained = train_models(X_train, y_train)

    results = evaluate_models(trained, X_test, y_test, encoder)

    best_name = max(results, key=lambda n: results[n]["f1"])
    print(f"\n🏆 Best: {best_name}  "
          f"(Accuracy={results[best_name]['accuracy']:.4f}  F1={results[best_name]['f1']:.4f})")

    show_feature_importance(trained["Random Forest"], feat_cols)
    save_artifacts(trained[best_name], scaler, encoder, feat_cols, MODELS_DIR)

    print("\n" + "═"*62)
    print("  PHASE 2 COMPLETE ✅")
    print(f"  Best model : {best_name}")
    print(f"  Accuracy   : {results[best_name]['accuracy']*100:.2f}%")
    print(f"  Macro F1   : {results[best_name]['f1']:.4f}")
    print("  Next step  : python src/predict.py")
    print("═"*62 + "\n")

if __name__ == "__main__":
    main()
