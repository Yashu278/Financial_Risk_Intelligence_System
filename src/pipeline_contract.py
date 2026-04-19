from pathlib import Path

import joblib


RAW_DATA_PATH = Path("data/raw/finance_data.csv")
FEATURES_PATH = Path("data/processed/features.csv")
LABELED_FEATURES_PATH = Path("data/processed/labeled_features.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "risk_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.pkl"
FEATURE_IMPORTANCES_PATH = Path("data/processed/feature_importances.csv")

REQUIRED_FEATURE_COLS = [
    "avg_income",
    "income_volatility",
    "income_growth_rate",
    "expense_ratio_mean",
    "expense_volatility",
    "irregular_freq",
    "avg_irregular_amt",
    "savings_volatility",
    "neg_savings_freq",
    "severe_overspend_freq",
    "max_neg_savings_streak",
    "city_tier_code",
]

DERIVED_FEATURE_COLS: list[str] = []
MODEL_FEATURE_COLS = REQUIRED_FEATURE_COLS


def load_feature_cols(path: Path = FEATURE_COLS_PATH) -> list[str]:
    """Load the canonical model feature order from the saved artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Missing feature columns artifact: {path}")

    feature_cols = joblib.load(path)
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError(f"Invalid feature columns artifact: {path}")

    if not all(isinstance(col, str) and col for col in feature_cols):
        raise ValueError(f"Feature columns artifact must contain only non-empty strings: {path}")

    return feature_cols


def save_feature_cols(feature_cols: list[str], path: Path = FEATURE_COLS_PATH) -> None:
    """Persist the canonical model feature order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_cols, path)


def required_input_fields(feature_cols: list[str] | None = None) -> list[str]:
    """Return the user-supplied fields required for inference."""
    if feature_cols is None:
        feature_cols = MODEL_FEATURE_COLS
    return [col for col in feature_cols if col not in DERIVED_FEATURE_COLS]


def require_columns(df, required_columns: list[str], dataset_name: str) -> None:
    """Fail fast if any required columns are missing from a dataframe."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")
