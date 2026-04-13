from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from data_generation import generate_data
from feature_engineering import engineer_features
from risk_model import main as run_risk_model
from validate_phase2 import main as run_validate_phase2
from pipeline_contract import (
    FEATURE_COLS_PATH,
    FEATURE_IMPORTANCES_PATH,
    FEATURES_PATH,
    LABELED_FEATURES_PATH,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    SCALER_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

GENERATED_FILES = [
    RAW_DATA_PATH,
    PROJECT_ROOT / "data/raw/finance_data_full.csv",
    FEATURES_PATH,
    LABELED_FEATURES_PATH,
    FEATURE_IMPORTANCES_PATH,
    MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    FEATURE_COLS_PATH,
]

STAGES = [
    ("data_generation", generate_data),
    ("feature_engineering", engineer_features),
    ("risk_model", run_risk_model),
    ("validate_phase2", run_validate_phase2),
]


def clean_outputs() -> None:
    for path in GENERATED_FILES:
        abs_path = PROJECT_ROOT / path
        if abs_path.is_file():
            abs_path.unlink()
        elif abs_path.is_dir():
            shutil.rmtree(abs_path)


def run_stage(stage_name: str, stage_func) -> dict:
    started_at = datetime.utcnow().isoformat() + "Z"
    stage_func()
    finished_at = datetime.utcnow().isoformat() + "Z"

    return {
        "stage": stage_name,
        "callable": stage_func.__name__,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the financial risk pipeline.")
    parser.add_argument("--clean", action="store_true", help="Remove generated artifacts before running.")
    args = parser.parse_args()

    if args.clean:
        clean_outputs()

    manifest = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "clean": bool(args.clean),
        "stages": [],
    }

    try:
        for stage_name, stage_func in STAGES:
            manifest["stages"].append(run_stage(stage_name, stage_func))
    except Exception as exc:
        manifest["finished_at"] = datetime.utcnow().isoformat() + "Z"
        manifest["status"] = f"failed: {exc}"
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path = LOGS_DIR / f"pipeline_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise

    manifest["finished_at"] = datetime.utcnow().isoformat() + "Z"
    manifest["status"] = "ok"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = LOGS_DIR / f"pipeline_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Pipeline completed. Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
