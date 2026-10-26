"""Retraining Engine for User-Provided Survey Datasets.

CLI Tool:
`python -m src.inference.retrain --file my_data.xlsx --outcome attrition [--confirm]`

Workflow:
1. Validates that the uploaded file contains a valid ground-truth binary outcome column.
2. Ingests and cleans dataset using deterministic cleaner.
3. Performs stratified train/val/test split.
4. Trains the model zoo.
5. Re-evaluates leaderboard and selects the winning model.
6. Only overwrites production artifacts if user provides `--confirm` flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.cleaning.cleaner import SurveyCleaner
from src.evaluation.evaluate_phase5 import run_phase5_evaluation
from src.inference.ingestion import align_dataframe_schema, load_raw_file
from src.preprocessing.split import create_grouped_stratified_split
from src.training.run_all import run_all
from src.utils.logging_config import setup_logger

logger = setup_logger("inference_retrain")


def retrain_pipeline_on_file(
    file_path: str,
    outcome_col: str = "attrition",
    confirm_overwrite: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end retraining on a newly uploaded user dataset."""
    logger.info(f"Starting retraining pipeline on file: {file_path}")

    # 1. Load and align file
    df_raw = load_raw_file(file_path)
    df_aligned, schema_report = align_dataframe_schema(df_raw)

    # Validate outcome column presence
    if outcome_col not in df_aligned.columns:
        err_msg = (
            f"Retraining rejected: Uploaded file does not contain the required ground-truth outcome column '{outcome_col}'. "
            "Supervised retraining is impossible without true outcomes. You can still score this file using "
            f"`python -m src.inference.predict --file {file_path}`."
        )
        logger.error(err_msg)
        print(f"\n[ERROR] {err_msg}\n")
        return {"status": "failed", "reason": err_msg}

    # Ensure outcome is binary (0 or 1)
    unique_vals = set(df_aligned[outcome_col].dropna().unique())
    if not unique_vals.issubset({0, 1, 0.0, 1.0}):
        err_msg = f"Retraining rejected: Outcome column '{outcome_col}' must be binary (0 or 1). Found values: {unique_vals}"
        logger.error(err_msg)
        print(f"\n[ERROR] {err_msg}\n")
        return {"status": "failed", "reason": err_msg}

    if not confirm_overwrite:
        warn_msg = (
            "SAFETY CHECK: Retraining requested without --confirm flag. "
            "Pipeline will validate data and show expected steps, but will NOT overwrite production artifacts. "
            "To execute full retraining and artifact update, re-run with: "
            f"python -m src.inference.retrain --file {file_path} --outcome {outcome_col} --confirm"
        )
        logger.warning(warn_msg)
        print(f"\n[WARNING] {warn_msg}\n")
        return {"status": "dry_run", "message": warn_msg}

    logger.info("Executing full retraining cycle with artifact overwrites...")

    # 2. Save raw file to data/raw/
    raw_target = Path("data/raw/user_retrain_data.csv")
    df_aligned.to_csv(raw_target, index=False)

    # 3. Clean dataset
    cleaner = SurveyCleaner()
    df_clean, clean_log = cleaner.clean(raw_target)
    clean_target = Path("data/interim/clean.csv")
    df_clean.to_csv(clean_target, index=False)

    # 4. Stratified Split
    train_ids, val_ids, test_ids = create_grouped_stratified_split(df_clean)
    logger.info(f"Split created: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # 5. Train Model Zoo
    logger.info("Training Model Zoo on new partitions...")
    train_report = run_all(track="C", quick=True, no_tune=True)

    # 6. Evaluate Leaderboard & Parsimony Selection
    logger.info("Evaluating Leaderboard and selecting champion...")
    phase5_res = run_phase5_evaluation()

    logger.info("Retraining completed successfully. Production artifacts updated.")
    return {
        "status": "success",
        "selected_model": phase5_res["selected_model"],
        "training_report": train_report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain ERI Model Pipeline on User Data")
    parser.add_argument("--file", type=str, required=True, help="Path to custom CSV or XLSX dataset")
    parser.add_argument("--outcome", type=str, default="attrition", help="Name of binary outcome column")
    parser.add_argument("--confirm", action="store_true", help="Explicitly confirm overwriting production artifacts")
    args = parser.parse_args()

    retrain_pipeline_on_file(args.file, args.outcome, args.confirm)
