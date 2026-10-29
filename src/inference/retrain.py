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
from typing import Any

from src.cleaning.cleaner import EnvironmentalFinancialCleaner
from src.preprocessing.pipeline import ALL_ALLOWED_FEATURES
from src.utils.logging_config import setup_logger

logger = setup_logger("inference_retrain")


def retrain_pipeline_on_file(
    file_path: str,
    outcome_col: str = "at_risk",
    confirm_overwrite: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end retraining on a newly uploaded user dataset."""
    logger.info(f"Starting retraining pipeline on file: {file_path}")

    # 1. Clean with the shared EF pipeline and persist its audit report.
    cleaner = EnvironmentalFinancialCleaner()
    clean_result = cleaner.clean(file_path, mode="training")
    saved_clean = cleaner.write_result(clean_result)
    df_clean = saved_clean.data
    if saved_clean.report["traffic_light"]["status"] == "red":
        err_msg = (
            "Retraining rejected because the EF cleaning report contains blocking "
            f"issues: {saved_clean.report['traffic_light']['messages']}"
        )
        logger.error(err_msg)
        return {
            "status": "failed",
            "reason": err_msg,
            "clean_parquet": str(saved_clean.parquet_path),
            "quality_report": str(saved_clean.report_path),
        }

    # Validate outcome column presence
    if outcome_col not in df_clean.columns:
        err_msg = (
            f"Retraining rejected: Uploaded file does not contain the required ground-truth outcome column '{outcome_col}'. "
            "Supervised retraining is impossible without true outcomes. You can still score this file using "
            f"`python -m src.inference.predict --file {file_path}`."
        )
        logger.error(err_msg)
        print(f"\n[ERROR] {err_msg}\n")
        return {
            "status": "failed",
            "reason": err_msg,
            "clean_parquet": str(saved_clean.parquet_path),
            "quality_report": str(saved_clean.report_path),
        }

    # Ensure outcome is binary (0 or 1)
    unique_vals = set(df_clean[outcome_col].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        err_msg = f"Retraining rejected: Outcome column '{outcome_col}' must be binary (0 or 1). Found values: {unique_vals}"
        logger.error(err_msg)
        print(f"\n[ERROR] {err_msg}\n")
        return {
            "status": "failed",
            "reason": err_msg,
            "clean_parquet": str(saved_clean.parquet_path),
            "quality_report": str(saved_clean.report_path),
        }

    missing_features = sorted(set(ALL_ALLOWED_FEATURES) - set(df_clean.columns))
    incompatibility = (
        "The EF questionnaire does not match the currently deployed model feature "
        "contract, which has not yet been migrated."
    )
    if missing_features:
        incompatibility += " Missing model features: " + ", ".join(missing_features)
    if confirm_overwrite:
        logger.error(incompatibility)
        return {
            "status": "failed",
            "reason": incompatibility,
            "clean_parquet": str(saved_clean.parquet_path),
            "quality_report": str(saved_clean.report_path),
        }

    warn_msg = (
        "EF data passed cleaning and outcome validation, but retraining is "
        f"unavailable until the model feature contract is migrated. {incompatibility}"
    )
    logger.warning(warn_msg)
    return {
        "status": "dry_run",
        "message": warn_msg,
        "clean_parquet": str(saved_clean.parquet_path),
        "quality_report": str(saved_clean.report_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate EF survey data for model retraining"
    )
    parser.add_argument(
        "--file", type=str, required=True, help="Path to custom CSV or XLSX dataset"
    )
    parser.add_argument(
        "--outcome", type=str, default="at_risk", help="Name of binary outcome column"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Explicitly confirm overwriting production artifacts",
    )
    args = parser.parse_args()

    retrain_pipeline_on_file(args.file, args.outcome, args.confirm)
