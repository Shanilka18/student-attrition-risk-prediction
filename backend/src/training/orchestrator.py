"""
Training Pipeline Orchestrator.
Coordinates model zoo training, validation, multi-seed aggregation,
progress logging, resumability, and validation leaderboard summary creation.

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.src.config import config
from backend.src.models.base import ModelRegistry
from backend.src.training.data_loader import TrainingDataLoader
from backend.src.training.engine import train_and_validate_model

logger = logging.getLogger("AcademicIntelligence.Orchestrator")

BUDGET_CONFIGS = {
    "quick": {
        "tabular_trials": 3,
        "deep_trials": 2,
        "deep_epochs": 3,
        "cv_folds": 2,
    },
    "standard": {
        "tabular_trials": 10,
        "deep_trials": 4,
        "deep_epochs": 15,
        "cv_folds": 3,
    },
    "full": {
        "tabular_trials": 25,
        "deep_trials": 8,
        "deep_epochs": 40,
        "cv_folds": 5,
    },
}


def run_training_pipeline(
    budget: str = "quick",
    seeds: Optional[List[int]] = None,
    threshold_strategy: str = "recall_constrained",
    min_recall: float = 0.80,
    calibration_method: str = "platt",
    target_models: Optional[List[str]] = None,
    target_categories: Optional[List[str]] = None,
    force: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Main entry point for `make train` / `python run.py train`.
    Trains, tunes, calibrates, and benchmarks every registered model on the validation split.
    """
    if seeds is None:
        seeds = [42, 123, 999]

    budget_settings = BUDGET_CONFIGS.get(budget.lower(), BUDGET_CONFIGS["quick"])
    logger.info(
        f"Starting Training Pipeline | Budget: {budget.upper()} | Seeds: {seeds} | "
        f"Threshold Strategy: {threshold_strategy} (min_recall={min_recall}) | "
        f"Calibration: {calibration_method} | Force Retrain: {force}"
    )

    # 1. Initialize Data Loader and Partitions
    loader = TrainingDataLoader()
    cv_folds = loader.get_train_cv_folds(n_splits=budget_settings["cv_folds"], seed=seeds[0])

    # 2. Select Models to Train
    all_registered = ModelRegistry.list_all()
    models_to_run = []

    for item in all_registered:
        k = item["key"]
        cat = item["category"]

        if target_models and k not in target_models:
            continue
        if target_categories and cat not in target_categories:
            continue
        models_to_run.append(item)

    logger.info(f"Identified {len(models_to_run)} models to process across Categories A-F.")

    # Cache dataset representations to prevent disk thrashing
    cached_data: Dict[str, Any] = {}
    cached_data["tabular"] = loader.load_tabular_data()
    cached_data["static"] = loader.load_static_data()
    cached_data["sequence"] = loader.load_sequence_data()

    results: List[Dict[str, Any]] = []

    # 3. Process Each Model
    for idx, m_info in enumerate(models_to_run, 1):
        m_key = m_info["key"]
        cat = m_info["category"]
        is_seq = m_info["is_sequence_model"]

        logger.info(f"\n[{idx}/{len(models_to_run)}] Processing Model: {m_key} (Category {cat})...")

        if progress_callback:
            try:
                progress_callback({
                    "phase": "training",
                    "current_model": m_key,
                    "models_done": idx - 1,
                    "models_total": len(models_to_run),
                })
            except Exception as pe:
                logger.debug(f"Progress callback non-fatal exception: {pe}")

        # Select data representation
        if cat == "F":
            m_data = cached_data["static"]
        elif cat == "E":
            m_data = cached_data["sequence"]
        else:
            m_data = cached_data["tabular"]

        # Determine Optuna trial budget
        if cat in ("D", "E") or is_seq:
            n_trials = budget_settings["deep_trials"]
            d_epochs = budget_settings["deep_epochs"]
        else:
            n_trials = budget_settings["tabular_trials"]
            d_epochs = 0

        # Execute single model training
        summary = train_and_validate_model(
            model_key=m_key,
            data=m_data,
            cv_folds=cv_folds,
            category=cat,
            n_tuning_trials=n_trials,
            deep_epochs=d_epochs,
            seeds=seeds,
            threshold_strategy=threshold_strategy,
            min_recall=min_recall,
            calibration_method=calibration_method,
            force=force,
        )
        results.append(summary)

        if progress_callback:
            try:
                progress_callback({
                    "phase": "training",
                    "current_model": m_key,
                    "models_done": idx,
                    "models_total": len(models_to_run),
                })
            except Exception as pe:
                logger.debug(f"Progress callback non-fatal exception: {pe}")

    # 4. Generate Validation Summary Table & Artifacts
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = config.metrics_dir / "validation_summary.json"
    summary_csv_path = config.metrics_dir / "validation_summary.csv"
    summary_md_path = config.metrics_dir / "validation_summary.md"

    table_rows = []
    for r in results:
        status = r.get("status", "UNKNOWN")
        if status == "COMPLETED":
            vm = r["validation_metrics"]
            sm = r["stability_summary"]
            th = r["threshold"]
            cal = r["calibration"]
            table_rows.append({
                "Model Key": r["model_key"],
                "Category": r["category"],
                "Category Label": r.get("category_label", ""),
                "Proposal Role": "Proposal Model" if r.get("is_proposal_model") else "Baseline",
                "Status": "COMPLETED",
                "Val ROC-AUC": vm["roc_auc"],
                "AUC Std (Seeds)": sm["roc_auc_std"],
                "PR-AUC": vm["pr_auc"],
                "F1 Score": vm["f1_score"],
                "Recall": vm["recall"],
                "Precision": vm["precision"],
                "Brier (Raw)": cal["brier_score_before"],
                "Brier (Cal)": cal["brier_score_after"],
                "Threshold": th["optimal_threshold"],
                "Runtime (s)": r.get("runtime_seconds", 0.0),
            })
        else:
            table_rows.append({
                "Model Key": r["model_key"],
                "Category": r.get("category", "?"),
                "Category Label": "",
                "Proposal Role": "",
                "Status": f"FAILED ({r.get('error_message', 'Error')[:30]}...)",
                "Val ROC-AUC": 0.0,
                "AUC Std (Seeds)": 0.0,
                "PR-AUC": 0.0,
                "F1 Score": 0.0,
                "Recall": 0.0,
                "Precision": 0.0,
                "Brier (Raw)": 1.0,
                "Brier (Cal)": 1.0,
                "Threshold": 0.5,
                "Runtime (s)": 0.0,
            })

    df_summary = pd.DataFrame(table_rows)
    # Sort by Category, then Val ROC-AUC descending
    if not df_summary.empty:
        df_summary = df_summary.sort_values(
            by=["Category", "Val ROC-AUC"], ascending=[True, False]
        ).reset_index(drop=True)

    # Save to disk
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    df_summary.to_csv(summary_csv_path, index=False)

    md_content = "# Validation Leaderboard Summary (Phase 5)\n\n"
    md_content += f"**Budget**: {budget.upper()} | **Seeds**: {seeds} | **Threshold Strategy**: {threshold_strategy}\n\n"
    md_content += df_summary.to_markdown(index=False)
    md_content += "\n\n> [!NOTE]\n"
    md_content += "> The test partition remained strictly untouched throughout Phase 5. All reported metrics are validation-only.\n"

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Print Terminal Table
    print("\n" + "=" * 105)
    print("VALIDATION SUMMARY TABLE — ACADEMIC INTELLIGENCE FRAMEWORK (PHASE 5)")
    print("=" * 105)
    print(df_summary.to_string(index=False))
    print("=" * 105)
    print(f"Artifacts written to: {summary_json_path}")
    print(f"Markdown table written to: {summary_md_path}\n")

    return {
        "status": "COMPLETED",
        "n_models": len(results),
        "n_completed": sum(1 for r in results if r.get("status") == "COMPLETED"),
        "n_failed": sum(1 for r in results if r.get("status") == "FAILED"),
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "summary_md": str(summary_md_path),
    }
