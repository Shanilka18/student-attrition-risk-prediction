"""Phase 6 Explainability Orchestrator.

Integrates:
1. Adaptive explainer computation
2. Deterministic narrative generation
3. Cross-fold explanation stability analysis
4. Diagnostic visualizations (bar, beeswarm, dependence)
5. Frontend JSON artifact export
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.explain.narrative import batch_generate_narratives
from src.explain.shap_explainer import compute_shap_explanations
from src.explain.stability import compute_shap_stability_across_folds
from src.explain.visualize import generate_shap_plots
from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir

logger = setup_logger("run_phase6")


def run_phase6_explainability() -> Dict[str, Any]:
    """Execute complete Phase 6 XAI pipeline."""
    logger.info("=== Starting Phase 6: Explainability with SHAP & Deterministic Narratives ===")

    # 1. Compute SHAP explanations on validation split
    shap_data = compute_shap_explanations(dataset_split="val")

    # 2. Load ERI scores for validation students from predictions
    model_name = shap_data["model_name"]
    preds_file = get_artifacts_dir("preds") / f"{model_name}_C_val_preds.npy"
    if preds_file.exists():
        val_probs = np.load(str(preds_file))
        eri_scores = val_probs * 100.0
    else:
        eri_scores = np.full(len(shap_data["sample_ids"]), 50.0)

    # 3. Batch generate student narratives
    narratives = batch_generate_narratives(
        sample_ids=shap_data["sample_ids"],
        eri_scores=eri_scores,
        feature_names=shap_data["feature_names"],
        shap_matrix=shap_data["shap_values"],
    )

    shap_dir = get_artifacts_dir("shap")
    with open(shap_dir / "student_narratives.json", "w", encoding="utf-8") as f:
        json.dump(narratives, f, indent=2)
    logger.info(f"Generated {len(narratives)} deterministic plain-language narratives.")

    # 4. Cross-fold stability audit
    stability_results = compute_shap_stability_across_folds(model_name=model_name, n_folds=5)

    # 5. Diagnostic plots
    plot_paths = generate_shap_plots(
        shap_values=shap_data["shap_values"],
        X_target=shap_data["X_target"],
        feature_names=shap_data["feature_names"],
    )

    # 6. Frontend JSON data export
    frontend_dir = get_artifacts_dir("frontend_data")
    frontend_dir.mkdir(parents=True, exist_ok=True)

    with open(frontend_dir / "global_importance.json", "w", encoding="utf-8") as f:
        json.dump(shap_data["global_importance"], f, indent=2)

    with open(frontend_dir / "student_explanations.json", "w", encoding="utf-8") as f:
        json.dump(narratives, f, indent=2)

    with open(frontend_dir / "shap_stability.json", "w", encoding="utf-8") as f:
        json.dump(stability_results, f, indent=2)

    logger.info("=== Phase 6 Complete: Artifacts Exported ===")
    return {
        "status": "success",
        "model_name": model_name,
        "explainer_type": shap_data["explainer_type"],
        "num_students_explained": len(narratives),
        "mean_spearman_stability": stability_results["mean_spearman_rank_correlation"],
        "plot_paths": plot_paths,
    }
