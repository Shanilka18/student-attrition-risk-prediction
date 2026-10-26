"""SHAP Visualization and Visual Diagnostic Plots.

Generates:
1. Global mean |SHAP| feature importance bar chart
2. Summary beeswarm plot
3. Dependence scatter plots for Top-5 features
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir

logger = setup_logger("shap_plots")


def generate_shap_plots(
    shap_values: np.ndarray,
    X_target: np.ndarray,
    feature_names: List[str],
    figures_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Generate and save publication-quality SHAP diagnostic charts."""
    if figures_dir is None:
        figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    # 1. Global Mean |SHAP| Bar Chart
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:15]  # Top 15 features
    top_feats = [feature_names[i] for i in top_indices]
    top_vals = [mean_abs[i] for i in top_indices]

    plt.figure(figsize=(9, 6))
    plt.barh(range(len(top_feats)), top_vals[::-1], color="#1f77b4", edgecolor="none")
    plt.yticks(range(len(top_feats)), top_feats[::-1], fontsize=9)
    plt.xlabel("Mean |SHAP value| (average impact on model risk output)", fontsize=10)
    plt.title("Global Feature Importance (SHAP Attribution)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    bar_path = figures_dir / "shap_feature_importance_bar.png"
    plt.savefig(bar_path, dpi=200)
    plt.close()
    plot_paths["bar_chart"] = str(bar_path)

    # 2. Beeswarm Plot
    try:
        plt.figure(figsize=(10, 7))
        # Create an Explanation object for shap.plots.beeswarm
        exp = shap.Explanation(
            values=shap_values,
            data=X_target,
            feature_names=feature_names,
        )
        shap.plots.beeswarm(exp, max_display=15, show=False)
        plt.title("SHAP Beeswarm: Impact & Directionality Across Students", fontsize=12, fontweight="bold")
        plt.tight_layout()
        bee_path = figures_dir / "shap_beeswarm.png"
        plt.savefig(bee_path, dpi=200)
        plt.close()
        plot_paths["beeswarm"] = str(bee_path)
    except Exception as e:
        logger.warning(f"Could not generate standard beeswarm plot: {e}")

    # 3. Dependence Plots for Top 5 features
    dep_paths = []
    for idx in top_indices[:5]:
        feat_name = feature_names[idx]
        try:
            plt.figure(figsize=(7, 4.5))
            plt.scatter(
                X_target[:, idx],
                shap_values[:, idx],
                alpha=0.6,
                c="#2ca02c",
                edgecolors="none",
                s=20,
            )
            plt.xlabel(f"Feature Value: {feat_name}", fontsize=10)
            plt.ylabel(f"SHAP Value for {feat_name}", fontsize=10)
            plt.title(f"SHAP Dependence: {feat_name}", fontsize=11, fontweight="bold")
            plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            plt.tight_layout()
            clean_name = feat_name.replace("/", "_").replace(" ", "_")
            p = figures_dir / f"shap_dependence_{clean_name}.png"
            plt.savefig(p, dpi=180)
            plt.close()
            dep_paths.append(str(p))
        except Exception as e:
            logger.warning(f"Failed dependence plot for {feat_name}: {e}")

    plot_paths["dependence_plots"] = dep_paths
    logger.info(f"SHAP visualizations generated in {figures_dir}")
    return plot_paths
