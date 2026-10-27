"""
Evaluation Plot Generator for Publication & Viva Defense.
Saves high-resolution figures to artifacts/plots/evaluation/:
  1. roc_curves_overlay.png
  2. pr_curves_overlay.png
  3. calibration_curves.png
  4. confusion_matrices.png
  5. metric_heatmap.png
  6. lead_time_distribution.png
  7. risk_index_distribution.png
  8. metric_vs_complexity.png

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger("AcademicIntelligence.EvaluationPlots")

# Clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"


def plot_roc_curves_overlay(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray,
    output_path: Path,
    title: str = "Test Set ROC Curves: Sequence Champion vs Baselines",
) -> Path:
    """Plots overlay of ROC curves for key models with AUROC in legend."""
    plt.figure(figsize=(9, 7), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, label="Chance (AUC = 0.500)")

    for label, (y_prob, meta) in models_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = meta.get("roc_auc", 0.5)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc_val:.3f})")

    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_pr_curves_overlay(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y_true: np.ndarray,
    output_path: Path,
    title: str = "Test Set Precision-Recall Curves (Minority Class)",
) -> Path:
    """Plots overlay of PR curves with baseline prevalence."""
    plt.figure(figsize=(9, 7), dpi=300)
    baseline_prev = float(np.mean(y_true))
    plt.axhline(y=baseline_prev, color="gray", linestyle="--", alpha=0.7, label=f"Prevalence ({baseline_prev:.1%})")

    for label, (y_prob, meta) in models_predictions.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = meta.get("pr_auc", 0.0)
        plt.plot(rec, prec, lw=2, label=f"{label} (PR-AUC = {pr_auc:.3f})")

    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_calibration_curves(
    curves_data: Dict[str, List[Dict[str, float]]],
    output_path: Path,
    title: str = "Probability Calibration Reliability Diagram",
) -> Path:
    """Plots reliability diagrams (predicted confidence vs empirical positive fraction)."""
    plt.figure(figsize=(8, 7), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.8, label="Perfect Calibration")

    for label, points in curves_data.items():
        confs = [p["mean_predicted_prob"] for p in points if p["bin_count"] > 0]
        fracs = [p["fraction_of_positives"] for p in points if p["bin_count"] > 0]
        plt.plot(confs, fracs, marker="o", lw=2, label=label)

    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Observed Attrition Fraction", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="upper left", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_confusion_matrices(
    matrices: Dict[str, Dict[str, int]],
    output_path: Path,
) -> Path:
    """Plots side-by-side confusion matrix heatmaps for champion and key baselines."""
    n_models = len(matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), dpi=300)
    if n_models == 1:
        axes = [axes]

    for ax, (name, cm) in zip(axes, matrices.items()):
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        im = ax.imshow(mat, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Retained", "Attrited"])
        ax.set_yticklabels(["Retained", "Attrited"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Outcome")

        thresh = mat.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{mat[i, j]:,}",
                    ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black",
                    fontweight="bold", fontsize=12
                )

    plt.suptitle("Confusion Matrices (Test Set, Recall-Constrained Thresholds)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_metric_heatmap(
    leaderboard_df: Any,
    output_path: Path,
) -> Path:
    """Plots metric heatmap across all evaluated models."""
    import pandas as pd
    if not isinstance(leaderboard_df, pd.DataFrame):
        leaderboard_df = pd.DataFrame(leaderboard_df)

    # Key columns to display in heatmap
    cols = ["Val ROC-AUC", "PR-AUC", "F1 Score", "Recall", "Brier (Cal)", "Threshold"]
    available_cols = [c for c in cols if c in leaderboard_df.columns]

    if leaderboard_df.empty or not available_cols:
        return output_path

    plot_df = leaderboard_df.set_index("Model Key")[available_cols].head(15)

    plt.figure(figsize=(10, max(6, len(plot_df) * 0.45)), dpi=300)
    plt.imshow(plot_df.values, cmap="YlGnBu", aspect="auto")
    plt.colorbar(label="Normalized Metric Value")
    plt.xticks(range(len(available_cols)), available_cols, rotation=30, ha="right", fontsize=10)
    plt.yticks(range(len(plot_df)), plot_df.index, fontsize=10)

    for i in range(len(plot_df)):
        for j in range(len(available_cols)):
            val = plot_df.iloc[i, j]
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=8)

    plt.title("Performance Metric Heatmap (Top 15 Models)", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_lead_time_distribution(
    lead_times: List[float],
    output_path: Path,
    champion_name: str = "Champion Model",
) -> Path:
    """Plots histogram of lead times in semesters before actual attrition."""
    plt.figure(figsize=(8, 5.5), dpi=300)
    if not lead_times:
        lead_times = [1.0]

    bins = np.arange(-0.5, max(lead_times) + 1.5, 1.0)
    plt.hist(lead_times, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.8, rwidth=0.8)
    avg_lead = np.mean(lead_times)
    plt.axvline(avg_lead, color="red", linestyle="--", lw=2, label=f"Average Lead Time = {avg_lead:.1f} semesters")

    plt.xlabel("Lead Time (Semesters Before Actual Attrition)", fontsize=12)
    plt.ylabel("Number of Attriting Students Flagged", fontsize=12)
    plt.title(f"Early Warning Lead-Time Distribution ({champion_name})", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="upper right", fontsize=11, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_risk_index_distribution(
    ari_attrited: np.ndarray,
    ari_retained: np.ndarray,
    output_path: Path,
    threshold_cutoff: float = 5.0,
) -> Path:
    """Plots Academic Risk Index (0-100) distribution for attrited vs retained students."""
    plt.figure(figsize=(9, 6), dpi=300)

    bins = np.linspace(0.0, 100.0, 40)
    plt.hist(ari_retained, bins=bins, density=True, alpha=0.6, color="green", edgecolor="black", label="Retained Students")
    plt.hist(ari_attrited, bins=bins, density=True, alpha=0.6, color="red", edgecolor="black", label="Attrited Students")
    plt.axvline(threshold_cutoff, color="black", linestyle="--", lw=2, label=f"Operational Threshold (ARI = {threshold_cutoff:.1f})")

    plt.xlabel("Academic Risk Index (0 - 100)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title("Academic Risk Index Separation: Attrited vs Retained Cohort", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="upper right", fontsize=11, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_efficiency_vs_performance(
    efficiency_data: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Scatter plot: Inference Latency per Student vs Test AUROC."""
    plt.figure(figsize=(9, 6), dpi=300)

    for item in efficiency_data:
        lat = item["inference_time_ms_per_student"]
        auc = item["roc_auc"]
        name = item["model_key"]
        cat = item["category"]
        color = "#1f77b4" if cat == "E" else ("#ff7f0e" if cat == "C" else "#2ca02c")
        plt.scatter(lat, auc, color=color, s=80, alpha=0.85)
        plt.annotate(name, (lat, auc), textcoords="offset points", xytext=(5, 2), fontsize=8)

    # Proposal NFR1 requirement: < 2,000 ms
    plt.axvline(2000.0, color="red", linestyle=":", lw=1.5, label="Proposal NFR1 Constraint (2,000 ms)")

    plt.xlabel("Inference Latency per Student (ms)", fontsize=12)
    plt.ylabel("Test ROC-AUC", fontsize=12)
    plt.title("Model Efficiency Frontier: Accuracy vs Inference Latency", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path
