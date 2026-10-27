"""
Explainability & Risk Index Plot Generator.

Generates:
  1. shap_summary_global.png (Mean absolute SHAP feature ranking)
  2. temporal_attention_heatmap.png (Semester-by-Feature contribution matrix)
  3. feature_deletion_faithfulness.png (Monotonic risk reduction curve)
  4. risk_band_distribution.png (Cohort distribution across Low/Moderate/High/Critical)
  5. waterfall_sample_high_risk.png (Local SHAP contribution waterfall for high-risk student)

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("AcademicIntelligence.ExplainPlots")

# Clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"


def plot_global_shap_importance(
    df_importance: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> Path:
    """Horizontal bar chart of top-N features ranked by mean absolute SHAP."""
    df_plot = df_importance.head(top_n).iloc[::-1]  # Reverse for top-down display

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(
        df_plot["display_name"],
        df_plot["mean_abs_shap"],
        color="#1f77b4",
        edgecolor="black",
        alpha=0.85,
    )

    for bar, pct in zip(bars, df_plot["relative_importance_pct"]):
        width = bar.get_width()
        ax.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Mean |SHAP Value| (Impact on Model Log-Odds)", fontsize=11)
    ax.set_title(
        f"Global Feature Importance (Champion CatBoost — Top {top_n} Factors)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_temporal_attribution_heatmap(
    df_matrix: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Heatmap of attribution across (semester, feature) grid from sequence model."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    cax = ax.imshow(df_matrix.values, cmap="YlOrRd", aspect="auto")

    fig.colorbar(cax, ax=ax, label="Mean Attribution Magnitude (Integrated Gradients)")

    ax.set_xticks(range(len(df_matrix.columns)))
    ax.set_xticklabels(df_matrix.columns, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(len(df_matrix.index)))
    ax.set_yticklabels(df_matrix.index, fontsize=10)

    for i in range(len(df_matrix.index)):
        for j in range(len(df_matrix.columns)):
            val = df_matrix.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="black", fontsize=9)

    ax.set_title("Temporal Feature Attribution Heatmap (Semester x Modality)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_faithfulness_deletion_curve(
    deletion_curve: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Plots risk score reduction as top-k features are progressively neutralized."""
    k_vals = [pt["k_features_neutralized"] for pt in deletion_curve]
    risks = [pt["mean_academic_risk_index"] for pt in deletion_curve]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(k_vals, risks, marker="o", color="#d62728", lw=2.5, markersize=8, label="At-Risk Cohort Mean ARI")

    for x, y in zip(k_vals, risks):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of Top Risk-Increasing Features Neutralized (k)", fontsize=11)
    ax.set_ylabel("Mean Academic Risk Index (0-100)", fontsize=11)
    ax.set_title("Explanation Faithfulness: Feature Deletion Monotonicity Test", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_risk_band_distribution(
    band_summary: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Bar chart showing cohort distribution across clinical risk tiers."""
    counts = band_summary["band_counts"]
    pcts = band_summary["band_percentages"]
    labels = ["Low", "Moderate", "High", "Critical"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#7f0000"]

    y_vals = [counts.get(l, 0) for l in labels]
    p_vals = [pcts.get(l, 0.0) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(labels, y_vals, color=colors, edgecolor="black", alpha=0.85, width=0.55)

    for bar, count, pct in zip(bars, y_vals, p_vals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(y_vals) * 0.015,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Number of Student Observations", fontsize=11)
    ax.set_xlabel("Academic Risk Tier", fontsize=11)
    ax.set_title("Test Cohort Risk Band Distribution (Calibrated Decision Boundaries)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(y_vals) * 1.15)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_local_waterfall_sample(
    sample_factors: List[Dict[str, Any]],
    student_id: str,
    risk_index: float,
    output_path: Path,
) -> Path:
    """Waterfall-style horizontal bar chart of local SHAP attributions for one student."""
    factors = sample_factors[:6][::-1]  # top 6 drivers
    names = [f["display_name"] for f in factors]
    contribs = [f["contribution"] for f in factors]
    colors = ["#d62728" if c > 0 else "#2ca02c" for c in contribs]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bars = ax.barh(names, contribs, color=colors, edgecolor="black", alpha=0.85)

    for bar, val in zip(bars, contribs):
        width = bar.get_width()
        pos_x = width + (0.01 if width >= 0 else -0.01)
        ax.text(
            pos_x,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if width >= 0 else "right",
            fontsize=9,
            fontweight="bold",
        )

    ax.axvline(0, color="black", linestyle="-", lw=1.2)
    ax.set_xlabel("Signed SHAP Value (Impact on Log-Odds)", fontsize=11)
    ax.set_title(
        f"Local Attribution Waterfall: {student_id} (ARI = {risk_index:.1f}/100)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path
