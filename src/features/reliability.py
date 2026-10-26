"""Psychometric Reliability Analysis Module.

Computes:
- Cronbach's alpha per factor.
- Alpha-if-item-deleted per item.
- Corrected item-total correlations (r_it).
- Inter-factor correlation matrix.
- McDonald's omega (composite reliability) via factor loadings.
- Automated pass/fail validation flags (alpha >= 0.70 threshold).
- Exports metrics to /artifacts/metrics/reliability.json and diagnostic plots.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger("features.reliability")

FACTOR_ITEMS: Dict[str, List[str]] = {
    "stress": ["S1", "S2", "S3"],
    "motivation": ["M1", "M2", "M3"],
    "belongingness": ["B1", "B2", "B3"],
    "self_efficacy": ["SE1", "SE2", "SE3"],
    "peer_support": ["PS1", "PS2", "PS3"],
}


def calculate_cronbach_alpha(items_df: pd.DataFrame) -> float:
    """Calculate Cronbach's alpha for a matrix or DataFrame of survey items.

    Formula:
        alpha = (k / (k - 1)) * (1 - sum(s_i^2) / s_total^2)

    Args:
        items_df: DataFrame containing item columns for a single factor.

    Returns:
        float: Calculated Cronbach's alpha coefficient.
    """
    clean_df = items_df.dropna()
    k = clean_df.shape[1]
    if k <= 1 or len(clean_df) < 5:
        return 0.0

    item_variances = clean_df.var(axis=0, ddof=1).sum()
    total_scores = clean_df.sum(axis=1)
    total_variance = total_scores.var(ddof=1)

    if total_variance <= 0:
        return 0.0

    alpha = (k / (k - 1.0)) * (1.0 - (item_variances / total_variance))
    return float(alpha)


def calculate_mcdonald_omega(items_df: pd.DataFrame) -> float:
    """Calculate McDonald's omega composite reliability coefficient.

    Uses a single-factor principal loading approximation:
        omega = (sum(lambda_i))^2 / [ (sum(lambda_i))^2 + sum(1 - lambda_i^2) ]

    Args:
        items_df: DataFrame of items for a single factor.

    Returns:
        float: Calculated McDonald's omega.
    """
    clean_df = items_df.dropna()
    k = clean_df.shape[1]
    if k <= 1 or len(clean_df) < 10:
        return 0.0

    # Correlation matrix of items
    corr_matrix = clean_df.corr().values
    # Eigen decomposition to obtain principal factor loadings
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
    # Principal eigenvector corresponding to largest eigenvalue
    largest_idx = np.argmax(eigenvals)
    first_eigenval = eigenvals[largest_idx]
    if first_eigenval <= 0:
        return 0.0

    loadings = eigenvecs[:, largest_idx] * np.sqrt(first_eigenval)
    # Ensure positive orientation
    if np.sum(loadings) < 0:
        loadings = -loadings

    loadings = np.clip(loadings, 0.05, 0.99)
    sum_loadings = np.sum(loadings)
    sum_error_var = np.sum(1.0 - (loadings ** 2))

    denom = (sum_loadings ** 2) + sum_error_var
    if denom <= 0:
        return 0.0

    omega = (sum_loadings ** 2) / denom
    return float(omega)


def calculate_item_analysis(
    df: pd.DataFrame,
    items: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute alpha-if-item-deleted and corrected item-total correlation for each item in a scale.

    Args:
        df: Input DataFrame containing the item columns.
        items: List of item column names in the subscale.

    Returns:
        Dict[str, Dict[str, float]]: Mapping of item name to metrics:
            - alpha_if_deleted
            - corrected_item_total_r
            - item_mean
            - item_std
    """
    clean_df = df[items].dropna()
    results: Dict[str, Dict[str, float]] = {}

    for item in items:
        remaining_items = [it for it in items if it != item]
        alpha_deleted = calculate_cronbach_alpha(clean_df[remaining_items])

        # Corrected item-total correlation: corr(item, sum(remaining_items))
        remainder_sum = clean_df[remaining_items].sum(axis=1)
        corr_matrix = np.corrcoef(clean_df[item], remainder_sum)
        r_val = float(corr_matrix[0, 1])

        results[item] = {
            "alpha_if_deleted": round(alpha_deleted, 4),
            "corrected_item_total_r": round(r_val, 4),
            "item_mean": round(float(clean_df[item].mean()), 3),
            "item_std": round(float(clean_df[item].std()), 3),
        }

    return results


def run_full_reliability_analysis(
    df: pd.DataFrame,
    alpha_threshold: float = 0.70,
    factor_mapping: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Execute complete psychometric reliability analysis across all questionnaire factors.

    Args:
        df: Survey DataFrame with item columns.
        alpha_threshold: Minimum acceptable Cronbach's alpha (default 0.70).
        factor_mapping: Mapping of factor names to item lists.

    Returns:
        Dict[str, Any]: Complete reliability audit report.
    """
    if factor_mapping is None:
        factor_mapping = FACTOR_ITEMS

    logger.info("Executing psychometric reliability analysis on N=%d records...", len(df))
    factor_results: Dict[str, Any] = {}
    factor_scores_dict: Dict[str, pd.Series] = {}
    all_passed = True

    for factor_name, items in factor_mapping.items():
        present_items = [it for it in items if it in df.columns]
        if len(present_items) < len(items):
            logger.warning("Factor %s missing expected items: %s", factor_name, items)
            continue

        sub_df = df[present_items]
        alpha = calculate_cronbach_alpha(sub_df)
        omega = calculate_mcdonald_omega(sub_df)
        item_analysis = calculate_item_analysis(df, present_items)

        passed = alpha >= alpha_threshold
        if not passed:
            all_passed = False

        status = "EXCELLENT" if alpha >= 0.80 else ("ACCEPTABLE" if alpha >= 0.70 else "UNACCEPTABLE")

        factor_results[factor_name] = {
            "cronbach_alpha": round(alpha, 4),
            "mcdonald_omega": round(omega, 4),
            "status": status,
            "passed": passed,
            "n_items": len(present_items),
            "item_analysis": item_analysis,
        }

        # Compute factor mean score for inter-factor correlation
        factor_scores_dict[factor_name] = sub_df.mean(axis=1)

    # Inter-factor correlation matrix
    factors_df = pd.DataFrame(factor_scores_dict)
    inter_factor_corr = factors_df.corr().round(4).to_dict()

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(df),
        "alpha_threshold": alpha_threshold,
        "all_factors_passed": all_passed,
        "factors": factor_results,
        "inter_factor_correlation_matrix": inter_factor_corr,
    }

    return report


def export_reliability_artifacts(
    report: Dict[str, Any],
    metrics_path: str = "artifacts/metrics/reliability.json",
    plots_dir: str = "artifacts/plots",
) -> List[str]:
    """Save reliability JSON report and generate diagnostic visualization charts."""
    m_path = Path(metrics_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Reliability report saved to %s", m_path)

    p_dir = Path(plots_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: List[str] = []

    sns.set_theme(style="whitegrid")

    # 1. Cronbach's Alpha & McDonald's Omega Bar Chart
    factors = list(report["factors"].keys())
    alphas = [report["factors"][f]["cronbach_alpha"] for f in factors]
    omegas = [report["factors"][f]["mcdonald_omega"] for f in factors]

    x = np.arange(len(factors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, alphas, width, label="Cronbach's Alpha (α)", color="#3B82F6")
    rects2 = ax.bar(x + width/2, omegas, width, label="McDonald's Omega (ω)", color="#10B981")

    ax.axhline(0.70, color="orange", linestyle="--", linewidth=1.5, label="Acceptable Threshold (0.70)")
    ax.axhline(0.80, color="green", linestyle=":", linewidth=1.5, label="Good/Excellent Threshold (0.80)")
    ax.set_ylabel("Reliability Coefficient", fontsize=12)
    ax.set_title("Psychometric Subscale Internal Consistency Reliability", fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", " ").title() for f in factors], fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2, h + 0.01), ha="center", fontsize=9, weight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2, h + 0.01), ha="center", fontsize=9, weight="bold")

    plt.tight_layout()
    p1 = str(p_dir / "reliability_alpha_bars.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    saved_plots.append(p1)

    # 2. Inter-Factor Correlation Heatmap
    corr_df = pd.DataFrame(report["inter_factor_correlation_matrix"])
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        corr_df,
        annot=True,
        cmap="coolwarm",
        vmin=-0.6,
        vmax=0.6,
        fmt=".3f",
        square=True,
        linewidths=0.5,
    )
    plt.title("Inter-Factor Correlation Matrix (5 Latent Constructs)", fontsize=13, weight="bold")
    plt.tight_layout()
    p2 = str(p_dir / "inter_factor_correlations.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    saved_plots.append(p2)

    logger.info("Saved %d reliability plots to %s", len(saved_plots), plots_dir)
    return saved_plots
