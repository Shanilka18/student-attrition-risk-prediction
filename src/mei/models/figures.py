"""Publication figures generation for MEI Phase 6 classification benchmarking."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Unified brand palette
NAVY = "#1E3A8A"
EMERALD = "#10B981"
AMBER = "#F59E0B"
ROSE = "#EF4444"
SLATE = "#6B7280"
DARK_TEXT = "#1A1D21"
MUTED_LINE = "#DDD8CE"
BACKGROUND = "#FFFFFF"

PALETTE_CATEGORIES = {
    "baseline": SLATE,
    "linear": "#2B5C8A",
    "classical": AMBER,
    "boosting": "#1F4E46",
    "hybrid": EMERALD,
    "ensemble": "#8C4A3B",
}


def _save_fig(fig: plt.Figure, base_name: str, output_dirs: List[Path]) -> None:
    """Save figure to multiple directories in 300 DPI PNG and SVG formats."""
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{base_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(d / f"{base_name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_leaderboard_bars(
    lb_df: pd.DataFrame,
    output_dirs: List[Path],
    top_n: int = 25,
) -> None:
    """FIG-CLASS-01: Leaderboard ranking bar chart by Leaderboard Score."""
    df_plot = lb_df.head(top_n).sort_values("leaderboard_score", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    colors = [PALETTE_CATEGORIES.get(cat, SLATE) for cat in df_plot["category"]]
    bars = ax.barh(
        df_plot["display_name"],
        df_plot["leaderboard_score"],
        color=colors,
        height=0.65,
        edgecolor="none",
    )

    ax.set_xlabel(
        "Leaderboard Score (CV Composite - Overfit Penalty)",
        fontsize=11,
        fontweight="bold",
        color=DARK_TEXT,
    )
    ax.set_title(
        "MEI Classification Benchmark Leaderboard (Top Models Ranked by Composite Score)",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color=DARK_TEXT,
    )
    ax.set_xlim(0.4, 0.95)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color=MUTED_LINE)

    for bar, score in zip(bars, df_plot["leaderboard_score"]):
        ax.text(
            score + 0.005,
            bar.get_y() + bar.get_height() / 2.0,
            f"{score:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            color=DARK_TEXT,
            fontweight="bold",
        )

    # Legend for categories
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=col, label=cat.capitalize())
        for cat, col in PALETTE_CATEGORIES.items()
        if cat in df_plot["category"].values
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor=MUTED_LINE,
    )

    _save_fig(fig, "class_01_leaderboard_bars", output_dirs)


def plot_generalization_gap(
    lb_df: pd.DataFrame,
    output_dirs: List[Path],
) -> None:
    """FIG-CLASS-02: Generalization Gap (Train F1 - Val F1) with overfit thresholds."""
    df_plot = lb_df.sort_values("generalization_gap", ascending=False).reset_index(
        drop=True
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    status_colors = {"Good fit": EMERALD, "Slight overfit": AMBER, "Overfit": ROSE}
    bar_colors = [status_colors.get(st, SLATE) for st in df_plot["overfit_status"]]

    ax.bar(
        range(len(df_plot)), df_plot["generalization_gap"], color=bar_colors, width=0.6
    )

    ax.axhline(
        0.05,
        color=AMBER,
        linestyle="--",
        linewidth=1.5,
        label="Slight Overfit Threshold (gap = 0.05)",
    )
    ax.axhline(
        0.10,
        color=ROSE,
        linestyle="--",
        linewidth=1.5,
        label="Severe Overfit Threshold (gap = 0.10)",
    )

    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot["model_id"], rotation=60, ha="right", fontsize=9)
    ax.set_ylabel(
        "Generalization Gap (Train F1 - Val F1)",
        fontsize=11,
        fontweight="bold",
        color=DARK_TEXT,
    )
    ax.set_title(
        "Model Generalization Gap & Overfitting Audit Across 15 CV Folds",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color=DARK_TEXT,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5, color=MUTED_LINE)
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor=MUTED_LINE,
    )

    _save_fig(fig, "class_02_generalization_gap", output_dirs)


def plot_metric_heatmap(
    lb_df: pd.DataFrame,
    output_dirs: List[Path],
    top_n: int = 15,
) -> None:
    """FIG-CLASS-03: Multi-metric heatmap across top models."""
    sub_df = lb_df.head(top_n).set_index("display_name")
    cols = [
        ("val_f1_optimal", "F1 At-Risk"),
        ("val_auc_roc", "AUC-ROC"),
        ("val_balanced_accuracy", "Bal Acc"),
        ("val_mcc", "MCC"),
        ("val_pr_auc", "PR-AUC"),
        ("val_precision", "Precision"),
        ("val_recall", "Recall"),
        ("val_specificity", "Specificity"),
    ]
    data = sub_df[[c[0] for c in cols]]
    data.columns = [c[1] for c in cols]

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(BACKGROUND)

    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar_kws={"label": "Metric Value"},
        ax=ax,
        linewidths=0.5,
        linecolor=MUTED_LINE,
    )
    ax.set_title(
        "Performance Metric Heatmap Across Top Ranked Models",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color=DARK_TEXT,
    )
    ax.set_ylabel("")

    _save_fig(fig, "class_03_metric_heatmap", output_dirs)


def plot_roc_curves_by_category(
    all_results: List[Dict[str, Any]],
    per_cat_best: Dict[str, Any],
    output_dirs: List[Path],
) -> None:
    """FIG-CLASS-04: ROC curve overlays of champion model per category."""
    results_map = {r["spec"]["id"]: r for r in all_results}

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    for cat, info in per_cat_best.items():
        mid = info["model_id"]
        res = results_map.get(mid)
        if not res or "curves" not in res:
            continue
        curves = res["curves"]
        auc_val = info["val_auc_roc"]
        color = PALETTE_CATEGORIES.get(cat, SLATE)
        ax.plot(
            curves["roc_fpr"],
            curves["roc_tpr"],
            label=f"{info['display_name']} ({cat.capitalize()}) - AUC: {auc_val:.4f}",
            color=color,
            linewidth=2.0,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color=SLATE,
        linestyle="--",
        linewidth=1.5,
        label="Random Guess (AUC = 0.5000)",
    )
    ax.set_xlabel(
        "False Positive Rate (1 - Specificity)",
        fontsize=11,
        fontweight="bold",
        color=DARK_TEXT,
    )
    ax.set_ylabel(
        "True Positive Rate (Sensitivity / Recall)",
        fontsize=11,
        fontweight="bold",
        color=DARK_TEXT,
    )
    ax.set_title(
        "ROC Curve Comparison Across Category Champions (15-Fold OOF)",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color=DARK_TEXT,
    )
    ax.grid(True, linestyle="--", alpha=0.5, color=MUTED_LINE)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor=MUTED_LINE,
        fontsize=9,
    )

    _save_fig(fig, "class_04_roc_curves_by_category", output_dirs)


def plot_pr_curves_by_category(
    all_results: List[Dict[str, Any]],
    per_cat_best: Dict[str, Any],
    prevalence: float,
    output_dirs: List[Path],
) -> None:
    """FIG-CLASS-05: Precision-Recall curves of best model per category."""
    results_map = {r["spec"]["id"]: r for r in all_results}

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    for cat, info in per_cat_best.items():
        mid = info["model_id"]
        res = results_map.get(mid)
        if not res or "curves" not in res:
            continue
        curves = res["curves"]
        pr_auc = info["val_pr_auc"]
        color = PALETTE_CATEGORIES.get(cat, SLATE)
        ax.plot(
            curves["pr_recall"],
            curves["pr_precision"],
            label=f"{info['display_name']} - PR-AUC: {pr_auc:.4f}",
            color=color,
            linewidth=2.0,
        )

    ax.axhline(
        prevalence,
        color=SLATE,
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline Prevalence ({prevalence * 100:.1f}%)",
    )
    ax.set_xlabel(
        "Recall (Coverage of At-Risk Students)",
        fontsize=11,
        fontweight="bold",
        color=DARK_TEXT,
    )
    ax.set_ylabel(
        "Precision (Positive Predictive Value)",
        fontsize=11,
        fontweight="bold",
        color=DARK_TEXT,
    )
    ax.set_title(
        "Precision-Recall Curves Across Category Champions",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color=DARK_TEXT,
    )
    ax.grid(True, linestyle="--", alpha=0.5, color=MUTED_LINE)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor=MUTED_LINE,
        fontsize=9,
    )

    _save_fig(fig, "class_05_pr_curves_by_category", output_dirs)


def plot_calibration_curves(
    top_models: List[Dict[str, Any]],
    output_dirs: List[Path],
) -> None:
    """FIG-CLASS-06: Calibration reliability diagrams for top models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    colors = [NAVY, EMERALD, AMBER, "#8C4A3B", "#2B5C8A"]

    for idx, m in enumerate(top_models[:5]):
        curves = m.get("curves", {})
        cal_true = curves.get("cal_prob_true", [])
        cal_pred = curves.get("cal_prob_pred", [])
        if not cal_true or not cal_pred:
            continue
        ece_val = m["leaderboard_metrics"]["val_ece"]
        name = m["spec"]["display_name"]
        color = colors[idx % len(colors)]
        ax.plot(
            cal_pred,
            cal_true,
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"{name} (ECE: {ece_val:.4f})",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        color=SLATE,
        linestyle="--",
        linewidth=1.5,
        label="Perfect Calibration",
    )
    ax.set_xlabel(
        "Mean Predicted Probability", fontsize=11, fontweight="bold", color=DARK_TEXT
    )
    ax.set_ylabel(
        "Empirical Fraction of Positives (At-Risk)",
        fontsize=11,
        fontweight="bold",
        color=DARK_TEXT,
    )
    ax.set_title(
        "Probability Calibration Reliability Diagrams (Top Models)",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color=DARK_TEXT,
    )
    ax.grid(True, linestyle="--", alpha=0.5, color=MUTED_LINE)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor=MUTED_LINE,
        fontsize=9,
    )

    _save_fig(fig, "class_06_calibration_curves", output_dirs)


def plot_boosting_loss_curves(
    all_results: List[Dict[str, Any]],
    output_dirs: List[Path],
) -> None:
    """FIG-CLASS-07: Train vs Val Log-Loss vs Trees for XGBoost, LightGBM, CatBoost."""
    boosting_ids = ["xgboost", "lightgbm", "catboost"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.patch.set_facecolor(BACKGROUND)

    results_map = {r["spec"]["id"]: r for r in all_results}

    for idx, mid in enumerate(boosting_ids):
        ax = axes[idx]
        ax.set_facecolor(BACKGROUND)
        res = results_map.get(mid)
        if res and res.get("iteration_losses"):
            tr_l = res["iteration_losses"]["train_loss"]
            v_l = res["iteration_losses"]["val_loss"]
            iters = range(1, len(tr_l) + 1)
            ax.plot(iters, tr_l, label="Train Loss", color=NAVY, linewidth=2.0)
            ax.plot(iters, v_l, label="Val Loss", color=AMBER, linewidth=2.0)
            ax.set_title(
                res["spec"]["display_name"],
                fontsize=11,
                fontweight="bold",
                color=DARK_TEXT,
            )
            ax.set_xlabel("Boosting Iterations (Trees)", fontsize=10, color=DARK_TEXT)
            ax.grid(True, linestyle="--", alpha=0.5, color=MUTED_LINE)
            ax.legend(
                loc="upper right", frameon=True, facecolor="white", edgecolor=MUTED_LINE
            )
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")

    axes[0].set_ylabel("Log-Loss", fontsize=11, fontweight="bold", color=DARK_TEXT)
    fig.suptitle(
        "Boosting Optimization: Training vs Validation Loss Across Iterations",
        fontsize=13,
        fontweight="bold",
        y=1.02,
        color=DARK_TEXT,
    )

    _save_fig(fig, "class_07_boosting_loss_curves", output_dirs)


def plot_learning_curves(
    top_models: List[Dict[str, Any]],
    output_dirs: List[Path],
) -> None:
    """FIG-CLASS-08: Learning curves (F1 Score vs Sample Size) for top 3 models."""
    fig, axes = plt.subplots(1, min(3, len(top_models)), figsize=(15, 5), sharey=True)
    fig.patch.set_facecolor(BACKGROUND)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for idx, m in enumerate(top_models[:3]):
        ax = axes[idx]
        ax.set_facecolor(BACKGROUND)
        lc = m.get("learning_curves", {})
        sizes = lc.get("train_sizes", [])
        tr_s = lc.get("train_scores", [])
        v_s = lc.get("val_scores", [])

        if sizes and tr_s and v_s:
            ax.plot(
                sizes, tr_s, marker="o", color=NAVY, label="Train F1", linewidth=2.0
            )
            ax.plot(
                sizes, v_s, marker="s", color=EMERALD, label="CV Val F1", linewidth=2.0
            )
            ax.set_title(
                m["spec"]["display_name"],
                fontsize=11,
                fontweight="bold",
                color=DARK_TEXT,
            )
            ax.set_xlabel("Training Set Size (Samples)", fontsize=10, color=DARK_TEXT)
            ax.grid(True, linestyle="--", alpha=0.5, color=MUTED_LINE)
            ax.legend(
                loc="lower right", frameon=True, facecolor="white", edgecolor=MUTED_LINE
            )

    axes[0].set_ylabel(
        "F1 Score (At-Risk Class)", fontsize=11, fontweight="bold", color=DARK_TEXT
    )
    fig.suptitle(
        "Empirical Learning Curves: F1 Performance vs Training Cohort Size",
        fontsize=13,
        fontweight="bold",
        y=1.02,
        color=DARK_TEXT,
    )

    _save_fig(fig, "class_08_learning_curves", output_dirs)


def plot_validation_curves(
    top_models: List[Dict[str, Any]],
    output_dirs: List[Path],
) -> None:
    """FIG-CLASS-09: Validation curves (Hyperparameter Sweep vs F1) for selected models."""
    valid_models = [
        m
        for m in top_models
        if m.get("validation_curves", {}).get("param_name") is not None
    ][:3]
    if not valid_models:
        return

    fig, axes = plt.subplots(1, len(valid_models), figsize=(15, 5))
    fig.patch.set_facecolor(BACKGROUND)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for idx, m in enumerate(valid_models):
        ax = axes[idx]
        ax.set_facecolor(BACKGROUND)
        vc = m["validation_curves"]
        param_name = vc["param_name"]
        p_vals = vc["param_values"]
        tr_s = vc["train_scores"]
        v_s = vc["val_scores"]

        x_coords = range(len(p_vals))
        ax.plot(x_coords, tr_s, marker="o", color=NAVY, label="Train F1", linewidth=2.0)
        ax.plot(
            x_coords, v_s, marker="s", color=EMERALD, label="CV Val F1", linewidth=2.0
        )

        ax.set_xticks(x_coords)
        ax.set_xticklabels(p_vals, rotation=45, ha="right", fontsize=9)
        ax.set_title(
            f"{m['spec']['display_name']}\nSweep: {param_name}",
            fontsize=11,
            fontweight="bold",
            color=DARK_TEXT,
        )
        ax.set_xlabel(param_name, fontsize=10, color=DARK_TEXT)
        ax.set_ylabel("F1 Score", fontsize=10, color=DARK_TEXT)
        ax.grid(True, linestyle="--", alpha=0.5, color=MUTED_LINE)
        ax.legend(
            loc="lower right", frameon=True, facecolor="white", edgecolor=MUTED_LINE
        )

    fig.suptitle(
        "Hyperparameter Validation Curves: Bias-Variance Tradeoff Analysis",
        fontsize=13,
        fontweight="bold",
        y=1.02,
        color=DARK_TEXT,
    )
    _save_fig(fig, "class_09_validation_curves", output_dirs)


def plot_smote_comparison(
    smote_res: Dict[str, Any],
    output_dirs: List[Path],
) -> None:
    """FIG-CLASS-10: Grouped bar chart comparing Class Weights vs In-Fold SMOTE."""
    models = list(smote_res.keys())
    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    x = np.arange(len(models))
    width = 0.35

    cw_f1 = [smote_res[m]["class_weights"]["f1"] for m in models]
    sm_f1 = [smote_res[m]["smote"]["f1"] for m in models]
    labels = [smote_res[m]["model_name"] for m in models]

    b1 = ax.bar(
        x - width / 2.0,
        cw_f1,
        width,
        label="Class Weights (Cost-Sensitive)",
        color=NAVY,
    )
    b2 = ax.bar(
        x + width / 2.0, sm_f1, width, label="SMOTE (Synthetic In-Fold)", color=AMBER
    )

    ax.set_ylabel(
        "CV Validation F1 (At-Risk)", fontsize=11, fontweight="bold", color=DARK_TEXT
    )
    ax.set_title(
        "Class Imbalance Strategy Comparison: Class Weights vs In-Fold SMOTE",
        fontsize=12,
        fontweight="bold",
        pad=15,
        color=DARK_TEXT,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0.5, 0.75)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color=MUTED_LINE)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=MUTED_LINE)

    for bar in list(b1) + list(b2):
        val = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    _save_fig(fig, "class_10_smote_vs_class_weights", output_dirs)
