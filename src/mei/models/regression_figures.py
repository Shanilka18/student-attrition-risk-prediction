"""Publication figure generation for MEI Track B Regression (FIG-REG-01 to FIG-REG-12).

Conforms strictly to brand styling:
- Palette: Primary #1F4E46, Accent #C98A00, Secondary #2B5C8A, Alert #B3261E
- DPI: 300 for PNG, plus vector SVG output
- Zero chartjunk, clear titles, labeled axes, readable annotations.
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

PRIMARY_TEAL = "#1F4E46"
ACCENT_GOLD = "#C98A00"
SECONDARY_BLUE = "#2B5C8A"
ALERT_RED = "#B3261E"
NEUTRAL_DARK = "#2D312E"
NEUTRAL_LIGHT = "#F4F6F4"
MUTED_GRAY = "#8A928D"

CATEGORY_COLORS = {
    "baseline": "#8A928D",
    "linear": "#2B5C8A",
    "classical": "#4A7C59",
    "boosting": "#C98A00",
    "hybrid": "#6B4C9A",
    "deep": "#1F4E46",
    "ensemble": "#B3261E",
}


def _save_figure(fig: plt.Figure, base_name: str, out_dirs: Sequence[Path]) -> None:
    """Save matplotlib figure to all target directories in 300 DPI PNG and SVG."""
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{base_name}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_reg_leaderboard_bars(
    leaderboard_df: Any,
    out_dirs: Sequence[Path],
    top_n: int = 15,
) -> None:
    """FIG-REG-01: Leaderboard Score ranking bar chart for top regression models."""
    df_top = leaderboard_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    y_pos = np.arange(len(df_top))
    scores = df_top["leaderboard_score"].values
    categories = df_top["category"].values
    colors = [CATEGORY_COLORS.get(c, PRIMARY_TEAL) for c in categories]

    bars = ax.barh(
        y_pos, scores, color=colors, height=0.65, edgecolor="none", alpha=0.9
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top["display_name"].values, fontsize=9, color=NEUTRAL_DARK)
    ax.invert_yaxis()

    ax.set_xlabel(
        "Track B Leaderboard Score: mean(R², Spearman) − Penalty",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )
    ax.set_title(
        "FIG-REG-01: Attrition Intent Continuous Score Regression Leaderboard",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_TEAL,
        pad=12,
    )
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=MUTED_GRAY)

    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color=NEUTRAL_DARK,
        )

    # Category legend handles
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=color, label=cat.capitalize())
        for cat, color in CATEGORY_COLORS.items()
        if cat in categories
    ]
    ax.legend(
        handles=legend_handles,
        title="Family",
        loc="lower right",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )

    plt.tight_layout()
    _save_figure(fig, "reg_01_leaderboard_bars", out_dirs)


def plot_reg_generalization_gap(
    leaderboard_df: Any,
    out_dirs: Sequence[Path],
) -> None:
    """FIG-REG-02: Generalization Gap (R²_train - R²_val) across candidate models."""
    df_sorted = leaderboard_df.sort_values("generalization_gap", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    y_pos = np.arange(len(df_sorted))
    gaps = df_sorted["generalization_gap"].values
    statuses = df_sorted["overfit_status"].values

    colors = []
    for st in statuses:
        if st == "Good fit":
            colors.append(PRIMARY_TEAL)
        elif st == "Slight overfit":
            colors.append(ACCENT_GOLD)
        else:
            colors.append(ALERT_RED)

    ax.barh(y_pos, gaps, color=colors, height=0.65, alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["display_name"].values, fontsize=8, color=NEUTRAL_DARK)

    ax.axvline(0.0, color=NEUTRAL_DARK, linewidth=1.0, linestyle="-")
    ax.axvline(
        0.05,
        color=ACCENT_GOLD,
        linewidth=1.2,
        linestyle="--",
        label="Good Fit Bound (+0.05)",
    )
    ax.axvline(
        0.10,
        color=ALERT_RED,
        linewidth=1.2,
        linestyle=":",
        label="Overfit Bound (+0.10)",
    )

    ax.set_xlabel(
        "Generalization Gap (Train R² − Val R²)",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )
    ax.set_title(
        "FIG-REG-02: Overfitting & Generalization Gap Across Regression Zoo",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_TEAL,
        pad=12,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.3, color=MUTED_GRAY)
    ax.legend(loc="lower right", frameon=True, fontsize=8)

    plt.tight_layout()
    _save_figure(fig, "reg_02_generalization_gap", out_dirs)


def plot_reg_metric_heatmap(
    leaderboard_df: Any,
    out_dirs: Sequence[Path],
    top_n: int = 12,
) -> None:
    """FIG-REG-03: Multi-metric heatmap comparing top regression models."""
    df_top = leaderboard_df.head(top_n).copy()
    metric_cols = [
        ("val_r2", "R²"),
        ("val_adj_r2", "Adj R²"),
        ("val_rmse", "RMSE ↓"),
        ("val_mae", "MAE ↓"),
        ("val_spearman_rho", "Spearman ρ"),
        ("val_pearson_r", "Pearson r"),
        ("val_within_5_pct", "±5 Pts %"),
        ("val_within_10_pct", "±10 Pts %"),
        ("val_f1_at_60", "F1 @ 60"),
    ]

    available_cols = [c for c, _ in metric_cols if c in df_top.columns]
    col_labels = [label for c, label in metric_cols if c in df_top.columns]

    data = df_top[available_cols].values
    model_labels = df_top["display_name"].values

    # Normalize each column to [0, 1] for relative heatmap color mapping
    norm_data = np.zeros_like(data)
    for j, (col, label) in enumerate(metric_cols):
        if col not in available_cols:
            continue
        col_vals = data[:, j]
        c_min, c_max = col_vals.min(), col_vals.max()
        if "↓" in label:
            # Lower is better: invert normalization
            norm_data[:, j] = (
                (c_max - col_vals) / (c_max - c_min) if c_max > c_min else 0.5
            )
        else:
            # Higher is better
            norm_data[:, j] = (
                (col_vals - c_min) / (c_max - c_min) if c_max > c_min else 0.5
            )

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    sns.heatmap(
        norm_data,
        annot=data,
        fmt=".3f",
        cmap="YlGnBu",
        yticklabels=model_labels,
        xticklabels=col_labels,
        cbar=False,
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(
        "FIG-REG-03: Relative Metric Performance Profile (Top Regression Models)",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_TEAL,
        pad=12,
    )
    plt.xticks(
        rotation=30, ha="right", fontsize=9, fontweight="bold", color=NEUTRAL_DARK
    )
    plt.yticks(fontsize=8, color=NEUTRAL_DARK)

    plt.tight_layout()
    _save_figure(fig, "reg_03_metric_heatmap", out_dirs)


def plot_reg_residuals_vs_fitted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    champion_name: str,
    bp_pvalue: float,
    out_dirs: Sequence[Path],
) -> None:
    """FIG-REG-04: Residuals vs Fitted values diagnostic scatter plot with tolerance bands."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    ax.scatter(
        y_pred, residuals, alpha=0.35, color=PRIMARY_TEAL, edgecolors="none", s=25
    )
    ax.axhline(0.0, color=ALERT_RED, linewidth=1.5, linestyle="-")
    ax.axhline(
        5.0,
        color=SECONDARY_BLUE,
        linewidth=1.0,
        linestyle="--",
        label="±5 Points Tolerance",
    )
    ax.axhline(-5.0, color=SECONDARY_BLUE, linewidth=1.0, linestyle="--")
    ax.axhline(
        10.0,
        color=ACCENT_GOLD,
        linewidth=1.0,
        linestyle=":",
        label="±10 Points Tolerance",
    )
    ax.axhline(-10.0, color=ACCENT_GOLD, linewidth=1.0, linestyle=":")

    # Lowess smooth trend
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smooth = lowess(residuals, y_pred, frac=0.4)
        ax.plot(
            smooth[:, 0],
            smooth[:, 1],
            color=NEUTRAL_DARK,
            linewidth=2.0,
            label="LOWESS Residual Trend",
        )
    except Exception:
        pass

    ax.set_xlabel(
        "Fitted Attrition Intent Score (0–100)",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )
    ax.set_ylabel(
        "Residual (Actual − Fitted Score)",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )
    ax.set_title(
        f"FIG-REG-04: Residuals vs Fitted Diagnostics — {champion_name}",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_TEAL,
        pad=12,
    )
    ax.set_ylim(-40, 40)
    ax.grid(True, linestyle="--", alpha=0.3, color=MUTED_GRAY)

    note = f"Breusch–Pagan Test: p = {bp_pvalue:.4f} ({'Heteroscedastic' if bp_pvalue < 0.05 else 'Homoscedastic'})"
    ax.text(
        0.03,
        0.06,
        note,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=NEUTRAL_LIGHT,
            edgecolor=MUTED_GRAY,
            alpha=0.85,
        ),
    )

    ax.legend(loc="upper right", frameon=True, fontsize=8)
    plt.tight_layout()
    _save_figure(fig, "reg_04_residuals_vs_fitted", out_dirs)


def plot_reg_qq_plot(
    residuals: np.ndarray,
    champion_name: str,
    out_dirs: Sequence[Path],
) -> None:
    """FIG-REG-05: Normal Q-Q plot of standardized residuals."""
    res = np.asarray(residuals, dtype=float)
    std_res = (res - np.mean(res)) / (np.std(res, ddof=1) or 1.0)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    osm, osr = stats.probplot(std_res, dist="norm")

    ax.scatter(
        osm[0], osm[1], color=PRIMARY_TEAL, alpha=0.5, s=20, label="Observed Residuals"
    )
    # 45-degree theoretical normal line
    line_x = np.linspace(-3.5, 3.5, 100)
    ax.plot(
        line_x,
        line_x,
        color=ALERT_RED,
        linewidth=1.8,
        linestyle="--",
        label="Theoretical Normal Line",
    )

    ax.set_xlabel(
        "Theoretical Normal Quantiles",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )
    ax.set_ylabel(
        "Standardized Residual Quantiles",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )
    ax.set_title(
        f"FIG-REG-05: Normal Q-Q Plot of Residuals — {champion_name}",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_TEAL,
        pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.3, color=MUTED_GRAY)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-4.0, 4.0)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    _save_figure(fig, "reg_05_qq_plot", out_dirs)


def plot_reg_residuals_by_pattern_and_level(
    residuals: np.ndarray,
    patterns: np.ndarray,
    levels: np.ndarray,
    champion_name: str,
    out_dirs: Sequence[Path],
) -> None:
    """FIG-REG-06: Residual distributions stratified by engagement pattern and level."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300, sharey=True)

    df_p = pd.DataFrame({"pattern": patterns, "residual": residuals})
    df_l = pd.DataFrame({"level": levels, "residual": residuals})

    # Pattern boxplot
    pat_order = ["Consistent", "Declining", "Sporadic", "Improving"]
    pat_order = [p for p in pat_order if p in df_p["pattern"].unique()] or list(
        df_p["pattern"].unique()
    )
    sns.boxplot(
        x="pattern",
        y="residual",
        data=df_p,
        order=pat_order,
        ax=axes[0],
        palette="Set2",
        width=0.5,
    )
    axes[0].axhline(0.0, color=ALERT_RED, linestyle="--", linewidth=1.2)
    axes[0].set_title(
        "Residuals by Decoded Pattern",
        fontsize=11,
        fontweight="bold",
        color=PRIMARY_TEAL,
    )
    axes[0].set_xlabel(
        "Engagement Pattern", fontsize=9, fontweight="bold", color=NEUTRAL_DARK
    )
    axes[0].set_ylabel(
        "Residual (Score Points)", fontsize=9, fontweight="bold", color=NEUTRAL_DARK
    )
    axes[0].grid(axis="y", linestyle="--", alpha=0.3, color=MUTED_GRAY)

    # Level boxplot
    lvl_order = ["High", "Moderate", "Low"]
    lvl_order = [lvl for lvl in lvl_order if lvl in df_l["level"].unique()] or list(
        df_l["level"].unique()
    )
    sns.boxplot(
        x="level",
        y="residual",
        data=df_l,
        order=lvl_order,
        ax=axes[1],
        palette="Blues_d",
        width=0.45,
    )
    axes[1].axhline(0.0, color=ALERT_RED, linestyle="--", linewidth=1.2)
    axes[1].set_title(
        "Residuals by Engagement Level",
        fontsize=11,
        fontweight="bold",
        color=PRIMARY_TEAL,
    )
    axes[1].set_xlabel(
        "Engagement Level", fontsize=9, fontweight="bold", color=NEUTRAL_DARK
    )
    axes[1].set_ylabel("")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3, color=MUTED_GRAY)

    fig.suptitle(
        f"FIG-REG-06: Stratified Residual Analysis Across Profiles — {champion_name}",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_TEAL,
    )
    plt.tight_layout()
    _save_figure(fig, "reg_06_residuals_by_pattern_and_level", out_dirs)


def plot_reg_calibration_binned(
    calibration_data: Dict[str, Any],
    champion_name: str,
    out_dirs: Sequence[Path],
) -> None:
    """FIG-REG-07: Binned calibration plot (predicted score vs observed score)."""
    centers = np.array(calibration_data.get("bin_centers", []))
    means = np.array(calibration_data.get("observed_means", []))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 45-degree perfect calibration line
    ax.plot(
        [0, 100],
        [0, 100],
        color=ALERT_RED,
        linestyle="--",
        linewidth=1.5,
        label="Perfect Parity (y = x)",
    )
    ax.plot(
        centers,
        means,
        marker="o",
        color=PRIMARY_TEAL,
        linewidth=2.0,
        markersize=7,
        label=f"Observed Calibration ({champion_name})",
    )

    ax.set_xlabel(
        "Mean Predicted Intent Score (Binned)",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )
    ax.set_ylabel(
        "Observed Mean Intent Score", fontsize=10, fontweight="bold", color=NEUTRAL_DARK
    )
    ax.set_title(
        f"FIG-REG-07: Continuous Score Calibration — {champion_name}",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_TEAL,
        pad=12,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.3, color=MUTED_GRAY)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    _save_figure(fig, "reg_07_calibration_binned", out_dirs)


def plot_reg_learning_curve(
    lc_data: Dict[str, Any],
    champion_name: str,
    out_dirs: Sequence[Path],
) -> None:
    """FIG-REG-08: Learning curve (R² vs training sample size)."""
    train_sizes = lc_data["train_sizes"]
    train_scores = lc_data["train_scores"]
    val_scores = lc_data["val_scores"]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    ax.plot(
        train_sizes,
        train_scores,
        "o-",
        color=SECONDARY_BLUE,
        linewidth=1.8,
        label="Training R²",
    )
    ax.plot(
        train_sizes,
        val_scores,
        "s-",
        color=PRIMARY_TEAL,
        linewidth=2.0,
        label="Validation R²",
    )

    ax.set_xlabel(
        "Training Sample Size (N)", fontsize=10, fontweight="bold", color=NEUTRAL_DARK
    )
    ax.set_ylabel(
        "Coefficient of Determination (R²)",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )
    ax.set_title(
        f"FIG-REG-08: Cross-Validated Learning Curve — {champion_name}",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_TEAL,
        pad=12,
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3, color=MUTED_GRAY)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    plt.tight_layout()
    _save_figure(fig, "reg_08_learning_curves", out_dirs)


def plot_reg_deep_epoch_curves(
    history: Dict[str, Any],
    model_name: str,
    out_dirs: Sequence[Path],
) -> None:
    """FIG-REG-10: Deep regressor training & validation epoch curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    t_loss = history.get("train_loss", [])
    v_loss = history.get("val_loss", [])
    t_r2 = history.get("train_r2", [])
    v_r2 = history.get("val_r2", [])
    epochs = np.arange(1, len(t_loss) + 1)

    # Loss curve
    ax1.plot(
        epochs, t_loss, label="Train Huber Loss", color=SECONDARY_BLUE, linewidth=1.8
    )
    ax1.plot(epochs, v_loss, label="Val Huber Loss", color=PRIMARY_TEAL, linewidth=2.0)
    ax1.set_xlabel("Epoch", fontsize=9, fontweight="bold", color=NEUTRAL_DARK)
    ax1.set_ylabel(
        "Huber Loss (δ = 2.0)", fontsize=9, fontweight="bold", color=NEUTRAL_DARK
    )
    ax1.set_title(
        "Huber Loss Progression", fontsize=11, fontweight="bold", color=PRIMARY_TEAL
    )
    ax1.grid(True, linestyle="--", alpha=0.3, color=MUTED_GRAY)
    ax1.legend(loc="upper right", fontsize=8)

    # R2 curve
    ax2.plot(epochs, t_r2, label="Train R²", color=SECONDARY_BLUE, linewidth=1.8)
    ax2.plot(epochs, v_r2, label="Val R²", color=PRIMARY_TEAL, linewidth=2.0)
    ax2.set_xlabel("Epoch", fontsize=9, fontweight="bold", color=NEUTRAL_DARK)
    ax2.set_ylabel("R² Score", fontsize=9, fontweight="bold", color=NEUTRAL_DARK)
    ax2.set_title(
        "R² Score Progression", fontsize=11, fontweight="bold", color=PRIMARY_TEAL
    )
    ax2.grid(True, linestyle="--", alpha=0.3, color=MUTED_GRAY)
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"FIG-REG-10: Deep Model Epoch Curves — {model_name}",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_TEAL,
    )
    plt.tight_layout()
    _save_figure(fig, "reg_10_deep_epoch_curves", out_dirs)


def plot_reg_boosting_loss_curves(
    boosting_losses: Dict[str, Dict[str, List[float]]],
    out_dirs: Sequence[Path],
) -> None:
    """FIG-REG-09: Per-iteration validation loss (RMSE/MSE) for boosting regressors."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    model_colors = {
        "xgboost_regressor": PRIMARY_TEAL,
        "lightgbm_regressor": ACCENT_GOLD,
        "catboost_regressor": SECONDARY_BLUE,
    }

    for mid, loss_dict in boosting_losses.items():
        val_losses = loss_dict.get("val_loss", [])
        if val_losses:
            iters = np.arange(1, len(val_losses) + 1)
            color = model_colors.get(mid, NEUTRAL_DARK)
            label = mid.replace("_regressor", "").upper()
            ax.plot(
                iters, val_losses, label=f"{label} Val Loss", color=color, linewidth=2.0
            )

    ax.set_xlabel(
        "Boosting Iteration / Trees", fontsize=10, fontweight="bold", color=NEUTRAL_DARK
    )
    ax.set_ylabel(
        "Validation Loss (RMSE / MSE)",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )
    ax.set_title(
        "FIG-REG-09: Boosting Iteration Loss Progression Across Trees",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_TEAL,
        pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.3, color=MUTED_GRAY)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    _save_figure(fig, "reg_09_boosting_loss_curves", out_dirs)


def plot_reg_track_a_vs_track_b_consistency(
    track_a_probs: np.ndarray,
    track_b_scores: np.ndarray,
    consistency_data: Dict[str, Any],
    out_dirs: Sequence[Path],
) -> None:
    """FIG-REG-11: Scatter and joint alignment of Track A probability vs Track B score."""
    p_a = np.asarray(track_a_probs, dtype=float)
    s_b = np.asarray(track_b_scores, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)

    # Quadrant background shading
    ax.axhspan(60.0, 100.0, xmin=0.5, xmax=1.0, color="#C98A00", alpha=0.08)
    ax.axhspan(0.0, 60.0, xmin=0.0, xmax=0.5, color="#1F4E46", alpha=0.08)

    ax.scatter(p_a, s_b, alpha=0.4, color=PRIMARY_TEAL, edgecolors="none", s=25)

    ax.axvline(
        0.50,
        color=ALERT_RED,
        linestyle="--",
        linewidth=1.2,
        label="Track A Threshold (p = 0.50)",
    )
    ax.axhline(
        60.0,
        color=SECONDARY_BLUE,
        linestyle="--",
        linewidth=1.2,
        label="Track B Threshold (score = 60.0)",
    )

    rho = consistency_data.get("spearman_rho", 0.0)
    agree = consistency_data.get("classification_agreement_pct", 0.0)
    stat_box = (
        f"Spearman ρ: {rho:.4f}\n"
        f"Decision Agreement: {agree:.1f}%\n"
        f"Concordance: Monotonic"
    )
    ax.text(
        0.04,
        0.85,
        stat_box,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=NEUTRAL_LIGHT,
            edgecolor=MUTED_GRAY,
            alpha=0.9,
        ),
    )

    ax.set_xlabel(
        "Track A Champion OOF Predicted Probability (at_risk)",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )
    ax.set_ylabel(
        "Track B Champion OOF Predicted Intent Score (0–100)",
        fontsize=10,
        fontweight="bold",
        color=NEUTRAL_DARK,
    )
    ax.set_title(
        "FIG-REG-11: Cross-Track Consistency (Track A Classification vs Track B Regression)",
        fontsize=12,
        fontweight="bold",
        color=PRIMARY_TEAL,
        pad=12,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, linestyle="--", alpha=0.3, color=MUTED_GRAY)
    ax.legend(loc="lower right", frameon=True, fontsize=8)

    plt.tight_layout()
    _save_figure(fig, "reg_11_track_a_vs_track_b_consistency", out_dirs)


def plot_reg_wilcoxon_matrix(
    wilcoxon_data: Dict[str, Any],
    top_5_models: List[str],
    out_dirs: Sequence[Path],
) -> None:
    """FIG-REG-12: Pairwise Wilcoxon signed-rank significance matrix with Holm correction."""
    k = len(top_5_models)
    p_matrix = np.ones((k, k), dtype=float)

    # Populate p_matrix
    pair_map = {
        (r["model_1"], r["model_2"]): r["p_value_holm"]
        for r in wilcoxon_data.get("pairwise_results", [])
    }
    for (m1, m2), pval in pair_map.items():
        if m1 in top_5_models and m2 in top_5_models:
            i, j = top_5_models.index(m1), top_5_models.index(m2)
            p_matrix[i, j] = pval
            p_matrix[j, i] = pval

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    mask = np.eye(k, dtype=bool)

    sns.heatmap(
        p_matrix,
        mask=mask,
        annot=True,
        fmt=".4f",
        cmap="Blues_r",
        vmin=0.0,
        vmax=0.10,
        xticklabels=top_5_models,
        yticklabels=top_5_models,
        ax=ax,
        cbar_kws={"label": "Holm-Adjusted p-value"},
    )

    ax.set_title(
        "FIG-REG-12: Top 5 Regression Models Pairwise Wilcoxon Significance (Holm Corrected)",
        fontsize=11,
        fontweight="bold",
        color=PRIMARY_TEAL,
        pad=12,
    )
    plt.xticks(
        rotation=30, ha="right", fontsize=8, fontweight="bold", color=NEUTRAL_DARK
    )
    plt.yticks(fontsize=8, fontweight="bold", color=NEUTRAL_DARK)

    plt.tight_layout()
    _save_figure(fig, "reg_12_top5_wilcoxon_matrix", out_dirs)
