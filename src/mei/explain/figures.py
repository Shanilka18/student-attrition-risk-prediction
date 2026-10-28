"""Publication figure rendering for the MEI explainability subsystem.

Generates 300-DPI PNG and SVG figures adhering to the project visual identity:
Paper #F5F2EC, Ink #1A1D21, Primary Forest #1F4E46, Warning Amber #C98A00, Risk High #B3261E.
"""

import os

# Set matplotlib cache directory to avoid Windows permissions error
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Publication Style Constants
COLOR_PAPER = "#F5F2EC"
COLOR_SURFACE = "#FFFFFF"
COLOR_INK = "#1A1D21"
COLOR_MUTED = "#5B6168"
COLOR_LINE = "#DDD8CE"
COLOR_FOREST = "#1F4E46"
COLOR_INFO = "#2B5C8A"
COLOR_WARNING = "#C98A00"
COLOR_RISK_LOW = "#2E7D4F"
COLOR_RISK_MOD = "#C98A00"
COLOR_RISK_HIGH = "#B3261E"


def _apply_axes_styling(
    ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = ""
) -> None:
    """Apply standard publication styling to a matplotlib Axes."""
    ax.set_facecolor(COLOR_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)
    ax.tick_params(colors=COLOR_INK, labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_LINE)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=COLOR_INK, pad=12)
    if xlabel:
        ax.set_xlabel(
            xlabel, fontsize=11, fontweight="semibold", color=COLOR_INK, labelpad=8
        )
    if ylabel:
        ax.set_ylabel(
            ylabel, fontsize=11, fontweight="semibold", color=COLOR_INK, labelpad=8
        )


def _save_figure(fig: plt.Figure, output_dirs: List[Path], filename_base: str) -> None:
    """Save figure in both 300-DPI PNG and vector SVG across specified directories."""
    fig.patch.set_facecolor(COLOR_PAPER)
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out_dir / f"{filename_base}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor=COLOR_PAPER,
        )
        fig.savefig(
            out_dir / f"{filename_base}.svg", bbox_inches="tight", facecolor=COLOR_PAPER
        )
    plt.close(fig)


def plot_global_bar(
    top_features: List[Dict[str, Any]],
    output_dirs: List[Path],
    top_n: int = 15,
) -> None:
    """FIG-SHAP-01: Horizontal bar chart of top drivers ranked by mean |SHAP|."""
    fig, ax = plt.subplots(figsize=(10, 7))

    top_feats = top_features[:top_n][::-1]
    names = [f["plain_name"] for f in top_feats]
    values = [f["mean_abs_shap"] * 100.0 for f in top_feats]  # in MEI percentage points

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=COLOR_FOREST, height=0.65, edgecolor=COLOR_LINE)

    # Annotate values
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.2f} pts",
            ha="left",
            va="center",
            fontsize=9,
            color=COLOR_INK,
            fontweight="semibold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    _apply_axes_styling(
        ax,
        title=f"Top {top_n} Global Drivers of Student Attrition Risk (Mean |SHAP| Impact)",
        xlabel="Mean Absolute SHAP Impact on MEI Score (Points)",
    )
    ax.set_xlim(0, max(values) * 1.18)

    _save_figure(fig, output_dirs, "fig_shap_01_global_bar")


def plot_beeswarm_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    plain_names: List[str],
    output_dirs: List[Path],
    top_n: int = 12,
) -> None:
    """FIG-SHAP-02: Beeswarm-style summary showing attribution distribution and feature values."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(-mean_abs)[:top_n][::-1]

    fig, ax = plt.subplots(figsize=(11, 7.5))

    y_positions = np.arange(top_n)

    for i, idx in enumerate(top_indices):
        s_vals = shap_values[:, idx] * 100.0  # in MEI points
        f_vals = X_sample[:, idx]

        # Normalize feature values to [0, 1] for colormap
        min_v, max_v = np.percentile(f_vals, 2), np.percentile(f_vals, 98)
        norm_v = np.clip((f_vals - min_v) / (max_v - min_v + 1e-8), 0.0, 1.0)

        # Add slight jitter on y-axis
        jitter = np.random.normal(0, 0.08, size=len(s_vals))
        colors = plt.cm.coolwarm(norm_v)

        ax.scatter(
            s_vals,
            y_positions[i] + jitter,
            c=colors,
            s=16,
            alpha=0.65,
            edgecolors="none",
        )

    ax.axvline(0, color=COLOR_INK, linestyle="--", alpha=0.6, linewidth=1.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([plain_names[idx] for idx in top_indices], fontsize=10)
    _apply_axes_styling(
        ax,
        title="SHAP Summary Beeswarm: Impact on Student Attrition Risk",
        xlabel="SHAP Value (Impact on MEI: Positive = Higher Risk, Negative = Protective)",
    )

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, aspect=20)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low Feature Value", "High Feature Value"], fontsize=9)
    cbar.ax.tick_params(colors=COLOR_INK)

    _save_figure(fig, output_dirs, "fig_shap_02_beeswarm")


def plot_dependence_top8(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    plain_names: List[str],
    output_dirs: List[Path],
) -> None:
    """FIG-SHAP-03: 8-panel dependence plot grid for top 8 drivers with interaction coloring."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top8_indices = np.argsort(-mean_abs)[:8]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    axes = axes.flatten()

    for i, idx in enumerate(top8_indices):
        ax = axes[i]
        f_vals = X_sample[:, idx]
        s_vals = shap_values[:, idx] * 100.0

        # Find best interaction feature (highest correlation among remaining features)
        corrs = [
            abs(np.corrcoef(X_sample[:, idx], X_sample[:, j])[0, 1]) if j != idx else 0
            for j in range(len(feature_names))
        ]
        inter_idx = (
            int(np.nanargmax(corrs))
            if np.max(corrs) > 0
            else (idx + 1) % len(feature_names)
        )
        inter_vals = X_sample[:, inter_idx]
        norm_inter = (inter_vals - np.min(inter_vals)) / (np.ptp(inter_vals) + 1e-8)

        ax.scatter(
            f_vals,
            s_vals,
            c=norm_inter,
            cmap="viridis",
            s=18,
            alpha=0.7,
            edgecolors="none",
        )
        ax.axhline(0, color=COLOR_MUTED, linestyle="--", linewidth=1.0)

        # Fit trend curve
        sort_order = np.argsort(f_vals)
        from scipy.ndimage import uniform_filter1d

        smooth = uniform_filter1d(s_vals[sort_order], size=max(10, len(f_vals) // 10))
        ax.plot(
            f_vals[sort_order], smooth, color=COLOR_FOREST, linewidth=2.0, alpha=0.9
        )

        _apply_axes_styling(
            ax,
            title=f"{i + 1}. {plain_names[idx][:28]}",
            xlabel="Feature Value",
            ylabel="SHAP Impact (MEI pts)",
        )

    plt.tight_layout()
    _save_figure(fig, output_dirs, "fig_shap_03_dependence_top8")


def plot_grouped_importance(
    grouped_data: Dict[str, Dict[str, Any]],
    output_dirs: List[Path],
) -> None:
    """FIG-SHAP-04: Multi-panel grouped importance by Indicator, Period, and Feature Family."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel A: By Indicator
    ax_ind = axes[0]
    by_ind = grouped_data["by_indicator"]
    ind_labels = [k.replace("_", " ").title() for k in by_ind.keys()]
    ind_shares = [v["share_pct"] for v in by_ind.values()]
    ax_ind.barh(
        np.arange(len(ind_labels))[::-1], ind_shares, color=COLOR_FOREST, height=0.65
    )
    ax_ind.set_yticks(np.arange(len(ind_labels))[::-1])
    ax_ind.set_yticklabels(ind_labels, fontsize=9.5)
    _apply_axes_styling(
        ax_ind,
        title="A. Share by Indicator Dimension",
        xlabel="Share of Total Importance (%)",
    )
    for i, s in enumerate(ind_shares):
        ax_ind.text(
            s + 0.5,
            len(ind_labels) - 1 - i,
            f"{s:.1f}%",
            va="center",
            fontsize=8.5,
            color=COLOR_INK,
        )

    # Panel B: By Academic Period
    ax_p = axes[1]
    by_p = grouped_data["by_period"]
    p_labels = list(by_p.keys())
    p_shares = [v["share_pct"] for v in by_p.values()]
    colors_p = [COLOR_INFO, COLOR_FOREST, COLOR_WARNING, COLOR_RISK_HIGH, COLOR_MUTED][
        : len(p_labels)
    ]
    ax_p.bar(p_labels, p_shares, color=colors_p, width=0.55)
    _apply_axes_styling(
        ax_p,
        title="B. Share by Academic Period",
        ylabel="Share of Total Importance (%)",
    )
    for i, s in enumerate(p_shares):
        ax_p.text(
            i,
            s + 0.5,
            f"{s:.1f}%",
            ha="center",
            fontsize=9,
            color=COLOR_INK,
            fontweight="semibold",
        )

    # Panel C: By Feature Family
    ax_fam = axes[2]
    by_fam = grouped_data["by_family"]
    fam_labels = [k.replace("_", " ").title() for k in by_fam.keys()]
    fam_shares = [v["share_pct"] for v in by_fam.values()]
    colors_fam = [COLOR_FOREST, COLOR_INFO, COLOR_WARNING][: len(fam_labels)]
    wedges, texts, autotexts = ax_fam.pie(
        fam_shares,
        labels=fam_labels,
        autopct="%1.1f%%",
        colors=colors_fam,
        startangle=140,
        textprops={"fontsize": 10, "color": COLOR_INK},
    )
    for at in autotexts:
        at.set_color(COLOR_SURFACE)
        at.set_fontweight("bold")
    ax_fam.set_title(
        "C. Share by Feature Family",
        fontsize=13,
        fontweight="bold",
        color=COLOR_INK,
        pad=12,
    )

    plt.tight_layout()
    _save_figure(fig, output_dirs, "fig_shap_04_grouped_importance")


def plot_fold_consistency(
    consistency_data: Dict[str, Any],
    output_dirs: List[Path],
) -> None:
    """FIG-SHAP-05: Fold-level SHAP consistency (Spearman correlation matrix and Jaccard overlap)."""
    corr_matrix = np.array(consistency_data["spearman_matrix"])
    jaccards = consistency_data["pairwise_jaccard"]
    kendall_w = consistency_data["kendall_w"]
    mean_corr = consistency_data["mean_spearman"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Pairwise Spearman correlation heatmap across 15 folds
    ax_hm = axes[0]
    im = ax_hm.imshow(corr_matrix, cmap="YlGn", vmin=0.6, vmax=1.0)
    ax_hm.set_xticks(np.arange(15))
    ax_hm.set_yticks(np.arange(15))
    ax_hm.set_xticklabels([f"F{i}" for i in range(15)], fontsize=8)
    ax_hm.set_yticklabels([f"F{i}" for i in range(15)], fontsize=8)
    _apply_axes_styling(
        ax_hm, title=f"A. 15-Fold SHAP Rank Correlation (Mean r = {mean_corr:.3f})"
    )
    plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    # Panel B: Jaccard overlap distribution of top-10 features
    ax_bx = axes[1]
    ax_bx.hist(jaccards, bins=10, color=COLOR_FOREST, edgecolor=COLOR_LINE, alpha=0.85)
    mean_j = float(np.mean(jaccards))
    ax_bx.axvline(
        mean_j,
        color=COLOR_RISK_HIGH,
        linestyle="--",
        linewidth=2.0,
        label=f"Mean Jaccard = {mean_j:.3f}",
    )
    ax_bx.axvline(
        0.5,
        color=COLOR_MUTED,
        linestyle=":",
        linewidth=1.5,
        label="High Stability Threshold (0.50)",
    )
    _apply_axes_styling(
        ax_bx,
        title=f"B. Top-10 Jaccard Overlap (Kendall's W = {kendall_w:.3f})",
        xlabel="Pairwise Jaccard Overlap",
        ylabel="Fold Pairs Count",
    )
    ax_bx.legend(loc="upper left", fontsize=9.5)

    plt.tight_layout()
    _save_figure(fig, output_dirs, "fig_shap_05_fold_consistency")


def plot_cross_method_comparison(
    comparison_data: Dict[str, Any],
    output_dirs: List[Path],
) -> None:
    """FIG-SHAP-06: Cross-method agreement scatter plots comparing SHAP, Permutation, and LogReg."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    corr_sp = comparison_data["correlations"]
    ranks = comparison_data["ranks_table"]

    shap_ranks = [r["shap_rank"] for r in ranks]
    perm_ranks = [r["perm_rank"] for r in ranks]
    lr_ranks = [r["lr_rank"] for r in ranks]

    # Panel A: SHAP vs Permutation Importance
    ax_sp = axes[0]
    ax_sp.scatter(
        shap_ranks, perm_ranks, color=COLOR_FOREST, alpha=0.7, edgecolors="none"
    )
    ax_sp.plot([1, 100], [1, 100], color=COLOR_MUTED, linestyle="--")
    _apply_axes_styling(
        ax_sp,
        title=f"A. SHAP vs Permutation (rho = {corr_sp['shap_vs_permutation']:.3f})",
        xlabel="SHAP Rank (1 = Top Driver)",
        ylabel="Permutation Rank",
    )

    # Panel B: SHAP vs Logistic Regression
    ax_sl = axes[1]
    ax_sl.scatter(shap_ranks, lr_ranks, color=COLOR_INFO, alpha=0.7, edgecolors="none")
    ax_sl.plot([1, 100], [1, 100], color=COLOR_MUTED, linestyle="--")
    _apply_axes_styling(
        ax_sl,
        title=f"B. SHAP vs LogReg |beta| (rho = {corr_sp['shap_vs_logistic']:.3f})",
        xlabel="SHAP Rank",
        ylabel="LogReg Rank",
    )

    # Panel C: Permutation vs Logistic Regression
    ax_pl = axes[2]
    ax_pl.scatter(
        perm_ranks, lr_ranks, color=COLOR_WARNING, alpha=0.7, edgecolors="none"
    )
    ax_pl.plot([1, 100], [1, 100], color=COLOR_MUTED, linestyle="--")
    _apply_axes_styling(
        ax_pl,
        title=f"C. Permutation vs LogReg (rho = {corr_sp['permutation_vs_logistic']:.3f})",
        xlabel="Permutation Rank",
        ylabel="LogReg Rank",
    )

    plt.tight_layout()
    _save_figure(fig, output_dirs, "fig_shap_06_cross_method_comparison")


def plot_pattern_drivers(
    pattern_drivers: Dict[str, Any],
    output_dirs: List[Path],
) -> None:
    """FIG-SHAP-07: Top drivers of attrition risk stratified by student engagement pattern."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    patterns = ["Consistent", "Declining", "Sporadic", "Improving"]
    palette = [COLOR_INFO, COLOR_RISK_HIGH, COLOR_WARNING, COLOR_RISK_LOW]

    for i, pat in enumerate(patterns):
        ax = axes[i]
        p_data = pattern_drivers.get(pat, {"drivers": [], "n_students": 0})
        drivers = p_data["drivers"][:5][::-1]
        n_stu = p_data["n_students"]

        names = [d["plain_name"][:28] for d in drivers]
        vals = [d["mean_shap"] * 100.0 for d in drivers]

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, vals, color=palette[i], height=0.6)
        for b in bars:
            w = b.get_width()
            ax.text(
                w + 0.1,
                b.get_y() + b.get_height() / 2,
                f"{w:+.1f} pts",
                va="center",
                fontsize=9,
                color=COLOR_INK,
                fontweight="bold",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9.5)
        _apply_axes_styling(
            ax,
            title=f"{pat} Students (N = {n_stu})",
            xlabel="Mean Risk Impact (MEI pts)",
        )

    plt.tight_layout()
    _save_figure(fig, output_dirs, "fig_shap_07_pattern_drivers")


def plot_waterfall_examples(
    examples: List[Dict[str, Any]],
    output_dirs: List[Path],
) -> None:
    """FIG-SHAP-08: Waterfall attribution charts for representative student cases."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for i, ex in enumerate(examples[:3]):
        ax = axes[i]
        stu_id = ex["student_id"]
        mei = ex["mei"]
        base_mei = ex["baseline_mei"]
        band = ex["band"]
        reasons = ex["reasons"]

        labels = (
            ["Cohort Baseline"]
            + [r.get("plain_name", r.get("feature", "Driver"))[:22] for r in reasons]
            + ["Final MEI"]
        )
        [base_mei] + [r["points"] for r in reasons] + [0]

        cumulative = np.cumsum([base_mei] + [r["points"] for r in reasons])
        bottoms = [0] + list(cumulative[:-1]) + [0]

        colors = (
            [COLOR_MUTED]
            + [COLOR_RISK_HIGH if r["points"] > 0 else COLOR_RISK_LOW for r in reasons]
            + [COLOR_FOREST]
        )

        y_pos = np.arange(len(labels))
        ax.bar(
            y_pos[:-1],
            [base_mei] + [r["points"] for r in reasons],
            bottom=bottoms[:-1],
            color=colors[:-1],
            width=0.55,
        )
        ax.bar(y_pos[-1], mei, bottom=0, color=colors[-1], width=0.55)

        ax.set_xticks(y_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
        _apply_axes_styling(
            ax, title=f"{stu_id} (MEI: {mei}, {band})", ylabel="MEI Score (0-100)"
        )

    plt.tight_layout()
    _save_figure(fig, output_dirs, "fig_shap_08_waterfall_examples")
