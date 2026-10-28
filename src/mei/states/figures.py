"""Publication figure generation for student engagement states and trajectory patterns.

Generates 7 figures in 300 DPI PNG and SVG formats using the project brand palette:
1. FIG-STATE-01: Gaussian HMM emission profiles across 7 indicators
2. FIG-STATE-02: State transition probability matrix and expected dwell times
3. FIG-STATE-03: Longitudinal composite engagement trajectories by decoded pattern
4. FIG-STATE-04: Frequency distribution of engagement patterns and overall levels
5. FIG-STATE-05: Observed at-risk rates by trajectory pattern with 95% Wilson CIs
6. FIG-STATE-06: Silhouette sample analysis across selected student clusters
7. FIG-STATE-07: HMM model selection curves (BIC and validation log-likelihood vs K)
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_samples

logger = logging.getLogger(__name__)

# Brand color palette (Rule G10 / G9)
PALETTE = {
    "paper": "#F5F2EC",
    "surface": "#FFFFFF",
    "ink": "#1A1D21",
    "muted": "#5B6168",
    "line": "#DDD8CE",
    "primary": "#1F4E46",
    "blue": "#2B5C8A",
    "amber": "#C98A00",
    "risk_low": "#2E7D4F",
    "risk_mod": "#C98A00",
    "risk_high": "#B3261E",
    "extra_1": "#8C4A3B",
    "extra_2": "#6E7F3F",
    "extra_3": "#7A6F63",
}

PATTERN_COLORS = {
    "Consistent": PALETTE["primary"],
    "Declining": PALETTE["risk_high"],
    "Improving": PALETTE["risk_low"],
    "Sporadic": PALETTE["amber"],
}


def _save_figure(fig: plt.Figure, base_name: str, output_dirs: List[Path]) -> None:
    """Save matplotlib figure to multiple directories in both PNG and SVG at 300 DPI."""
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            d / f"{base_name}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        fig.savefig(
            d / f"{base_name}.svg",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


def plot_emission_profiles(
    emission_means: np.ndarray,
    indicator_names: List[str],
    state_names: Dict[int, str],
    output_dirs: List[Path],
) -> None:
    """Plot FIG-STATE-01: Heatmap of HMM emission means for 7 indicators across states."""
    k = emission_means.shape[0]
    fig, ax = plt.subplots(figsize=(9, 4 + 0.5 * k), facecolor=PALETTE["surface"])
    ax.set_facecolor(PALETTE["paper"])

    state_labels = [
        f"State {i + 1}: {state_names.get(i, f'S{i + 1}')}" for i in range(k)
    ]
    clean_indicators = [ind.replace("_", " ").title() for ind in indicator_names]

    sns.heatmap(
        emission_means,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=clean_indicators,
        yticklabels=state_labels,
        cbar_kws={"label": "Standardized Emission Mean (Z-score)"},
        linewidths=0.5,
        linecolor=PALETTE["line"],
        ax=ax,
    )

    ax.set_title(
        "FIG-STATE-01: Gaussian HMM Latent State Emission Profiles",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        pad=12,
    )
    plt.xticks(rotation=30, ha="right", fontsize=9, color=PALETTE["ink"])
    plt.yticks(rotation=0, fontsize=9, color=PALETTE["ink"])

    _save_figure(fig, "state_01_emission_profiles", output_dirs)


def plot_transition_matrix(
    transmat: np.ndarray,
    dwell_times: Dict[int, float],
    state_names: Dict[int, str],
    output_dirs: List[Path],
) -> None:
    """Plot FIG-STATE-02: State transition probability matrix and expected dwell times."""
    k = transmat.shape[0]
    fig, ax = plt.subplots(figsize=(8, 4 + 0.6 * k), facecolor=PALETTE["surface"])
    ax.set_facecolor(PALETTE["paper"])

    labels = [
        f"State {i + 1}: {state_names.get(i, f'S{i + 1}')}\n(Dwell: {dwell_times.get(i, 0.0):.1f}p)"
        for i in range(k)
    ]

    sns.heatmap(
        transmat,
        annot=True,
        fmt=".3f",
        cmap="Greens",
        xticklabels=[
            f"State {i + 1}: {state_names.get(i, f'S{i + 1}')}" for i in range(k)
        ],
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Transition Probability $A_{ij}$"},
        linewidths=0.5,
        linecolor=PALETTE["line"],
        ax=ax,
    )

    ax.set_title(
        "FIG-STATE-02: State Transition Probabilities and Expected Dwell Times",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        pad=12,
    )
    ax.set_xlabel("Next Period State $s_{t+1}$", fontsize=10, color=PALETTE["ink"])
    ax.set_ylabel("Current Period State $s_t$", fontsize=10, color=PALETTE["ink"])

    _save_figure(fig, "state_02_transition_matrix", output_dirs)


def plot_trajectories_by_pattern(
    clean_df: pd.DataFrame,
    patterns: List[str],
    output_dirs: List[Path],
) -> None:
    """Plot FIG-STATE-03: Mean composite engagement across P1..P4 by decoded pattern."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=PALETTE["surface"])
    ax.set_facecolor(PALETTE["paper"])

    comp_cols = ["composite_p1", "composite_p2", "composite_p3", "composite_p4"]
    periods = [1, 2, 3, 4]

    # Calculate composites if not present
    if not all(c in clean_df.columns for c in comp_cols):
        indicators = [
            "resource_access",
            "study_time",
            "review_time",
            "quiz_practice",
            "assignment_engagement",
            "study_consistency",
            "participation",
        ]
        p_vals = np.zeros((len(clean_df), 4))
        for p_idx, p in enumerate(periods):
            cols = [f"{ind}_p{p}" for ind in indicators]
            p_vals[:, p_idx] = (
                (clean_df[cols].mean(axis=1).values - 1.0) / 4.0
            ) * 100.0
    else:
        p_vals = clean_df[comp_cols].values

    pat_array = np.array(patterns)
    unique_patterns = ["Consistent", "Declining", "Improving", "Sporadic"]

    for pat in unique_patterns:
        mask = pat_array == pat
        if not np.any(mask):
            continue
        sub_vals = p_vals[mask]
        mean_traj = sub_vals.mean(axis=0)
        std_traj = sub_vals.std(axis=0)
        color = PATTERN_COLORS.get(pat, PALETTE["muted"])

        ax.plot(
            periods,
            mean_traj,
            marker="o",
            linewidth=2.5,
            label=f"{pat} (n={np.sum(mask)})",
            color=color,
        )
        ax.fill_between(
            periods,
            np.clip(mean_traj - std_traj, 0, 100),
            np.clip(mean_traj + std_traj, 0, 100),
            alpha=0.15,
            color=color,
        )

    ax.set_title(
        "FIG-STATE-03: Mean Composite Engagement Trajectories by Pattern",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        pad=12,
    )
    ax.set_xlabel("Semester Period (Weeks 1-16)", fontsize=10, color=PALETTE["ink"])
    ax.set_ylabel(
        "Composite Engagement Index (0-100)", fontsize=10, color=PALETTE["ink"]
    )
    ax.set_xticks(periods)
    ax.set_xticklabels(["P1 (W1-4)", "P2 (W5-8)", "P3 (W9-12)", "P4 (W13-16)"])
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle="--", alpha=0.5, color=PALETTE["line"])
    ax.legend(frameon=True, facecolor=PALETTE["surface"], edgecolor=PALETTE["line"])

    _save_figure(fig, "state_03_trajectories_by_pattern", output_dirs)


def plot_pattern_distribution(
    patterns: List[str],
    levels: List[str],
    output_dirs: List[Path],
) -> None:
    """Plot FIG-STATE-04: Frequency distribution of engagement patterns and levels."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5), facecolor=PALETTE["surface"]
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(PALETTE["paper"])

    pat_series = pd.Series(patterns).value_counts()
    lvl_series = pd.Series(levels).value_counts()

    # Pattern bar chart
    pat_order = ["Consistent", "Declining", "Improving", "Sporadic"]
    pat_order = [p for p in pat_order if p in pat_series.index]
    colors_pat = [PATTERN_COLORS.get(p, PALETTE["muted"]) for p in pat_order]

    bars1 = ax1.bar(
        pat_order,
        [pat_series.get(p, 0) for p in pat_order],
        color=colors_pat,
        edgecolor=PALETTE["line"],
        width=0.6,
    )
    for bar in bars1:
        yval = bar.get_height()
        pct = (yval / len(patterns)) * 100.0
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 15,
            f"{yval}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["ink"],
        )
    ax1.set_title(
        "Trajectory Patterns", fontsize=11, fontweight="bold", color=PALETTE["ink"]
    )
    ax1.set_ylabel("Student Count", fontsize=10, color=PALETTE["ink"])
    ax1.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"], axis="y")

    # Level bar chart
    lvl_order = ["Low", "Medium", "High"]
    lvl_order = [lvl for lvl in lvl_order if lvl in lvl_series.index]
    colors_lvl = [PALETTE["risk_high"], PALETTE["amber"], PALETTE["risk_low"]]

    bars2 = ax2.bar(
        lvl_order,
        [lvl_series.get(lvl, 0) for lvl in lvl_order],
        color=colors_lvl[: len(lvl_order)],
        edgecolor=PALETTE["line"],
        width=0.5,
    )
    for bar in bars2:
        yval = bar.get_height()
        pct = (yval / len(levels)) * 100.0
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 15,
            f"{yval}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["ink"],
        )
    ax2.set_title(
        "Overall Engagement Levels",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax2.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"], axis="y")

    fig.suptitle(
        "FIG-STATE-04: Student Engagement Pattern & Level Distributions",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        y=1.02,
    )

    _save_figure(fig, "state_04_pattern_distribution", output_dirs)


def plot_at_risk_by_pattern(
    patterns: List[str],
    at_risk: np.ndarray,
    output_dirs: List[Path],
) -> None:
    """Plot FIG-STATE-05: At-risk rate by pattern with Wilson 95% confidence intervals."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=PALETTE["surface"])
    ax.set_facecolor(PALETTE["paper"])

    pat_order = ["Consistent", "Declining", "Improving", "Sporadic"]
    df = pd.DataFrame({"pattern": patterns, "at_risk": at_risk})

    rates = []
    cis_lower = []
    cis_upper = []
    ns = []

    for pat in pat_order:
        sub = df[df["pattern"] == pat]
        n = len(sub)
        ns.append(n)
        if n == 0:
            rates.append(0.0)
            cis_lower.append(0.0)
            cis_upper.append(0.0)
            continue

        p_hat = sub["at_risk"].mean()
        rates.append(p_hat * 100.0)

        # Wilson score 95% CI
        z = 1.96
        denom = 1.0 + (z**2) / n
        center = (p_hat + (z**2) / (2.0 * n)) / denom
        delta = (
            z * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * n**2))
        ) / denom
        cis_lower.append(max(0.0, center - delta) * 100.0)
        cis_upper.append(min(1.0, center + delta) * 100.0)

    yerr = [
        np.array(rates) - np.array(cis_lower),
        np.array(cis_upper) - np.array(rates),
    ]

    colors = [PATTERN_COLORS.get(p, PALETTE["muted"]) for p in pat_order]
    bars = ax.bar(
        pat_order,
        rates,
        yerr=yerr,
        capsize=5,
        color=colors,
        edgecolor=PALETTE["line"],
        width=0.55,
    )

    for idx, bar in enumerate(bars):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 4.0,
            f"{yval:.1f}%\n(n={ns[idx]})",
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["ink"],
        )

    ax.set_title(
        "FIG-STATE-05: At-Risk Prevalence by Trajectory Pattern (with 95% Wilson CIs)",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        pad=12,
    )
    ax.set_ylabel("At-Risk Proportion (%)", fontsize=10, color=PALETTE["ink"])
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"], axis="y")

    _save_figure(fig, "state_05_at_risk_by_pattern", output_dirs)


def plot_cluster_silhouette(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_names: Dict[int, str],
    output_dirs: List[Path],
) -> None:
    """Plot FIG-STATE-06: Silhouette sample analysis across selected student clusters."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=PALETTE["surface"])
    ax.set_facecolor(PALETTE["paper"])

    sample_sil = silhouette_samples(X_scaled, cluster_labels)
    k = len(np.unique(cluster_labels))
    avg_sil = np.mean(sample_sil)

    y_lower = 10
    colors = [
        PALETTE["primary"],
        PALETTE["blue"],
        PALETTE["amber"],
        PALETTE["risk_high"],
        PALETTE["extra_1"],
        PALETTE["extra_2"],
    ]

    for i in range(k):
        c_sil = sample_sil[cluster_labels == i]
        c_sil.sort()
        size_c = len(c_sil)
        y_upper = y_lower + size_c

        color = colors[i % len(colors)]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            c_sil,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
            label=f"Cluster {i}: {cluster_names.get(i, f'C{i}')} (n={size_c})",
        )
        y_lower = y_upper + 10

    ax.axvline(
        x=avg_sil,
        color=PALETTE["risk_high"],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean Silhouette: {avg_sil:.3f}",
    )

    ax.set_title(
        "FIG-STATE-06: Cluster Silhouette Profile and Sample Cohesion",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        pad=12,
    )
    ax.set_xlabel("Silhouette Coefficient", fontsize=10, color=PALETTE["ink"])
    ax.set_ylabel("Sorted Samples per Cluster", fontsize=10, color=PALETTE["ink"])
    ax.set_yticks([])
    ax.set_xlim(-0.2, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"], axis="x")
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=PALETTE["surface"],
        edgecolor=PALETTE["line"],
        fontsize=8,
    )

    _save_figure(fig, "state_06_cluster_silhouette", output_dirs)


def plot_k_selection_curves(
    model_table: pd.DataFrame,
    output_dirs: List[Path],
) -> None:
    """Plot FIG-STATE-07: HMM model selection curves: BIC and validation log-likelihood vs K."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.5), facecolor=PALETTE["surface"]
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(PALETTE["paper"])

    ks = model_table["k"].values
    bics = model_table["bic"].values
    train_lls = model_table["train_log_likelihood"].values

    # Left plot: BIC
    ax1.plot(
        ks, bics, marker="o", color=PALETTE["primary"], linewidth=2.0, markersize=8
    )
    for k_val, bic_val in zip(ks, bics):
        ax1.annotate(
            f"{bic_val:.0f}",
            (k_val, bic_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color=PALETTE["ink"],
        )
    ax1.set_title(
        "Bayesian Information Criterion (BIC)",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax1.set_xlabel("Number of Latent States (K)", fontsize=10, color=PALETTE["ink"])
    ax1.set_ylabel("BIC (Lower is Better)", fontsize=10, color=PALETTE["ink"])
    ax1.set_xticks(ks)
    ax1.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"])

    # Right plot: Log-Likelihood
    ax2.plot(
        ks,
        train_lls,
        marker="s",
        color=PALETTE["blue"],
        linewidth=2.0,
        markersize=8,
        label="Training Log-Likelihood",
    )
    if (
        "val_log_likelihood" in model_table.columns
        and not model_table["val_log_likelihood"].isna().all()
    ):
        val_lls = model_table["val_log_likelihood"].values
        ax2.plot(
            ks,
            val_lls,
            marker="^",
            color=PALETTE["amber"],
            linewidth=2.0,
            markersize=8,
            linestyle="--",
            label="Validation Log-Likelihood",
        )
        ax2.legend(
            frameon=True, facecolor=PALETTE["surface"], edgecolor=PALETTE["line"]
        )

    ax2.set_title(
        "Log-Likelihood across K", fontsize=11, fontweight="bold", color=PALETTE["ink"]
    )
    ax2.set_xlabel("Number of Latent States (K)", fontsize=10, color=PALETTE["ink"])
    ax2.set_ylabel(
        "Log-Likelihood (Higher is Better)", fontsize=10, color=PALETTE["ink"]
    )
    ax2.set_xticks(ks)
    ax2.grid(True, linestyle="--", alpha=0.4, color=PALETTE["line"])

    fig.suptitle(
        "FIG-STATE-07: Gaussian HMM Model Selection Criteria (K in [2, 4])",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        y=1.02,
    )

    _save_figure(fig, "state_07_k_selection_curves", output_dirs)
