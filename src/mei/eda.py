"""Exploratory Data Analysis and publication-quality figure generation.

Adheres strictly to the MEI project visual identity (§10 of PROJECT_BRIEF.md):
Palette: Paper #F5F2EC, Ink #1A1D21, Primary Forest #1F4E46, Info Blue #2B5C8A,
Risk Low #2E7D4F, Risk High #B3261E, Warning Amber #C98A00.
Outputs both 300-dpi PNG and vector SVG formats.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mei.generator import RealityCheckStats

# Style Constants
PALETTE = {
    "paper": "#F5F2EC",
    "surface": "#FFFFFF",
    "ink": "#1A1D21",
    "muted": "#5B6168",
    "line": "#DDD8CE",
    "primary": "#1F4E46",
    "info": "#2B5C8A",
    "warning": "#C98A00",
    "risk_low": "#2E7D4F",
    "risk_moderate": "#C98A00",
    "risk_high": "#B3261E",
    "cat1": "#8C4A3B",
    "cat2": "#6E7F3F",
    "cat3": "#7A6F63",
}

ARCHETYPE_COLORS = {
    "consistent_high": PALETTE["risk_low"],
    "consistent_moderate": PALETTE["info"],
    "consistently_low": PALETTE["risk_high"],
    "declining": PALETTE["warning"],
    "sporadic": PALETTE["cat1"],
    "improving": PALETTE["cat2"],
}


def setup_figure_style() -> None:
    """Apply consistent styling across matplotlib plots."""
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = PALETTE["line"]
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["text.color"] = PALETTE["ink"]
    plt.rcParams["axes.labelcolor"] = PALETTE["ink"]
    plt.rcParams["xtick.color"] = PALETTE["muted"]
    plt.rcParams["ytick.color"] = PALETTE["muted"]


def save_figure(fig: plt.Figure, base_name: str, output_dirs: List[Path]) -> None:
    """Save figure in both 300-DPI PNG and vector SVG to all specified directories."""
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"{base_name}.png"
        svg_path = out_dir / f"{base_name}.svg"
        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        fig.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
    plt.close(fig)


def plot_marginal_distributions(raw_df: pd.DataFrame, output_dirs: List[Path]) -> None:
    """Plot response distributions for 28 indicators to check spread (<45%)."""
    setup_figure_style()

    # Identify indicator columns from raw_df (those with period brackets)
    indicator_cols = [
        c
        for c in raw_df.columns
        if "[" in c
        and ("Period 1" in c or "Period 2" in c or "Period 3" in c or "Period 4" in c)
    ]
    if len(indicator_cols) > 28:
        indicator_cols = indicator_cols[:28]

    fig, axes = plt.subplots(7, 4, figsize=(16, 18), sharex=True, sharey=True)
    fig.patch.set_facecolor(PALETTE["paper"])

    for idx, col in enumerate(indicator_cols):
        r = idx // 4
        c = idx % 4
        ax = axes[r, c]
        ax.set_facecolor(PALETTE["surface"])

        # Count frequencies
        series = raw_df[col].dropna()
        counts = series.value_counts(normalize=True).sort_index()

        x_vals = np.arange(len(counts))
        bars = ax.bar(
            x_vals,
            counts.values,
            color=PALETTE["primary"],
            alpha=0.85,
            edgecolor=PALETTE["surface"],
            linewidth=0.5,
        )

        # Highlight if any category exceeds 45% threshold
        for bar, val in zip(bars, counts.values):
            if val > 0.45:
                bar.set_color(PALETTE["risk_high"])

        # Reference dashed line at 45% ceiling
        ax.axhline(
            0.45,
            color=PALETTE["risk_high"],
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )

        # Short clean label: indicator name and period
        short_title = col.split("[")[0].strip()[:18]
        period_str = (
            col.split("[")[1].split(":")[0].strip() if "[" in col else f"P{c + 1}"
        )
        ax.set_title(
            f"{short_title} ({period_str})",
            fontsize=8,
            fontweight="bold",
            color=PALETTE["ink"],
        )
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 0.55)
        ax.grid(axis="y", linestyle=":", alpha=0.4, color=PALETTE["line"])

    fig.suptitle(
        "Marginal Response Distributions Across 28 Indicators (<45% Ceiling)",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["ink"],
        y=0.995,
    )
    plt.tight_layout()
    save_figure(fig, "eda_01_marginal_distributions", output_dirs)


def plot_indicator_correlations(raw_df: pd.DataFrame, output_dirs: List[Path]) -> None:
    """Plot correlation heatmap of the 7 engagement indicators averaged over periods."""
    setup_figure_style()

    from mei.io.codec import SurveyCodec
    from mei.io.schema import load_mapping

    mapping = load_mapping()
    codec = SurveyCodec()

    # Invert mapping to match raw Google Forms long headers to canonical names
    canonical_by_header = {}
    for col_name, variants in mapping.columns.items():
        for v in variants:
            if v in raw_df.columns:
                canonical_by_header[v] = col_name

    # Encode indicator columns to numeric codes
    encoded_df = pd.DataFrame()
    for hdr, col_name in canonical_by_header.items():
        if col_name in [
            "student_id",
            "timestamp",
            "consent",
            "attention_check",
            "intent_leave",
            "intent_defer",
            "intent_not_continue",
        ]:
            continue
        encoded_df[col_name] = raw_df[hdr].map(
            lambda x: codec.encode(col_name, x) if pd.notna(x) else np.nan
        )

    indicators = [
        "resource_access",
        "study_time",
        "review_time",
        "quiz_practice",
        "assignment_engagement",
        "study_consistency",
        "participation",
    ]

    # Create composite indicators by averaging the 4 periods
    comp_df = pd.DataFrame()
    for ind in indicators:
        cols = [
            f"{ind}_p{t}" for t in range(1, 5) if f"{ind}_p{t}" in encoded_df.columns
        ]
        if cols:
            comp_df[ind] = encoded_df[cols].mean(axis=1)

    corr = comp_df.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["surface"])

    cmap = sns.diverging_palette(45, 160, s=80, l=45, as_cmap=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-0.2,
        vmax=1.0,
        square=True,
        linewidths=1.0,
        linecolor=PALETTE["paper"],
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.set_title(
        "Inter-Indicator Correlation Structure (Averaged Over 4 Periods)",
        fontsize=11,
        fontweight="bold",
        pad=12,
        color=PALETTE["ink"],
    )
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    save_figure(fig, "eda_02_indicator_correlations", output_dirs)


def plot_temporal_archetypes(hidden_df: pd.DataFrame, output_dirs: List[Path]) -> None:
    """Plot latent trajectories e_1..e_4 across the 6 student archetypes."""
    setup_figure_style()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["surface"])

    periods = ["P1 (Early)", "P2 (Early-Mid)", "P3 (Late-Mid)", "P4 (Recent)"]
    x = np.arange(4)

    for arch, color in ARCHETYPE_COLORS.items():
        sub = hidden_df[hidden_df["archetype"] == arch]
        means = [sub[f"e_{t + 1}"].mean() for t in range(4)]
        stds = [sub[f"e_{t + 1}"].std() for t in range(4)]

        pct_str = f"{len(sub) / len(hidden_df):.1%}"
        label = f"{arch.replace('_', ' ').title()} (n={len(sub)}, {pct_str})"
        ax.plot(
            x,
            means,
            marker="o",
            markersize=6,
            linewidth=2.2,
            color=color,
            label=label,
        )
        ax.fill_between(
            x,
            np.subtract(means, stds),
            np.add(means, stds),
            color=color,
            alpha=0.12,
        )

    ax.set_title(
        "Latent Engagement Trajectories Across 6 Hidden Archetypes (Mean +/- 1 SD)",
        fontsize=11,
        fontweight="bold",
        pad=12,
        color=PALETTE["ink"],
    )
    ax.set_xlabel("Semester Observation Period", fontsize=10, labelpad=8)
    ax.set_ylabel("Latent Engagement Level (e_t in [0, 1])", fontsize=10, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.5, color=PALETTE["line"])
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)

    plt.tight_layout()
    save_figure(fig, "eda_03_temporal_archetypes", output_dirs)


def plot_at_risk_by_archetype(hidden_df: pd.DataFrame, output_dirs: List[Path]) -> None:
    """Plot empirical at-risk prevalence and mean risk score by archetype."""
    setup_figure_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax1.set_facecolor(PALETTE["surface"])
    ax2.set_facecolor(PALETTE["surface"])

    grp = hidden_df.groupby("archetype")
    rates = grp["at_risk"].mean().sort_values(ascending=True)
    scores = grp["attrition_intent_score"].mean().loc[rates.index]

    colors = [ARCHETYPE_COLORS.get(arch, PALETTE["primary"]) for arch in rates.index]

    # Subplot 1: Binary at_risk rate
    y_pos = np.arange(len(rates))
    bars1 = ax1.barh(
        y_pos,
        rates.values * 100,
        color=colors,
        alpha=0.88,
        edgecolor=PALETTE["surface"],
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([a.replace("_", " ").title() for a in rates.index], fontsize=9)
    ax1.set_xlabel("At-Risk Prevalence (%)", fontsize=10, labelpad=8)
    ax1.set_title("Attrition At-Risk Rate by Archetype", fontsize=10, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.grid(axis="x", linestyle=":", alpha=0.5, color=PALETTE["line"])

    for bar, val in zip(bars1, rates.values):
        ax1.text(
            val * 100 + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=PALETTE["ink"],
        )

    # Subplot 2: Continuous Attrition Intent Score
    bars2 = ax2.barh(
        y_pos,
        scores.values,
        color=colors,
        alpha=0.88,
        edgecolor=PALETTE["surface"],
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Mean Attrition Intent Score (0–100)", fontsize=10, labelpad=8)
    ax2.set_title("Mean Intent Score by Archetype", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.axvline(
        60.0,
        color=PALETTE["risk_high"],
        linestyle="--",
        linewidth=1.0,
        label="Threshold (60.0)",
    )
    ax2.grid(axis="x", linestyle=":", alpha=0.5, color=PALETTE["line"])
    ax2.legend(loc="lower right", fontsize=8)

    for bar, val in zip(bars2, scores.values):
        ax2.text(
            val + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=PALETTE["ink"],
        )

    fig.suptitle(
        "Target Distribution Across Latent Behavioral Archetypes",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        y=0.98,
    )
    plt.tight_layout()
    save_figure(fig, "eda_04_at_risk_by_archetype", output_dirs)


def plot_missingness_map(raw_df: pd.DataFrame, output_dirs: List[Path]) -> None:
    """Plot missingness heatmap across students and survey items."""
    setup_figure_style()

    indicator_cols = [
        c
        for c in raw_df.columns
        if "[" in c
        and ("Period 1" in c or "Period 2" in c or "Period 3" in c or "Period 4" in c)
    ]
    if len(indicator_cols) > 28:
        indicator_cols = indicator_cols[:28]

    # Sample 300 students for clean visual display
    sample_size = min(300, len(raw_df))
    sub = raw_df[indicator_cols].iloc[:sample_size]
    is_missing = sub.isna().astype(int)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor(PALETTE["paper"])
    ax1.set_facecolor(PALETTE["surface"])
    ax2.set_facecolor(PALETTE["surface"])

    # 1. Missingness binary grid
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([PALETTE["surface"], PALETTE["risk_high"]])
    ax1.imshow(
        is_missing.values,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )
    ax1.set_title(
        f"Missing Value Patterns across First {sample_size} Students (Red = Missing)",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax1.set_ylabel("Student Observations", fontsize=9)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 2. Column-level missingness percentage barplot
    col_missing_pct = raw_df[indicator_cols].isna().mean() * 100
    x_pos = np.arange(len(col_missing_pct))
    ax2.bar(
        x_pos,
        col_missing_pct.values,
        color=PALETTE["primary"],
        alpha=0.85,
        edgecolor=PALETTE["surface"],
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [f"I{i // 4 + 1}P{i % 4 + 1}" for i in range(len(indicator_cols))],
        rotation=90,
        fontsize=7,
    )
    ax2.set_ylabel("Missing (%)", fontsize=9)
    ax2.set_xlabel("Indicator & Period (I1P1 .. I7P4)", fontsize=9)
    ax2.set_ylim(0, max(col_missing_pct.max() * 1.3, 5.0))
    ax2.grid(axis="y", linestyle=":", alpha=0.5, color=PALETTE["line"])

    plt.tight_layout()
    save_figure(fig, "eda_05_missingness_map", output_dirs)


def plot_response_anomalies(stats: RealityCheckStats, output_dirs: List[Path]) -> None:
    """Plot reality check comparison of configured vs empirical anomaly rates."""
    setup_figure_style()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(PALETTE["paper"])
    ax.set_facecolor(PALETTE["surface"])

    metrics = stats.metrics
    labels = [
        "Prevalence (At-Risk)",
        "Item Missing (MCAR)",
        "Period Dropout (MAR)",
        "Acquiescent Style",
        "Extreme Responding",
        "Midpoint Responding",
        "Straight-Lining",
        "Attention Check Fail",
        "No-Consent Submissions",
        "Duplicate Submissions",
        "Option Label Typos",
    ]
    keys = [
        "prevalence",
        "item_missing_rate",
        "period_missing_rate",
        "acquiescent_rate",
        "extreme_rate",
        "midpoint_rate",
        "straight_line_rate",
        "attention_fail_rate",
        "no_consent_rate",
        "duplicate_rate",
        "typo_rate",
    ]

    configured_vals = [metrics[k]["configured"] * 100 for k in keys]
    empirical_vals = [metrics[k]["empirical"] * 100 for k in keys]

    y = np.arange(len(labels))
    height = 0.35

    ax.barh(
        y - height / 2,
        configured_vals,
        height,
        label="Configured Target (%)",
        color=PALETTE["info"],
        alpha=0.75,
        edgecolor=PALETTE["surface"],
    )
    ax.barh(
        y + height / 2,
        empirical_vals,
        height,
        label="Empirical Generated (%)",
        color=PALETTE["primary"],
        alpha=0.9,
        edgecolor=PALETTE["surface"],
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Percentage (%)", fontsize=10, labelpad=8)
    ax.set_title(
        "Synthetic Data Reality Check: Configured vs. Empirical Rates",
        fontsize=11,
        fontweight="bold",
        pad=12,
        color=PALETTE["ink"],
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5, color=PALETTE["line"])
    ax.legend(loc="lower right", fontsize=8)

    # Annotate empirical values
    for idx, (emp, cfg) in enumerate(zip(empirical_vals, configured_vals)):
        ax.text(
            max(emp, cfg) + 0.6,
            y[idx],
            f"act: {emp:.1f}%",
            va="center",
            fontsize=7.5,
            color=PALETTE["ink"],
        )

    plt.tight_layout()
    save_figure(fig, "eda_06_response_anomalies", output_dirs)


def generate_all_eda_figures(
    raw_df: pd.DataFrame,
    hidden_df: pd.DataFrame,
    stats: RealityCheckStats,
    output_dirs: List[Path],
) -> None:
    """Generate and save all 6 publication-ready exploratory figures."""
    plot_marginal_distributions(raw_df, output_dirs)
    plot_indicator_correlations(raw_df, output_dirs)
    plot_temporal_archetypes(hidden_df, output_dirs)
    plot_at_risk_by_archetype(hidden_df, output_dirs)
    plot_missingness_map(raw_df, output_dirs)
    plot_response_anomalies(stats, output_dirs)
