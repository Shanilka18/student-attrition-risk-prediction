"""
Data Audit and Quality Visualization Suite.
Produces high-resolution diagnostic plots for synthetic academic cohort data.

Component: Academic Intelligence Framework (IT23164680)
"""

from typing import Dict, Any
from pathlib import Path
import logging
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger("AcademicIntelligence.DataAudit")

# Research styling palette
sns.set_theme(style="whitegrid", font="sans-serif")
PALETTE = ["#4f46e5", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e", "#8b5cf6", "#64748b"]


def run_data_audit_suite(
    df_long: pd.DataFrame,
    df_outcomes: pd.DataFrame,
    meta: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Path]:
    """
    Generates the six canonical data audit plots and saves them to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: Dict[str, Path] = {}

    # Merge long format with outcomes for joint plotting
    merged = pd.merge(df_long, df_outcomes, on="student_id", how="left")
    audit_df = meta.get("audit_df")

    # 1. Distributions Plot
    p1 = output_dir / "distributions.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Observed Academic Feature Distributions [SYNTHETIC DATASET]", fontsize=14, fontweight="bold", y=0.98)

    sns.histplot(df_long["gpa"].dropna(), kde=True, ax=axes[0, 0], color="#4f46e5", bins=30)
    axes[0, 0].set_title("Semester GPA (0.0 - 4.0)")
    axes[0, 0].set_xlabel("GPA")

    sns.histplot(df_long["exam_marks"].dropna(), kde=True, ax=axes[0, 1], color="#06b6d4", bins=30)
    axes[0, 1].set_title("Average Examination Marks (%)")
    axes[0, 1].set_xlabel("Marks (%)")

    sns.histplot(df_long["attendance_pct"].dropna(), kde=True, ax=axes[1, 0], color="#10b981", bins=30)
    axes[1, 0].set_title("Attendance Rate (%)")
    axes[1, 0].set_xlabel("Attendance (%)")

    sns.histplot(df_long["assignment_delay_days"].dropna(), kde=True, ax=axes[1, 1], color="#f43f5e", bins=30)
    axes[1, 1].set_title("Assignment Submission Delay (Days)")
    axes[1, 1].set_xlabel("Days Late")

    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()
    plot_paths["distributions"] = p1

    # 2. Correlation Heatmap
    p2 = output_dir / "correlation_heatmap.png"
    plt.figure(figsize=(8, 6))
    num_cols = ["gpa", "exam_marks", "attendance_pct", "assignment_delay_days", "late_assignments_count"]
    corr = df_long[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, cbar_kws={"label": "Pearson Correlation"})
    plt.title("Correlation Matrix of Observable Academic Signals [SYNTHETIC]", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()
    plot_paths["correlation_heatmap"] = p2

    # 3. Trajectories by Type
    p3 = output_dir / "trajectories_by_type.png"
    if audit_df is not None:
        merged_audit = pd.merge(df_long, audit_df[["student_id", "trajectory_type"]].drop_duplicates(), on="student_id")
        plt.figure(figsize=(12, 7))
        types = merged_audit["trajectory_type"].unique()
        for idx, ttype in enumerate(types):
            sample_sids = merged_audit[merged_audit["trajectory_type"] == ttype]["student_id"].unique()[:3]
            for sid in sample_sids:
                s_data = merged_audit[merged_audit["student_id"] == sid].sort_values("semester")
                plt.plot(
                    s_data["semester"], s_data["gpa"],
                    marker="o", alpha=0.65, label=ttype if sid == sample_sids[0] else "",
                    color=PALETTE[idx % len(PALETTE)], linewidth=2
                )
        plt.title("Sample GPA Trajectories by Latent Trajectory Type [SYNTHETIC]", fontsize=12, fontweight="bold")
        plt.xlabel("Semester Index")
        plt.ylabel("Semester GPA")
        plt.ylim(0.5, 4.2)
        plt.legend(title="Latent Type", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(p3, dpi=300)
        plt.close()
        plot_paths["trajectories_by_type"] = p3

    # 4. Attrition by Trajectory Type
    p4 = output_dir / "attrition_by_type.png"
    if audit_df is not None:
        student_types = audit_df[["student_id", "trajectory_type"]].drop_duplicates()
        merged_out = pd.merge(df_outcomes, student_types, on="student_id")
        attr_rates = merged_out.groupby("trajectory_type")["attrited"].mean().reset_index()
        attr_rates["attrition_pct"] = attr_rates["attrited"] * 100

        plt.figure(figsize=(10, 5))
        barplot = sns.barplot(
            data=attr_rates, x="trajectory_type", y="attrition_pct",
            hue="trajectory_type", palette="magma", legend=False
        )
        plt.title("Attrition Rate by Latent Trajectory Archetype [SYNTHETIC]", fontsize=12, fontweight="bold")
        plt.xlabel("Latent Trajectory Type")
        plt.ylabel("Attrition Rate (%)")
        plt.xticks(rotation=25)
        for p in barplot.patches:
            barplot.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=10)
        plt.tight_layout()
        plt.savefig(p4, dpi=300)
        plt.close()
        plot_paths["attrition_by_type"] = p4

    # 5. Missingness Map
    p5 = output_dir / "missingness_map.png"
    plt.figure(figsize=(8, 4))
    miss_pct = (df_long.isnull().sum() / len(df_long)) * 100
    miss_df = pd.DataFrame({"Feature": miss_pct.index, "Missing_Pct": miss_pct.values})
    miss_df = miss_df[miss_df["Missing_Pct"] > 0]
    if len(miss_df) == 0:
        miss_df = pd.DataFrame({"Feature": ["None"], "Missing_Pct": [0.0]})
    sns.barplot(data=miss_df, x="Feature", y="Missing_Pct", color="#6366f1")
    plt.title("Missing Value Frequency (%) by Observable Feature [SYNTHETIC]", fontsize=12, fontweight="bold")
    plt.ylabel("Missing Percentage (%)")
    plt.ylim(0, max(6.0, miss_df["Missing_Pct"].max() * 1.3))
    plt.tight_layout()
    plt.savefig(p5, dpi=300)
    plt.close()
    plot_paths["missingness_map"] = p5

    # 6. Cohort GPA Trends Over Semesters
    p6 = output_dir / "cohort_gpa_trends.png"
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=merged, x="semester", y="gpa", hue="attrited",
        palette={0: "#10b981", 1: "#f43f5e"}, style="attrited", markers=True, dashes=False
    )
    plt.title("Longitudinal GPA Progression: Persisting vs Attrited Students [SYNTHETIC]", fontsize=12, fontweight="bold")
    plt.xlabel("Semester Index")
    plt.ylabel("Semester GPA")
    plt.legend(title="Cohort Outcome", labels=["Persisted (0)", "Attrited (1)"])
    plt.tight_layout()
    plt.savefig(p6, dpi=300)
    plt.close()
    plot_paths["cohort_gpa_trends"] = p6

    logger.info(f"Generated {len(plot_paths)} data audit plots in {output_dir}")
    return plot_paths
