"""Generate interpretable figures for discovered condition profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from src.profiles.selection import DEFAULT_INDEX_COLUMNS


def _profile_colors(count: int, palette: dict[str, str]) -> list[str]:
    base = [
        palette.get("emerald", "#059669"),
        palette.get("teal", "#0f766e"),
        palette.get("purple", "#7c3aed"),
        palette.get("amber", "#d97706"),
        palette.get("red", "#dc2626"),
    ]
    if count <= len(base):
        return base[:count]
    color_map = plt.get_cmap("viridis")
    return [color_map(value) for value in np.linspace(0.08, 0.92, count)]


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = (
        z * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    )
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def save_profile_figures(
    standardized_training: np.ndarray,
    training_labels: np.ndarray,
    index_means: pd.DataFrame,
    profile_statistics: pd.DataFrame,
    candidate_metrics: list[dict[str, Any]],
    profile_names: tuple[str, ...],
    output_dir: str | Path,
    palette: dict[str, str],
) -> list[Path]:
    """Write six profile, quality, selection, and projection figures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    k = len(profile_names)
    colors = _profile_colors(k, palette)
    figures = []

    fig, axis = plt.subplots(figsize=(max(8, k * 1.5), 5))
    image = axis.imshow(
        index_means.loc[:, DEFAULT_INDEX_COLUMNS].to_numpy().T,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axis.set_xticks(np.arange(k), profile_names, rotation=25, ha="right")
    axis.set_yticks(
        np.arange(len(DEFAULT_INDEX_COLUMNS)),
        [
            "Financial Stress",
            "Commute Burden",
            "Digital Access Gap",
            "Study Environment Pressure",
        ],
    )
    axis.set_title("Mean study-condition pressure by profile")
    fig.colorbar(image, ax=axis, label="Mean pressure (0-1)")
    fig.tight_layout()
    path = output / "profile_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        profile_names,
        profile_statistics["students"],
        color=colors,
    )
    axis.set(title="Student count by discovered profile", ylabel="Students")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = output / "profile_sizes.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    rates = profile_statistics["at_risk_rate"].to_numpy(dtype=float)
    lower = profile_statistics["at_risk_ci_low"].to_numpy(dtype=float)
    upper = profile_statistics["at_risk_ci_high"].to_numpy(dtype=float)
    positions = np.arange(k)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.errorbar(
        positions,
        rates,
        yerr=np.vstack([rates - lower, upper - rates]),
        fmt="o",
        capsize=5,
        color=palette.get("emerald", "#059669"),
    )
    axis.set_xticks(positions, profile_names, rotation=25, ha="right")
    axis.set(
        title="At-risk share by profile (95% Wilson intervals)",
        ylabel="At-risk share",
        ylim=(0, 1),
    )
    fig.tight_layout()
    path = output / "at_risk_by_profile.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    samples = silhouette_samples(standardized_training, training_labels)
    overall = silhouette_score(standardized_training, training_labels)
    fig, axis = plt.subplots(figsize=(9, 6))
    lower_bound = 10
    for cluster in range(k):
        cluster_scores = np.sort(samples[training_labels == cluster])
        upper_bound = lower_bound + len(cluster_scores)
        axis.fill_betweenx(
            np.arange(lower_bound, upper_bound),
            0,
            cluster_scores,
            facecolor=colors[cluster],
            alpha=0.75,
        )
        axis.text(-0.05, lower_bound + len(cluster_scores) / 2, profile_names[cluster])
        lower_bound = upper_bound + 10
    axis.axvline(overall, color="black", linestyle="--", linewidth=1)
    axis.set(
        title="Silhouette values for selected training profiles",
        xlabel="Silhouette coefficient",
        ylabel="Students grouped by profile",
    )
    fig.tight_layout()
    path = output / "profile_silhouette.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    candidates = pd.DataFrame(candidate_metrics)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for algorithm, group in candidates.groupby("algorithm", sort=True):
        group = group.sort_values("k")
        axes[0].plot(
            group["k"],
            group["stability_ari_mean"],
            marker="o",
            label=algorithm,
        )
        axes[1].plot(
            group["k"],
            group["silhouette"],
            marker="o",
            label=algorithm,
        )
    axes[0].set(
        title="Bootstrap stability by cluster count", xlabel="k", ylabel="Mean ARI"
    )
    axes[1].set(
        title="Silhouette by cluster count",
        xlabel="k",
        ylabel="Silhouette",
    )
    for axis in axes:
        axis.set_xticks(sorted(candidates["k"].unique()))
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = output / "profile_k_selection.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)

    projection = PCA(n_components=2, random_state=42).fit_transform(
        standardized_training
    )
    fig, axis = plt.subplots(figsize=(8, 6))
    for cluster in range(k):
        selected = training_labels == cluster
        axis.scatter(
            projection[selected, 0],
            projection[selected, 1],
            s=18,
            alpha=0.7,
            color=colors[cluster],
            label=profile_names[cluster],
        )
    axis.set(
        title="Training students projected onto the first two PCA components",
        xlabel="Principal component 1",
        ylabel="Principal component 2",
    )
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = output / "profile_pca.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures.append(path)
    return figures


def profile_statistics(
    data: pd.DataFrame,
    profile_ids: np.ndarray,
    profile_names: tuple[str, ...],
) -> pd.DataFrame:
    """Summarize profile size and at-risk shares with Wilson confidence bounds."""
    records = []
    labels = data["at_risk"].astype(int).to_numpy()
    for profile_id, name in enumerate(profile_names):
        selected = profile_ids == profile_id
        count = int(selected.sum())
        at_risk_count = int(labels[selected].sum())
        lower, upper = _wilson_interval(at_risk_count, count)
        means = data.loc[selected, DEFAULT_INDEX_COLUMNS].mean()
        records.append(
            {
                "profile_id": profile_id,
                "profile_name": name,
                "students": count,
                "at_risk_count": at_risk_count,
                "at_risk_rate": at_risk_count / count if count else 0.0,
                "at_risk_ci_low": lower,
                "at_risk_ci_high": upper,
                **{column: float(means[column]) for column in DEFAULT_INDEX_COLUMNS},
            }
        )
    return pd.DataFrame(records)
