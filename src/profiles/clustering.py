"""Candidate clustering algorithms and diagnostics for condition profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

CLUSTER_ALGORITHMS = ("kmeans", "gmm_full", "gmm_diag", "agglomerative_ward")


@dataclass
class ClusterFit:
    """A fitted clusterer plus centroids used for prediction when needed."""

    algorithm: str
    n_clusters: int
    estimator: Any
    centers: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        if self.algorithm == "kmeans":
            return self.estimator.predict(values)
        if self.algorithm.startswith("gmm_"):
            return self.estimator.predict(values)
        distances = np.linalg.norm(
            values[:, np.newaxis, :] - self.centers[np.newaxis, :, :], axis=2
        )
        return distances.argmin(axis=1)

    def membership_probabilities(self, values: np.ndarray) -> np.ndarray | None:
        if self.algorithm.startswith("gmm_"):
            return self.estimator.predict_proba(values)
        return None


def _new_estimator(algorithm: str, n_clusters: int, random_state: int) -> Any:
    if algorithm == "kmeans":
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    if algorithm == "gmm_full":
        return GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_state,
        )
    if algorithm == "gmm_diag":
        return GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            reg_covar=1e-6,
            random_state=random_state,
        )
    if algorithm == "agglomerative_ward":
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    raise ValueError(f"Unknown profile clustering algorithm: {algorithm}")


def _fit_candidate(
    values: np.ndarray,
    algorithm: str,
    n_clusters: int,
    random_state: int,
) -> tuple[ClusterFit, np.ndarray]:
    estimator = _new_estimator(algorithm, n_clusters, random_state)
    labels = estimator.fit_predict(values)
    centers = np.vstack(
        [values[labels == cluster].mean(axis=0) for cluster in range(n_clusters)]
    )
    return ClusterFit(algorithm, n_clusters, estimator, centers), labels


def evaluate_cluster_candidate(
    values: np.ndarray,
    algorithm: str,
    n_clusters: int,
    bootstrap_resamples: int = 30,
    bootstrap_fraction: float = 0.8,
    random_state: int = 42,
) -> tuple[ClusterFit, dict[str, float | int | None]]:
    """Fit a candidate and measure separation, information criterion, and stability."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Profile inputs must be a finite two-dimensional matrix")
    if n_clusters < 2 or n_clusters >= len(matrix):
        raise ValueError("Cluster count must be between 2 and n_rows - 1")
    if bootstrap_resamples < 1:
        raise ValueError("At least one bootstrap resample is required")
    if not 0 < bootstrap_fraction < 1:
        raise ValueError("bootstrap_fraction must be between 0 and 1")

    fitted, labels = _fit_candidate(matrix, algorithm, n_clusters, random_state)
    observed_clusters = np.unique(labels)
    if len(observed_clusters) < 2:
        raise ValueError(f"{algorithm} produced fewer than two clusters")
    silhouette = float(silhouette_score(matrix, labels))
    stability_scores = []
    rng = np.random.default_rng(random_state)
    sample_size = max(n_clusters + 1, int(np.ceil(len(matrix) * bootstrap_fraction)))
    for bootstrap_number in range(bootstrap_resamples):
        positions = rng.integers(0, len(matrix), size=sample_size)
        _, bootstrap_labels = _fit_candidate(
            matrix[positions],
            algorithm,
            n_clusters,
            random_state + bootstrap_number + 1,
        )
        reference_labels = fitted.predict(matrix[positions])
        stability_scores.append(
            float(adjusted_rand_score(reference_labels, bootstrap_labels))
        )

    bic: float | None = None
    if algorithm.startswith("gmm_"):
        bic = float(fitted.estimator.bic(matrix))
    metrics: dict[str, float | int | None] = {
        "algorithm": algorithm,
        "k": n_clusters,
        "silhouette": silhouette,
        "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
        "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
        "bic": bic,
        "stability_ari_mean": float(np.mean(stability_scores)),
        "stability_ari_std": (
            float(np.std(stability_scores, ddof=1))
            if len(stability_scores) > 1
            else 0.0
        ),
    }
    return fitted, metrics


def evaluate_clustering_candidates(
    values: np.ndarray,
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6),
    algorithms: tuple[str, ...] = CLUSTER_ALGORITHMS,
    bootstrap_resamples: int = 30,
    bootstrap_fraction: float = 0.8,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate every configured algorithm and cluster count deterministically."""
    matrix = np.asarray(values, dtype=float)
    candidates = []
    for algorithm_index, algorithm in enumerate(algorithms):
        for n_clusters in k_values:
            if n_clusters >= len(matrix):
                continue
            fitted, metrics = evaluate_cluster_candidate(
                matrix,
                algorithm,
                n_clusters,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_fraction=bootstrap_fraction,
                random_state=random_state + algorithm_index * 100 + n_clusters,
            )
            candidates.append({"fit": fitted, "metrics": metrics})
    if not candidates:
        raise ValueError("No valid profile clustering candidates were configured")
    return candidates
