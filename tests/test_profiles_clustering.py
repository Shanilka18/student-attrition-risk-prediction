"""Tests for profile clustering candidate metrics."""

from __future__ import annotations

import numpy as np

from src.profiles.clustering import evaluate_cluster_candidate


def test_candidate_metrics_detect_separated_toy_mixture():
    rng = np.random.default_rng(42)
    values = np.vstack(
        [
            rng.normal((-3.0, -3.0), 0.25, size=(60, 2)),
            rng.normal((3.0, 3.0), 0.25, size=(60, 2)),
        ]
    )

    fitted, metrics = evaluate_cluster_candidate(
        values,
        "kmeans",
        n_clusters=2,
        bootstrap_resamples=5,
    )

    assert fitted.n_clusters == 2
    assert metrics["silhouette"] > 0.8
    assert metrics["stability_ari_mean"] > 0.9
    assert metrics["calinski_harabasz"] > 1000
    assert metrics["davies_bouldin"] < 0.5
