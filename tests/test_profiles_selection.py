"""Tests for deterministic profile names and selection rules."""

from __future__ import annotations

import numpy as np

from src.profiles.clustering import ClusterFit
from src.profiles.selection import canonical_profile_order, select_profile_candidate


def test_profile_names_and_order_follow_centroid_pressures():
    centers = np.asarray(
        [
            [0.85, 0.2, 0.3, 0.2, 0.38],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.8, 0.25, 0.2, 0.36],
            [0.7, 0.65, 0.25, 0.2, 0.45],
        ]
    )

    order, names = canonical_profile_order(centers)
    shuffled_order, shuffled_names = canonical_profile_order(centers[[2, 0, 3, 1]])

    assert names == (
        "Few pressures",
        "Long commute",
        "Mainly financial strain",
        "Several pressures at once",
    )
    assert shuffled_names == names
    assert np.array_equal(order, np.asarray([1, 2, 0, 3]))
    assert np.array_equal(shuffled_order, np.asarray([3, 0, 1, 2]))


def test_selection_prefers_smaller_k_when_stability_ties(monkeypatch):
    centers = np.asarray(
        [
            [0.8, 0.1, 0.1, 0.1, 0.3],
            [0.1, 0.8, 0.1, 0.1, 0.3],
            [0.8, 0.8, 0.1, 0.1, 0.45],
        ]
    )
    candidates = []
    for n_clusters in (3, 2):
        fit = ClusterFit("kmeans", n_clusters, object(), centers[:n_clusters].copy())
        candidates.append(
            {
                "fit": fit,
                "metrics": {
                    "algorithm": "kmeans",
                    "k": n_clusters,
                    "silhouette": 0.4,
                    "stability_ari_mean": 0.9,
                },
            }
        )
    monkeypatch.setattr(
        "src.profiles.selection.evaluate_clustering_candidates",
        lambda *args, **kwargs: candidates,
    )

    fitted, summary = select_profile_candidate(
        np.zeros((10, 5)),
        settings={"algorithms": ["kmeans"], "silhouette_minimum": 0.15},
    )

    assert fitted.n_clusters == 2
    assert summary["chosen_k"] == 2
