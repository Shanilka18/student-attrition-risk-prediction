"""Stability-first profile model selection and deterministic profile naming."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.profiles.clustering import (
    CLUSTER_ALGORITHMS,
    ClusterFit,
    evaluate_clustering_candidates,
)

DEFAULT_INDEX_COLUMNS = (
    "financial_stress_index",
    "commute_burden_index",
    "digital_access_gap_index",
    "study_environment_pressure_index",
)

_PROFILE_LABELS = {
    "financial_stress_index": "Mainly financial strain",
    "commute_burden_index": "Long commute",
    "digital_access_gap_index": "Digital access gaps",
    "study_environment_pressure_index": "Study environment pressure",
}


def canonical_profile_order(
    centers: np.ndarray,
    index_columns: tuple[str, ...] = DEFAULT_INDEX_COLUMNS,
    low_pressure_maximum: float = 0.35,
    high_pressure_minimum: float = 0.60,
    multi_pressure_minimum: float = 0.55,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Name centroids from pressure patterns and return a stable profile order."""
    values = np.asarray(centers, dtype=float)
    if values.ndim != 2 or values.shape[1] < len(index_columns):
        raise ValueError("Centroids must contain all configured index dimensions")
    if len(index_columns) != len(_PROFILE_LABELS):
        raise ValueError("Exactly four index names are required for profile naming")

    named_profiles = []
    for cluster, centroid in enumerate(values):
        pressures = centroid[: len(index_columns)]
        ordered_indices = np.argsort(-pressures, kind="stable")
        highest = float(pressures[ordered_indices[0]])
        second_highest = float(pressures[ordered_indices[1]])
        if highest < low_pressure_maximum:
            name = "Few pressures"
        elif (
            highest >= high_pressure_minimum
            and second_highest >= multi_pressure_minimum
        ):
            name = "Several pressures at once"
        else:
            name = _PROFILE_LABELS[index_columns[int(ordered_indices[0])]]
        named_profiles.append((name, tuple(np.round(centroid, 12)), cluster))

    named_profiles.sort(key=lambda item: (item[0], item[1]))
    order = np.asarray([item[2] for item in named_profiles], dtype=int)
    names = tuple(item[0] for item in named_profiles)
    return order, names


def select_profile_candidate(
    values: np.ndarray,
    settings: dict[str, Any] | None = None,
    index_columns: tuple[str, ...] = DEFAULT_INDEX_COLUMNS,
    centers_for_naming: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[ClusterFit, dict[str, Any]]:
    """Choose the most stable candidate satisfying the silhouette minimum."""
    config = settings or {}
    k_values = tuple(int(value) for value in config.get("k_values", range(2, 7)))
    algorithms = tuple(config.get("algorithms", CLUSTER_ALGORITHMS))
    silhouette_minimum = float(config.get("silhouette_minimum", 0.15))
    candidates = evaluate_clustering_candidates(
        values,
        k_values=k_values,
        algorithms=algorithms,
        bootstrap_resamples=int(config.get("bootstrap_resamples", 30)),
        bootstrap_fraction=float(config.get("bootstrap_fraction", 0.8)),
        random_state=int(config.get("random_seed", 42)),
    )
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["metrics"]["silhouette"]) >= silhouette_minimum
    ]
    if not eligible:
        raise ValueError(
            "No profile clustering candidate met the minimum silhouette "
            f"({silhouette_minimum:.3f})"
        )

    algorithm_rank = {algorithm: rank for rank, algorithm in enumerate(algorithms)}
    selected = min(
        eligible,
        key=lambda candidate: (
            -float(candidate["metrics"]["stability_ari_mean"]),
            int(candidate["metrics"]["k"]),
            algorithm_rank[str(candidate["metrics"]["algorithm"])],
        ),
    )
    fitted: ClusterFit = selected["fit"]
    metrics = selected["metrics"]
    naming_centers = (
        fitted.centers
        if centers_for_naming is None
        else centers_for_naming(fitted.centers)
    )
    order, names = canonical_profile_order(
        naming_centers,
        index_columns=index_columns,
        low_pressure_maximum=float(config.get("low_pressure_maximum", 0.35)),
        high_pressure_minimum=float(config.get("high_pressure_minimum", 0.60)),
        multi_pressure_minimum=float(config.get("multi_pressure_minimum", 0.55)),
    )
    fitted.label_order = order
    fitted.centers = fitted.centers[order]
    fitted.profile_names_ = names
    candidate_metrics = [candidate["metrics"] for candidate in candidates]
    return fitted, {
        "chosen_k": fitted.n_clusters,
        "chosen_algorithm": fitted.algorithm,
        "chosen_profile_names": list(names),
        "selected_metrics": metrics,
        "candidate_metrics": candidate_metrics,
        "silhouette_minimum": silhouette_minimum,
        "selection_rule": "highest bootstrap stability among candidates meeting the silhouette minimum; ties prefer smaller k",
    }
