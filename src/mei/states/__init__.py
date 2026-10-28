"""Engagement states, trajectory patterns, and unsupervised sequence modeling.

This package provides:
- Clustering on longitudinal composite trajectories (KMeans, GMM, Ward)
- Gaussian Hidden Markov Models (hmmlearn) over short indicator sequences
- Deterministic pattern and overall level labeling
- Scikit-learn compatible StateFeatureTransformer with strict fold-wise isolation (Rule G3)
- Unsupervised archetype recovery and predictive value ablation benchmarks
- Brand-palette publication figures
"""

from mei.states.clustering import (
    ClusteringResult,
    evaluate_clusterings,
    extract_composite_trajectory_matrix,
    profile_cluster,
)
from mei.states.evaluation import (
    AblationSummary,
    GroundTruthRecoveryResult,
    evaluate_ground_truth_recovery,
    run_state_ablation_cv,
)
from mei.states.figures import (
    plot_at_risk_by_pattern,
    plot_cluster_silhouette,
    plot_emission_profiles,
    plot_k_selection_curves,
    plot_pattern_distribution,
    plot_trajectories_by_pattern,
    plot_transition_matrix,
)
from mei.states.hmm import (
    HMMModelSelectionResult,
    decode_hmm_sequences,
    fit_gaussian_hmm_with_restarts,
    prepare_sequence_data,
    select_optimal_hmm,
)
from mei.states.patterns import (
    PatternClassification,
    classify_all_trajectories,
    classify_trajectory_pattern,
)
from mei.states.transformers import StateFeatureTransformer

__all__ = [
    "ClusteringResult",
    "evaluate_clusterings",
    "extract_composite_trajectory_matrix",
    "profile_cluster",
    "HMMModelSelectionResult",
    "select_optimal_hmm",
    "fit_gaussian_hmm_with_restarts",
    "prepare_sequence_data",
    "decode_hmm_sequences",
    "PatternClassification",
    "classify_trajectory_pattern",
    "classify_all_trajectories",
    "StateFeatureTransformer",
    "GroundTruthRecoveryResult",
    "AblationSummary",
    "evaluate_ground_truth_recovery",
    "run_state_ablation_cv",
    "plot_emission_profiles",
    "plot_transition_matrix",
    "plot_trajectories_by_pattern",
    "plot_pattern_distribution",
    "plot_at_risk_by_pattern",
    "plot_cluster_silhouette",
    "plot_k_selection_curves",
]
