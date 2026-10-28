"""Scikit-learn compatible transformer for student engagement latent states.

Integrates longitudinal clustering and Gaussian Hidden Markov Model decoding into a
single pipeline component. Strictly fitted on training data inside CV folds (Rule G3).
Extracts state sequences, dwell times, transitions, posterior probabilities, sequence
log-likelihoods, cluster assignments, centroid distances, patterns, and levels.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from mei.features.transformers import CANONICAL_INDICATORS, EngagementImputer
from mei.states.clustering import extract_composite_trajectory_matrix
from mei.states.hmm import (
    decode_hmm_sequences,
    fit_gaussian_hmm_with_restarts,
    prepare_sequence_data,
)
from mei.states.patterns import classify_trajectory_pattern


class StateFeatureTransformer(BaseEstimator, TransformerMixin):
    """Transformer extracting latent state, trajectory pattern, and clustering features.

    Parameters:
    - clustering_algo: 'kmeans' | 'gmm_diag' | 'gmm_full' | 'ward'
    - n_clusters: Number of clusters k (default 4)
    - hmm_k: Number of HMM latent states K (default 3)
    - n_hmm_restarts: Number of random restarts for HMM (default 10)
    - random_state: Random seed for reproducibility (default 42)
    """

    def __init__(
        self,
        clustering_algo: str = "kmeans",
        n_clusters: int = 4,
        hmm_k: int = 3,
        n_hmm_restarts: int = 10,
        random_state: int = 42,
    ):
        self.clustering_algo = clustering_algo
        self.n_clusters = n_clusters
        self.hmm_k = hmm_k
        self.n_hmm_restarts = n_hmm_restarts
        self.random_state = random_state

        # Fitted state components
        self.imputer_: Optional[EngagementImputer] = None
        self.clustering_scaler_: Optional[StandardScaler] = None
        self.clustering_model_: Any = None
        self.cluster_centroids_: Optional[np.ndarray] = None  # Standardized centroids
        self.cluster_names_: Dict[int, str] = {}

        self.hmm_scaler_: Optional[StandardScaler] = None
        self.hmm_model_: Any = None
        self.feature_names_: List[str] = []

    def fit(self, X: Any, y: Any = None) -> "StateFeatureTransformer":
        """Fit imputer, clustering model, and GaussianHMM exclusively on training data."""
        expected_cols = [
            f"{ind}_p{p}" for ind in CANONICAL_INDICATORS for p in [1, 2, 3, 4]
        ]
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X, columns=expected_cols[: X.shape[1]])
        else:
            present_cols = [c for c in expected_cols if c in X.columns]
            df = X[present_cols].copy()

        # 1. Fit Imputer
        self.imputer_ = EngagementImputer()
        imputed_df = self.imputer_.fit_transform(df)

        # 2. Fit Clustering on composite trajectories
        X_comp_raw, _ = extract_composite_trajectory_matrix(imputed_df)
        self.clustering_scaler_ = StandardScaler()
        X_comp_scaled = self.clustering_scaler_.fit_transform(X_comp_raw)

        if self.clustering_algo == "kmeans":
            from sklearn.cluster import KMeans

            self.clustering_model_ = KMeans(
                n_clusters=self.n_clusters,
                n_init=10,
                random_state=self.random_state,
            )
            labels = self.clustering_model_.fit_predict(X_comp_scaled)
            self.cluster_centroids_ = self.clustering_model_.cluster_centers_
        elif self.clustering_algo == "gmm_diag":
            from sklearn.mixture import GaussianMixture

            self.clustering_model_ = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type="diag",
                n_init=5,
                random_state=self.random_state,
            )
            labels = self.clustering_model_.fit_predict(X_comp_scaled)
            self.cluster_centroids_ = self.clustering_model_.means_
        elif self.clustering_algo == "gmm_full":
            from sklearn.mixture import GaussianMixture

            self.clustering_model_ = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type="full",
                n_init=5,
                random_state=self.random_state,
            )
            labels = self.clustering_model_.fit_predict(X_comp_scaled)
            self.cluster_centroids_ = self.clustering_model_.means_
        else:  # ward
            from sklearn.cluster import AgglomerativeClustering

            self.clustering_model_ = AgglomerativeClustering(
                n_clusters=self.n_clusters, linkage="ward"
            )
            labels = self.clustering_model_.fit_predict(X_comp_scaled)
            self.cluster_centroids_ = np.zeros(
                (self.n_clusters, X_comp_scaled.shape[1])
            )
            for c in range(self.n_clusters):
                mask = labels == c
                if np.any(mask):
                    self.cluster_centroids_[c] = X_comp_scaled[mask].mean(axis=0)

        # 3. Fit GaussianHMM on 7-indicator sequences
        X_seq_scaled, lengths, self.hmm_scaler_, _ = prepare_sequence_data(
            imputed_df, fit_scaler=True
        )
        self.hmm_model_, _, _ = fit_gaussian_hmm_with_restarts(
            X_seq_scaled,
            lengths,
            k=self.hmm_k,
            n_restarts=self.n_hmm_restarts,
            max_iter=100,
            random_seed=self.random_state,
        )

        # 4. Define feature names
        names = []
        # State paths
        names.extend(["state_p1", "state_p2", "state_p3", "state_p4"])
        names.append("final_state")
        names.append("n_transitions")
        for k_idx in range(1, self.hmm_k + 1):
            names.append(f"time_in_state_{k_idx}")
        for k_idx in range(1, self.hmm_k + 1):
            names.append(f"posterior_p4_state_{k_idx}")
        names.append("sequence_log_likelihood")

        # Cluster features
        for c in range(self.n_clusters):
            names.append(f"cluster_{c}")
        names.append("dist_to_cluster_center")

        # Pattern one-hot
        for pat in ["consistent", "declining", "improving", "sporadic"]:
            names.append(f"pattern_{pat}")

        # Level one-hot
        for lvl in ["low", "medium", "high"]:
            names.append(f"level_{lvl}")

        self.feature_names_ = names
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """Extract all latent state, trajectory pattern, and clustering features."""
        if (
            self.imputer_ is None
            or self.clustering_model_ is None
            or self.hmm_model_ is None
        ):
            raise RuntimeError(
                "StateFeatureTransformer must be fitted before calling transform."
            )

        expected_cols = [
            f"{ind}_p{p}" for ind in CANONICAL_INDICATORS for p in [1, 2, 3, 4]
        ]
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X, columns=expected_cols[: X.shape[1]])
        else:
            present_cols = [c for c in expected_cols if c in X.columns]
            df = X[present_cols].copy()

        # 1. Impute using fitted imputer
        imputed_df = self.imputer_.transform(df)

        # 2. Clustering inference
        X_comp_raw, _ = extract_composite_trajectory_matrix(imputed_df)
        X_comp_scaled = self.clustering_scaler_.transform(X_comp_raw)

        if hasattr(self.clustering_model_, "predict"):
            cluster_preds = self.clustering_model_.predict(X_comp_scaled)
        else:
            cluster_preds = np.argmin(
                cdist(X_comp_scaled, self.cluster_centroids_), axis=1
            )

        n_samples = len(df)
        # Cluster one-hot and distance
        cluster_one_hot = np.zeros((n_samples, self.n_clusters), dtype=float)
        dist_to_center = np.zeros(n_samples, dtype=float)
        for i in range(n_samples):
            c = cluster_preds[i]
            cluster_one_hot[i, c] = 1.0
            dist_to_center[i] = np.linalg.norm(
                X_comp_scaled[i] - self.cluster_centroids_[c]
            )

        # 3. HMM inference
        X_seq_scaled, lengths, _, _ = prepare_sequence_data(
            imputed_df, scaler=self.hmm_scaler_, fit_scaler=False
        )
        viterbi_paths, posteriors, seq_ll = decode_hmm_sequences(
            self.hmm_model_, X_seq_scaled, lengths
        )

        final_state = viterbi_paths[:, 3].astype(float)

        # Number of transitions between consecutive periods
        n_transitions = np.sum(
            viterbi_paths[:, 1:] != viterbi_paths[:, :-1], axis=1
        ).astype(float)

        # Fraction of time spent in each state (1..K)
        time_in_state = np.zeros((n_samples, self.hmm_k), dtype=float)
        for k_idx in range(1, self.hmm_k + 1):
            time_in_state[:, k_idx - 1] = np.mean(viterbi_paths == k_idx, axis=1)

        # Posterior probabilities at period 4 (0-indexed period 3)
        posteriors_p4 = posteriors[:, 3, :]

        # 4. Pattern and Level classification
        pattern_one_hot = np.zeros((n_samples, 4), dtype=float)
        pattern_order = ["consistent", "declining", "improving", "sporadic"]
        level_one_hot = np.zeros((n_samples, 3), dtype=float)
        level_order = ["low", "medium", "high"]

        for i in range(n_samples):
            cls_res = classify_trajectory_pattern(viterbi_paths[i], k=self.hmm_k)
            pat_idx = pattern_order.index(cls_res.pattern.lower())
            pattern_one_hot[i, pat_idx] = 1.0

            lvl_str = cls_res.level.lower()
            if lvl_str in level_order:
                lvl_idx = level_order.index(lvl_str)
                level_one_hot[i, lvl_idx] = 1.0

        # 5. Assemble all features
        feature_matrix = np.column_stack(
            [
                viterbi_paths.astype(float),  # 4 features
                final_state,  # 1
                n_transitions,  # 1
                time_in_state,  # K
                posteriors_p4,  # K
                seq_ll,  # 1
                cluster_one_hot,  # n_clusters
                dist_to_center,  # 1
                pattern_one_hot,  # 4
                level_one_hot,  # 3
            ]
        )

        return pd.DataFrame(feature_matrix, columns=self.feature_names_, index=df.index)

    def get_feature_names_out(self, input_features: Optional[Any] = None) -> List[str]:
        """Return generated state feature names."""
        return list(self.feature_names_)
