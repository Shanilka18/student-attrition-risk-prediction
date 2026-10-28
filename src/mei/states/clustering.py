"""Unsupervised student engagement clustering module.

Evaluates KMeans, Gaussian Mixture Models, and Agglomerative Ward clustering
across k in [2, 6] on longitudinal composite engagement trajectories.
Selects optimal clustering based on silhouette thresholds and bootstrap ARI stability.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Composite trajectory column definitions
COMPOSITE_COLS = ["composite_p1", "composite_p2", "composite_p3", "composite_p4"]


@dataclass
class ClusteringResult:
    """Container for clustering evaluation and selected model."""

    best_model: Any
    best_algorithm: str
    best_k: int
    scaler: StandardScaler
    cluster_names: Dict[int, str]
    centroids: np.ndarray  # (k, n_features) unstandardized
    standardized_centroids: np.ndarray  # (k, n_features)
    metrics_table: pd.DataFrame
    best_silhouette: float
    best_stability: float


def extract_composite_trajectory_matrix(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """Extract student-level longitudinal composite trajectory features.

    Features:
    1. composite_p1 (Period 1 composite engagement, 0-100)
    2. composite_p2 (Period 2 composite engagement, 0-100)
    3. composite_p3 (Period 3 composite engagement, 0-100)
    4. composite_p4 (Period 4 composite engagement, 0-100)
    5. composite_slope (OLS trajectory slope over periods 1..4)
    6. composite_volatility (Standard deviation across the 4 periods)
    7. composite_level (Mean composite engagement across the 4 periods)
    """
    feature_names = [
        "composite_p1",
        "composite_p2",
        "composite_p3",
        "composite_p4",
        "composite_slope",
        "composite_volatility",
        "composite_level",
    ]

    # Check if features are already present in df
    if all(col in df.columns for col in feature_names):
        return df[feature_names].values.astype(float), feature_names

    # If composite_std is present instead of composite_volatility, or composite_mean instead of level
    if all(col in df.columns for col in COMPOSITE_COLS):
        p_vals = df[COMPOSITE_COLS].values.astype(float)
    else:
        # Compute composites from raw 28 columns
        periods = [1, 2, 3, 4]
        indicators = [
            "resource_access",
            "study_time",
            "review_time",
            "quiz_practice",
            "assignment_engagement",
            "study_consistency",
            "participation",
        ]
        p_vals = np.zeros((len(df), 4), dtype=float)
        for p_idx, p in enumerate(periods):
            cols = [f"{ind}_p{p}" for ind in indicators if f"{ind}_p{p}" in df.columns]
            if not cols:
                raise ValueError(f"Cannot find indicator columns for period {p}")
            # Rescale (mean(1..5) - 1) / 4 * 100
            p_vals[:, p_idx] = ((df[cols].mean(axis=1).values - 1.0) / 4.0) * 100.0

    t = np.array([1.0, 2.0, 3.0, 4.0])
    t_center = t - t.mean()
    denom = np.sum(t_center**2)  # 5.0

    centered_y = p_vals - p_vals.mean(axis=1, keepdims=True)
    slope = np.sum(centered_y * t_center, axis=1) / denom
    volatility = np.std(p_vals, axis=1, ddof=1)
    volatility = np.nan_to_num(volatility, nan=0.0)
    level = np.mean(p_vals, axis=1)

    matrix = np.column_stack([p_vals, slope, volatility, level])
    return matrix, feature_names


def _compute_bootstrap_stability(
    model_factory: Any,
    X: np.ndarray,
    base_labels: np.ndarray,
    n_bootstrap: int = 30,
    random_state: int = 42,
) -> float:
    """Compute cluster bootstrap stability via mean Adjusted Rand Index.

    Draws n_bootstrap resamples with replacement, fits the candidate model on each
    resample, predicts cluster assignments on the full dataset, and measures
    the ARI between each bootstrap clustering and the base clustering.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(X)
    ari_scores = []

    for b in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[boot_idx]

        try:
            boot_model = model_factory(random_state=rng.randint(0, 100000))
            if hasattr(boot_model, "fit_predict"):
                boot_model.fit(X_boot)

            # Assign labels to original X
            if hasattr(boot_model, "predict"):
                pred_labels = boot_model.predict(X)
            else:
                # For AgglomerativeClustering (no predict), assign to nearest centroid
                boot_labels = getattr(boot_model, "labels_", None)
                if boot_labels is None:
                    boot_labels = boot_model.fit_predict(X_boot)
                k = len(np.unique(boot_labels))
                centroids = np.zeros((k, X.shape[1]))
                for c in range(k):
                    mask = boot_labels == c
                    if np.any(mask):
                        centroids[c] = X_boot[mask].mean(axis=0)
                    else:
                        centroids[c] = X_boot.mean(axis=0)
                pred_labels = np.argmin(cdist(X, centroids), axis=1)

            ari = adjusted_rand_score(base_labels, pred_labels)
            ari_scores.append(ari)
        except Exception as e:
            logger.debug("Bootstrap iteration %d failed: %s", b, str(e))
            continue

    if not ari_scores:
        return 0.0
    return float(np.mean(ari_scores))


def profile_cluster(
    centroid: np.ndarray,
    feature_names: List[str],
) -> str:
    """Assign descriptive name to cluster centroid based on level and trajectory shape.

    centroid features: [composite_p1..p4, slope, volatility, level]
    """
    feat_map = dict(zip(feature_names, centroid))
    level_val = feat_map.get("composite_level", centroid[6])
    slope_val = feat_map.get("composite_slope", centroid[4])
    vol_val = feat_map.get("composite_volatility", centroid[5])

    # Level classification
    if level_val >= 65.0:
        level_name = "High"
    elif level_val >= 35.0:
        level_name = "Moderate"
    else:
        level_name = "Low"

    # Shape classification
    if abs(slope_val) < 2.0 and vol_val < 10.0:
        shape_name = "Consistent"
    elif slope_val <= -2.0:
        shape_name = "Declining"
    elif slope_val >= 2.0:
        shape_name = "Improving"
    else:
        shape_name = "Sporadic"

    return f"{level_name} {shape_name}"


def evaluate_clusterings(
    X_raw: np.ndarray,
    feature_names: List[str],
    k_range: Optional[List[int]] = None,
    n_bootstrap: int = 30,
    silhouette_threshold: float = 0.15,
    random_seed: int = 42,
) -> ClusteringResult:
    """Evaluate clustering algorithms across k_range and select the best model.

    Algorithms evaluated:
    - KMeans
    - GaussianMixture (full covariance)
    - GaussianMixture (diag covariance)
    - AgglomerativeClustering (Ward linkage)

    Selection rule:
    Highest bootstrap ARI stability among candidates whose silhouette >= silhouette_threshold.
    Ties broken by smaller k.
    """
    if k_range is None:
        k_range = [2, 3, 4, 5, 6]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    records = []
    fitted_models: Dict[str, Any] = {}
    model_labels: Dict[str, np.ndarray] = {}

    for k in k_range:
        # 1. KMeans
        km_key = f"kmeans_k{k}"
        km = KMeans(n_clusters=k, n_init=10, random_state=random_seed)
        km_labels = km.fit_predict(X_scaled)
        fitted_models[km_key] = km
        model_labels[km_key] = km_labels

        sil = float(silhouette_score(X_scaled, km_labels))
        ch = float(calinski_harabasz_score(X_scaled, km_labels))
        db = float(davies_bouldin_score(X_scaled, km_labels))
        stab = _compute_bootstrap_stability(
            lambda random_state: KMeans(
                n_clusters=k, n_init=5, random_state=random_state
            ),
            X_scaled,
            km_labels,
            n_bootstrap=n_bootstrap,
            random_state=random_seed,
        )
        records.append(
            {
                "model_key": km_key,
                "algorithm": "kmeans",
                "k": k,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "bic": np.nan,
                "bootstrap_stability": stab,
            }
        )

        # 2. GMM Diag
        gmm_d_key = f"gmm_diag_k{k}"
        gmm_d = GaussianMixture(
            n_components=k, covariance_type="diag", n_init=5, random_state=random_seed
        )
        gmm_d_labels = gmm_d.fit_predict(X_scaled)
        fitted_models[gmm_d_key] = gmm_d
        model_labels[gmm_d_key] = gmm_d_labels

        sil_gd = float(silhouette_score(X_scaled, gmm_d_labels))
        ch_gd = float(calinski_harabasz_score(X_scaled, gmm_d_labels))
        db_gd = float(davies_bouldin_score(X_scaled, gmm_d_labels))
        bic_gd = float(gmm_d.bic(X_scaled))
        stab_gd = _compute_bootstrap_stability(
            lambda random_state: GaussianMixture(
                n_components=k,
                covariance_type="diag",
                n_init=2,
                random_state=random_state,
            ),
            X_scaled,
            gmm_d_labels,
            n_bootstrap=n_bootstrap,
            random_state=random_seed,
        )
        records.append(
            {
                "model_key": gmm_d_key,
                "algorithm": "gmm_diag",
                "k": k,
                "silhouette": sil_gd,
                "calinski_harabasz": ch_gd,
                "davies_bouldin": db_gd,
                "bic": bic_gd,
                "bootstrap_stability": stab_gd,
            }
        )

        # 3. GMM Full
        gmm_f_key = f"gmm_full_k{k}"
        gmm_f = GaussianMixture(
            n_components=k, covariance_type="full", n_init=5, random_state=random_seed
        )
        gmm_f_labels = gmm_f.fit_predict(X_scaled)
        fitted_models[gmm_f_key] = gmm_f
        model_labels[gmm_f_key] = gmm_f_labels

        sil_gf = float(silhouette_score(X_scaled, gmm_f_labels))
        ch_gf = float(calinski_harabasz_score(X_scaled, gmm_f_labels))
        db_gf = float(davies_bouldin_score(X_scaled, gmm_f_labels))
        bic_gf = float(gmm_f.bic(X_scaled))
        stab_gf = _compute_bootstrap_stability(
            lambda random_state: GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=2,
                random_state=random_state,
            ),
            X_scaled,
            gmm_f_labels,
            n_bootstrap=n_bootstrap,
            random_state=random_seed,
        )
        records.append(
            {
                "model_key": gmm_f_key,
                "algorithm": "gmm_full",
                "k": k,
                "silhouette": sil_gf,
                "calinski_harabasz": ch_gf,
                "davies_bouldin": db_gf,
                "bic": bic_gf,
                "bootstrap_stability": stab_gf,
            }
        )

        # 4. Agglomerative (Ward)
        ward_key = f"ward_k{k}"
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
        ward_labels = ward.fit_predict(X_scaled)
        fitted_models[ward_key] = ward
        model_labels[ward_key] = ward_labels

        sil_w = float(silhouette_score(X_scaled, ward_labels))
        ch_w = float(calinski_harabasz_score(X_scaled, ward_labels))
        db_w = float(davies_bouldin_score(X_scaled, ward_labels))
        stab_w = _compute_bootstrap_stability(
            lambda random_state: AgglomerativeClustering(n_clusters=k, linkage="ward"),
            X_scaled,
            ward_labels,
            n_bootstrap=n_bootstrap,
            random_state=random_seed,
        )
        records.append(
            {
                "model_key": ward_key,
                "algorithm": "ward",
                "k": k,
                "silhouette": sil_w,
                "calinski_harabasz": ch_w,
                "davies_bouldin": db_w,
                "bic": np.nan,
                "bootstrap_stability": stab_w,
            }
        )

    metrics_df = pd.DataFrame(records)

    # Selection rule: highest stability among candidates with silhouette >= threshold, ties -> smaller k
    qualifying = metrics_df[metrics_df["silhouette"] >= silhouette_threshold]
    if qualifying.empty:
        logger.warning(
            "No clustering candidate met silhouette >= %.2f; selecting maximum silhouette.",
            silhouette_threshold,
        )
        best_row = metrics_df.sort_values(
            by=["silhouette", "bootstrap_stability", "k"],
            ascending=[False, False, True],
        ).iloc[0]
    else:
        # Highest stability, ties -> smaller k
        best_row = qualifying.sort_values(
            by=["bootstrap_stability", "silhouette", "k"],
            ascending=[False, False, True],
        ).iloc[0]

    best_key = str(best_row["model_key"])
    best_algo = str(best_row["algorithm"])
    best_k = int(best_row["k"])
    best_model = fitted_models[best_key]
    best_labels = model_labels[best_key]

    # Compute centroids for best model (both standardized and original scale)
    std_centroids = np.zeros((best_k, X_scaled.shape[1]))
    unstd_centroids = np.zeros((best_k, X_raw.shape[1]))

    for c in range(best_k):
        c_mask = best_labels == c
        if np.any(c_mask):
            std_centroids[c] = X_scaled[c_mask].mean(axis=0)
            unstd_centroids[c] = X_raw[c_mask].mean(axis=0)
        else:
            std_centroids[c] = X_scaled.mean(axis=0)
            unstd_centroids[c] = X_raw.mean(axis=0)

    # Determine cluster names
    cluster_names = {}
    for c in range(best_k):
        cluster_names[c] = profile_cluster(unstd_centroids[c], feature_names)

    logger.info(
        "Clustering selection complete: Algorithm=%s, k=%d, Silhouette=%.4f, Stability=%.4f",
        best_algo,
        best_k,
        best_row["silhouette"],
        best_row["bootstrap_stability"],
    )

    return ClusteringResult(
        best_model=best_model,
        best_algorithm=best_algo,
        best_k=best_k,
        scaler=scaler,
        cluster_names=cluster_names,
        centroids=unstd_centroids,
        standardized_centroids=std_centroids,
        metrics_table=metrics_df,
        best_silhouette=float(best_row["silhouette"]),
        best_stability=float(best_row["bootstrap_stability"]),
    )
