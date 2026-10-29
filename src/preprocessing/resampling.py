"""In-fold class imbalance mitigation strategies for attrition prediction.

Provides:
1. 'none': Standard empirical distribution (no resampling).
2. 'class_weights': Cost-sensitive inverse class frequency weights.
3. 'smote_in_fold': In-fold synthetic minority oversampling (training set only).

Hard Rule: Validation and test sets are NEVER resampled or synthetically augmented.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src.utils.logging_config import setup_logger

logger = setup_logger("resampling")


def get_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights for cost-sensitive loss functions.

    Formula: w_j = N / (n_classes * N_j)
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    logger.info(f"Computed balanced class weights: {class_weight_dict}")
    return class_weight_dict


def apply_smote_in_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_ratio: float = 0.50,
    k_neighbors: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Synthetic Minority Over-sampling Technique (SMOTE) strictly inside training fold.

    Parameters
    ----------
    X_train : np.ndarray
        Transformed numerical feature matrix of the training fold.
    y_train : np.ndarray
        Binary target vector of the training fold.
    target_ratio : float
        Ratio of minority to majority samples post-resampling (default 0.50).
    k_neighbors : int
        Number of nearest neighbors for interpolation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_resampled, y_resampled)
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_train, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]

    n_maj = int(np.max(counts))
    n_min = int(np.min(counts))

    target_n_min = int(n_maj * target_ratio)
    n_synthetic = max(0, target_n_min - n_min)

    if n_synthetic == 0:
        logger.info("Minority class already meets or exceeds target ratio. Skipping SMOTE.")
        return X_train, y_train

    logger.info(
        f"Applying in-fold SMOTE: generating {n_synthetic} synthetic samples for minority class {minority_class} "
        f"(original: {n_min}, target: {target_n_min})."
    )

    X_min = X_train[y_train == minority_class]

    # Simple, deterministic k-NN interpolation
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(X_min)))
    nn.fit(X_min)
    indices = nn.kneighbors(X_min, return_distance=False)

    synthetic_samples = np.zeros((n_synthetic, X_train.shape[1]), dtype=X_train.dtype)
    for i in range(n_synthetic):
        base_idx = rng.integers(0, len(X_min))
        # Pick one random neighbor among the k neighbors (excluding self at 0)
        neighbor_pool = indices[base_idx, 1:]
        chosen_neighbor_idx = rng.choice(neighbor_pool) if len(neighbor_pool) > 0 else base_idx
        diff = X_min[chosen_neighbor_idx] - X_min[base_idx]
        gap = rng.uniform(0.0, 1.0)
        synthetic_samples[i] = X_min[base_idx] + gap * diff

    X_resampled = np.vstack([X_train, synthetic_samples])
    y_resampled = np.concatenate([y_train, np.full(n_synthetic, minority_class, dtype=y_train.dtype)])

    # Shuffle resampled dataset
    shuffle_idx = rng.permutation(len(y_resampled))
    return X_resampled[shuffle_idx], y_resampled[shuffle_idx]
