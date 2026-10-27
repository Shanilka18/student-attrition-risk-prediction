"""
Class Imbalance Mitigation Strategies: Weights, SMOTE, and Focal Loss.
Designed strictly for training-time application (inside cross-validation folds).

Component: Academic Intelligence Framework (IT23164680)
"""

from typing import Dict, Tuple, Any, Optional
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("AcademicIntelligence.Imbalance")


def compute_balanced_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Computes inverse-frequency balanced class weights for shallow classifiers (e.g. Logistic Regression, XGBoost).
    Formula: n_samples / (n_classes * count_c)
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weights = {int(c): float(n_samples / (n_classes * count)) for c, count in zip(classes, counts)}
    logger.info(f"Computed class weights: {weights}")
    return weights


def compute_pytorch_pos_weight(y: np.ndarray) -> torch.Tensor:
    """
    Computes pos_weight tensor for torch.nn.BCEWithLogitsLoss.
    Formula: num_negatives / num_positives.
    """
    num_pos = np.sum(y == 1)
    num_neg = np.sum(y == 0)
    pos_weight = float(num_neg / max(num_pos, 1))
    return torch.tensor([pos_weight], dtype=torch.float32)


def apply_smote_tabular(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42,
    k_neighbors: int = 5
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) strictly on tabular training fold data.
    Never applied to validation or test splits.
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(k_neighbors=k_neighbors, random_state=seed)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(f"SMOTE applied via imblearn: {len(X)} -> {len(X_res)} samples")
        return pd.DataFrame(X_res, columns=X.columns), np.array(y_res)
    except ImportError:
        logger.warning("imblearn not available. Falling back to native Scikit-Learn SMOTE implementation.")
        from sklearn.neighbors import NearestNeighbors

        pos_mask = (y == 1)
        X_pos = X[pos_mask].values
        n_pos = len(X_pos)
        n_neg = np.sum(~pos_mask)
        n_synth = n_neg - n_pos

        if n_synth <= 0 or n_pos <= k_neighbors:
            return X.copy(), y.copy()

        nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_pos)
        _, indices = nn.kneighbors(X_pos)

        rng = np.random.default_rng(seed)
        synth_samples = []
        for _ in range(n_synth):
            i = rng.integers(0, n_pos)
            neighbor_idx = rng.choice(indices[i, 1:])
            diff = X_pos[neighbor_idx] - X_pos[i]
            gap = rng.random()
            synth_samples.append(X_pos[i] + gap * diff)

        X_synth = np.array(synth_samples)
        X_res = np.vstack([X.values, X_synth])
        y_res = np.concatenate([y, np.ones(n_synth, dtype=int)])
        logger.info(f"Native SMOTE applied: {len(X)} -> {len(X_res)} samples")
        return pd.DataFrame(X_res, columns=X.columns), y_res


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks with extreme class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float().view_as(logits)

        # p_t
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        # alpha_t
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        # focal weight
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
