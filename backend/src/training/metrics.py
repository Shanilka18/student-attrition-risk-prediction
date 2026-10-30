"""
Academic Risk Validation Metrics.
Computes AUROC, Average Precision (PR-AUC), Brier Calibration Score,
and thresholded metrics (F1, Recall, Precision, Specificity, Accuracy).

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_validation_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes discrimination and calibration metrics on validation partition.
    Strictly observes true labels and predicted continuous probabilities in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Discrimination (ranking) metrics
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))

    # Calibration error
    brier = float(brier_score_loss(y_true, y_prob))

    # Thresholded classification
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0.0

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "threshold": round(float(threshold), 4),
        "f1_score": round(f1, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "accuracy": round(acc, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
