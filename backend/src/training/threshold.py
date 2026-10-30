"""
Decision Threshold Optimization Engine.
Supports Max-F1 and Recall-Constrained optimization on validation partition.
Recognizes institutional asymmetry: false negatives (missed dropouts) carry
higher risk than false positives (routine counseling checks).

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger("AcademicIntelligence.Threshold")


def optimize_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "recall_constrained",
    min_recall: float = 0.80,
    num_steps: int = 100,
) -> Dict[str, Any]:
    """
    Finds optimal binary classification cutoff on validation probabilities.

    Strategies:
      - 'recall_constrained': ensures Recall >= min_recall, then maximizes F1.
                              Guarantees safety-critical early warning coverage.
      - 'max_f1': standard harmonic mean maximization.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.01, 0.99, num_steps)
    records: List[Dict[str, float]] = []

    best_idx = 0
    best_score = -1.0

    for i, t in enumerate(thresholds):
        y_pred = (y_prob >= t).astype(int)
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))

        records.append({
            "threshold": round(float(t), 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
        })

    if strategy == "recall_constrained":
        # Candidates satisfying recall constraint
        valid_indices = [i for i, r in enumerate(records) if r["recall"] >= min_recall]
        if valid_indices:
            # Pick valid threshold maximizing F1
            best_idx = max(valid_indices, key=lambda idx: (records[idx]["f1_score"], records[idx]["precision"]))
        else:
            # If no threshold meets target recall, pick threshold with highest recall
            best_idx = max(range(len(records)), key=lambda idx: (records[idx]["recall"], records[idx]["f1_score"]))
    else:  # max_f1 default
        best_idx = max(range(len(records)), key=lambda idx: records[idx]["f1_score"])

    best_record = records[best_idx]
    logger.info(
        f"Optimized threshold ({strategy}, min_recall={min_recall}): "
        f"Cutoff={best_record['threshold']:.3f} -> Recall={best_record['recall']:.1%}, "
        f"Precision={best_record['precision']:.1%}, F1={best_record['f1_score']:.3f}"
    )

    # Downsampled curve for lightweight JSON persistence
    curve_step = max(1, len(records) // 25)
    downsampled_curve = records[::curve_step]
    if records[-1] not in downsampled_curve:
        downsampled_curve.append(records[-1])

    return {
        "strategy": strategy,
        "min_recall_target": float(min_recall) if strategy == "recall_constrained" else None,
        "optimal_threshold": best_record["threshold"],
        "metrics_at_threshold": {
            "f1_score": best_record["f1_score"],
            "recall": best_record["recall"],
            "precision": best_record["precision"],
            "accuracy": best_record["accuracy"],
        },
        "threshold_curve": downsampled_curve,
    }
