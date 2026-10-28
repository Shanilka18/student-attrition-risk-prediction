"""Comprehensive classification metrics and evaluation engine for MEI.

Computes all metrics mandated by PROJECT_BRIEF.md §7:
- Threshold-dependent: Accuracy, Balanced Accuracy, Precision, Recall, Specificity,
  F1 (at-risk), Macro-F1, Matthews Correlation Coefficient (MCC), Cohen's Kappa,
  Confusion Matrices (raw and normalized).
- Threshold-independent & probabilistic: AUC-ROC, PR-AUC, Log-Loss, Brier Score,
  Expected Calibration Error (ECE), ROC curve, Precision-Recall curve, Calibration curve.
- Generalization gap, overfit penalties, and Leaderboard Score.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ThresholdMetrics:
    """Classification metrics evaluated at a fixed decision threshold."""

    threshold: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    macro_f1: float
    mcc: float
    cohen_kappa: float
    confusion_matrix_raw: List[List[int]]
    confusion_matrix_norm: List[List[float]]


@dataclass
class ProbabilisticMetrics:
    """Threshold-independent and probabilistic calibration metrics."""

    auc_roc: float
    pr_auc: float
    log_loss: float
    brier_score: float
    ece: float


@dataclass
class CurveData:
    """Diagnostic curve coordinates for visualization and artifact serialization."""

    roc_fpr: List[float]
    roc_tpr: List[float]
    roc_thresholds: List[float]
    pr_precision: List[float]
    pr_recall: List[float]
    pr_thresholds: List[float]
    cal_prob_true: List[float]
    cal_prob_pred: List[float]


@dataclass
class EvaluationResult:
    """Full evaluation packet for a single model evaluation partition."""

    metrics_t05: ThresholdMetrics
    metrics_optimal: ThresholdMetrics
    probabilistic: ProbabilisticMetrics
    curves: CurveData
    optimal_threshold: float
    n_samples: int
    prevalence: float


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins.

    Formula:
        ECE = sum_{m=1}^M ( |B_m| / N ) * | acc(B_m) - conf(B_m) |
    where acc(B_m) is the empirical accuracy (fraction of positives) in bin m,
    and conf(B_m) is the average predicted confidence in bin m.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    if n_total == 0:
        return 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            in_bin = (y_prob >= lower) & (y_prob < upper)

        n_bin = int(np.sum(in_bin))
        if n_bin > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_prob[in_bin]))
            ece += (n_bin / n_total) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute specificity (true negative rate = TN / (TN + FP))."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 0.0


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    """Compute all discrete threshold-dependent metrics at a specific probability cutoff."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    spec = compute_specificity(y_true, y_pred)
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    # Confusion matrix: rows = true, cols = pred -> [[TN, FP], [FN, TP]]
    cm_raw = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")

    return ThresholdMetrics(
        threshold=float(threshold),
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision=prec,
        recall=rec,
        specificity=spec,
        f1=f1,
        macro_f1=macro_f1,
        mcc=mcc,
        cohen_kappa=kappa,
        confusion_matrix_raw=cm_raw.tolist(),
        confusion_matrix_norm=cm_norm.tolist(),
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.01,
) -> float:
    """Find the probability threshold that maximizes F1-score for the positive (at-risk) class."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    thresholds = np.arange(min_threshold, max_threshold + (step / 2.0), step)
    best_threshold = 0.5
    best_f1 = -1.0

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)

    return float(best_threshold)


def _subsample_curve(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None,
    max_points: int = 100,
) -> Tuple[List[float], List[float], List[float]]:
    """Subsample diagnostic curves to a maximum number of points for JSON serialization."""
    n = len(x)
    if n <= max_points:
        z_list = [float(val) for val in z] if z is not None else []
        return [float(val) for val in x], [float(val) for val in y], z_list

    idx = np.round(np.linspace(0, n - 1, max_points)).astype(int)
    x_sub = [float(x[i]) for i in idx]
    y_sub = [float(y[i]) for i in idx]
    z_sub = [float(z[min(i, len(z) - 1)]) for i in idx] if z is not None else []
    return x_sub, y_sub, z_sub


def compute_probabilistic_metrics_and_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_cal_bins: int = 10,
    max_curve_points: int = 100,
) -> Tuple[ProbabilisticMetrics, CurveData]:
    """Compute threshold-independent metrics (AUC-ROC, PR-AUC, Log-Loss, Brier, ECE) and curve coordinates."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    # Probabilistic metrics
    try:
        auc_roc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc_roc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    try:
        ll = float(log_loss(y_true, y_prob_clipped))
    except Exception:
        ll = float("inf")

    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=n_cal_bins)

    prob_metrics = ProbabilisticMetrics(
        auc_roc=auc_roc,
        pr_auc=pr_auc,
        log_loss=ll,
        brier_score=brier,
        ece=ece,
    )

    # ROC curve
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    sub_fpr, sub_tpr, sub_roc_thresh = _subsample_curve(
        fpr, tpr, roc_thresh, max_points=max_curve_points
    )

    # PR curve
    precision_pts, recall_pts, pr_thresh = precision_recall_curve(y_true, y_prob)
    sub_rec, sub_prec, sub_pr_thresh = _subsample_curve(
        recall_pts, precision_pts, pr_thresh, max_points=max_curve_points
    )

    # Calibration reliability curve points
    bin_edges = np.linspace(0.0, 1.0, n_cal_bins + 1)
    cal_true: List[float] = []
    cal_pred: List[float] = []
    for i in range(n_cal_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        mask = (y_prob >= low) & (
            y_prob <= high if i == n_cal_bins - 1 else y_prob < high
        )
        if np.any(mask):
            cal_true.append(float(np.mean(y_true[mask])))
            cal_pred.append(float(np.mean(y_prob[mask])))

    curve_data = CurveData(
        roc_fpr=sub_fpr,
        roc_tpr=sub_tpr,
        roc_thresholds=sub_roc_thresh,
        pr_precision=sub_prec,
        pr_recall=sub_rec,
        pr_thresholds=sub_pr_thresh,
        cal_prob_true=cal_true,
        cal_prob_pred=cal_pred,
    )

    return prob_metrics, curve_data


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal_threshold: Optional[float] = None,
) -> EvaluationResult:
    """Perform a complete metric evaluation on binary predictions and predicted probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if optimal_threshold is None:
        opt_thresh = find_optimal_threshold(y_true, y_prob)
    else:
        opt_thresh = float(optimal_threshold)

    metrics_05 = compute_threshold_metrics(y_true, y_prob, threshold=0.5)
    metrics_opt = compute_threshold_metrics(y_true, y_prob, threshold=opt_thresh)
    prob_metrics, curves = compute_probabilistic_metrics_and_curves(y_true, y_prob)

    prevalence = float(np.mean(y_true)) if len(y_true) > 0 else 0.0

    return EvaluationResult(
        metrics_t05=metrics_05,
        metrics_optimal=metrics_opt,
        probabilistic=prob_metrics,
        curves=curves,
        optimal_threshold=opt_thresh,
        n_samples=len(y_true),
        prevalence=prevalence,
    )


def compute_leaderboard_score(
    val_f1: float,
    val_auc_roc: float,
    val_balanced_acc: float,
    val_mcc: float,
    train_f1: float,
) -> Tuple[float, float, float, str]:
    """Compute the official Track A Leaderboard Score, Generalization Gap, and Overfit Status.

    Brief §7 Formula:
        gap = train_f1 - val_f1
        overfit_penalty = 2.0 * max(0.0, gap - 0.05)
        raw_composite = mean(val_f1, val_auc_roc, val_balanced_acc, (val_mcc + 1.0) / 2.0)
        leaderboard_score = raw_composite - overfit_penalty

    Overfit Status:
        gap <= 0.05: 'Good fit'
        0.05 < gap <= 0.10: 'Slight overfit'
        gap > 0.10: 'Overfit'
    """
    gap = float(train_f1 - val_f1)
    overfit_penalty = float(2.0 * max(0.0, gap - 0.05))

    mcc_rescaled = (val_mcc + 1.0) / 2.0
    raw_composite = float(
        np.mean([val_f1, val_auc_roc, val_balanced_acc, mcc_rescaled])
    )
    score = float(raw_composite - overfit_penalty)

    if gap <= 0.05:
        status = "Good fit"
    elif gap <= 0.10:
        status = "Slight overfit"
    else:
        status = "Overfit"

    return score, gap, overfit_penalty, status
