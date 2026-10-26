"""Comprehensive evaluation metrics for PRI regression and classification tracks.

Regression metrics: R², Adjusted R², RMSE, MAE, MedianAE, MAPE-safe, SMAPE,
    Explained Variance, Spearman, Pearson, within-±10-point accuracy.
Classification metrics: Accuracy, Balanced Accuracy, Precision/Recall/F1
    (macro + weighted + per-class), ROC-AUC OvR, PR-AUC, MCC, Cohen's kappa
    (quadratic), Log Loss, Brier Score, ECE, Confusion Matrix, High-risk recall.
Operational metrics: train time, inference latency, model size, overfit gap.
"""

from typing import Dict, Any, List, Optional, Tuple
import hashlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
)
from sklearn.preprocessing import label_binarize

from src.utils.logger import get_logger

logger = get_logger("evaluation.metrics")

# Canonical label ordering for ordinal kappa
RISK_LABELS: List[str] = ["Low", "Moderate", "High"]
RISK_LABEL_TO_INT: Dict[str, int] = {"Low": 0, "Moderate": 1, "High": 2}


# ---------------------------------------------------------------------------
# Regression Metrics
# ---------------------------------------------------------------------------

def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (safe for zero denominator)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-8
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def _mape_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE that skips zero-denominator samples to avoid division-by-zero."""
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _within_n_points(
    y_true: np.ndarray, y_pred: np.ndarray, n: float = 10.0
) -> float:
    """Fraction of predictions within ±n of the true value."""
    return float(np.mean(np.abs(y_true - y_pred) <= n))


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int = 1,
    within_n: float = 10.0,
) -> Dict[str, float]:
    """Compute full regression metric suite.

    Args:
        y_true: Ground-truth PRI values (0-100).
        y_pred: Predicted PRI values.
        n_features: Number of features used, for Adjusted R² calculation.
        within_n: Threshold for 'within ±n points' accuracy.

    Returns:
        Dict mapping metric name to float value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)

    r2 = float(r2_score(y_true, y_pred))
    # Adjusted R²: penalises for number of predictors
    if n > n_features + 1:
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - n_features - 1))
    else:
        adj_r2 = float("nan")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    median_ae = float(median_absolute_error(y_true, y_pred))
    ev = float(explained_variance_score(y_true, y_pred))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mape = _mape_safe(y_true, y_pred)
    smape = _smape(y_true, y_pred)

    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    within_acc = _within_n_points(y_true, y_pred, within_n)

    return {
        "r2": round(r2, 6),
        "adj_r2": round(adj_r2, 6) if not np.isnan(adj_r2) else None,
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "median_ae": round(median_ae, 6),
        "mape_safe": round(mape, 4) if not np.isnan(mape) else None,
        "smape": round(smape, 4),
        "explained_variance": round(ev, 6),
        "pearson_r": round(float(pearson_r), 6),
        "pearson_p": round(float(pearson_p), 6),
        "spearman_r": round(float(spearman_r), 6),
        "spearman_p": round(float(spearman_p), 6),
        f"within_pm{int(within_n)}_acc": round(within_acc, 6),
        # primary / tiebreak surface for leaderboard
        "primary": round(r2, 6),
        "tiebreak": round(rmse, 6),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) — reliability-weighted confidence gap.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of confidence bins.

    Returns:
        ECE value in [0, 1].
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = np.mean(y_true[mask] == 1) if hasattr(y_true[mask], "mean") else float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def compute_multiclass_ece(
    y_true_int: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str],
    n_bins: int = 10,
) -> float:
    """ECE for multi-class classification (macro average over OvR ECEs)."""
    n_classes = len(labels)
    eces = []
    for c in range(n_classes):
        y_bin = (y_true_int == c).astype(int)
        prob_c = y_prob[:, c]
        eces.append(compute_ece(y_bin, prob_c, n_bins))
    return float(np.mean(eces))


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute full classification metric suite for ordinal risk classification.

    Args:
        y_true: String or integer ground-truth labels.
        y_pred: String or integer predicted labels.
        y_prob: Predicted class probabilities, shape (N, C).
            Column order must match ``labels``.
        labels: Ordered list of class labels e.g. ['Low', 'Moderate', 'High'].

    Returns:
        Dict mapping metric name to value (float, list, or nested dict).
    """
    if labels is None:
        labels = RISK_LABELS

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Convert string labels to ordinal integers for kappa
    if y_true.dtype.kind in ("U", "O"):
        y_true_int = np.array([RISK_LABEL_TO_INT.get(str(v), -1) for v in y_true])
        y_pred_int = np.array([RISK_LABEL_TO_INT.get(str(v), -1) for v in y_pred])
    else:
        y_true_int = y_true.astype(int)
        y_pred_int = y_pred.astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))

    # Per-class metrics
    per_class_f1 = dict(zip(labels, f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()))
    per_class_prec = dict(zip(labels, precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()))
    per_class_rec = dict(zip(labels, recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()))

    # HIGH-RISK recall is a key safety metric
    high_risk_recall = per_class_rec.get("High", float("nan"))

    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Quadratic weighted kappa (appropriate for ordinal 3-class prediction)
    try:
        qwk = float(cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic"))
    except Exception:
        qwk = float("nan")

    # Confusion matrix (Low / Moderate / High)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_precision": round(macro_prec, 6),
        "weighted_precision": round(weighted_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "weighted_recall": round(weighted_rec, 6),
        "per_class_f1": {k: round(v, 6) for k, v in per_class_f1.items()},
        "per_class_precision": {k: round(v, 6) for k, v in per_class_prec.items()},
        "per_class_recall": {k: round(v, 6) for k, v in per_class_rec.items()},
        "high_risk_recall": round(high_risk_recall, 6) if not np.isnan(high_risk_recall) else None,
        "mcc": round(mcc, 6),
        "quadratic_weighted_kappa": round(qwk, 6) if not np.isnan(qwk) else None,
        "confusion_matrix": cm,
        "confusion_matrix_labels": labels,
        # primary / tiebreak surface for leaderboard
        "primary": round(macro_f1, 6),
        "tiebreak": round(mcc, 6),
    }

    # Probability-dependent metrics
    if y_prob is not None and len(y_prob.shape) == 2:
        try:
            y_true_bin = label_binarize(y_true, classes=labels)
            roc_auc = float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro"))
            metrics["roc_auc_ovr_macro"] = round(roc_auc, 6)
        except Exception as exc:
            logger.warning("ROC-AUC failed: %s", exc)
            metrics["roc_auc_ovr_macro"] = None

        try:
            pr_aucs = []
            y_true_bin = label_binarize(y_true, classes=labels)
            for c in range(len(labels)):
                ap = average_precision_score(y_true_bin[:, c], y_prob[:, c])
                pr_aucs.append(float(ap))
            metrics["pr_auc_macro"] = round(float(np.mean(pr_aucs)), 6)
            metrics["pr_auc_per_class"] = dict(zip(labels, [round(v, 6) for v in pr_aucs]))
        except Exception as exc:
            logger.warning("PR-AUC failed: %s", exc)
            metrics["pr_auc_macro"] = None

        try:
            # Log loss and Brier score (macro for multi-class)
            ll = float(log_loss(y_true, y_prob, labels=labels))
            metrics["log_loss"] = round(ll, 6)

            y_true_bin = label_binarize(y_true, classes=labels)
            brier = float(np.mean([
                brier_score_loss(y_true_bin[:, c], y_prob[:, c])
                for c in range(len(labels))
            ]))
            metrics["brier_score"] = round(brier, 6)
        except Exception as exc:
            logger.warning("Log loss / Brier failed: %s", exc)

        try:
            ece = compute_multiclass_ece(y_true_int, y_prob, labels)
            metrics["ece"] = round(ece, 6)
        except Exception as exc:
            logger.warning("ECE computation failed: %s", exc)
            metrics["ece"] = None

    return metrics


# ---------------------------------------------------------------------------
# Statistical Comparison
# ---------------------------------------------------------------------------

def corrected_resampled_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_train: int,
    n_test: int,
) -> Tuple[float, float]:
    """Nadeau-Bengio corrected resampled t-test for CV fold comparisons.

    Corrects for the data reuse dependency in k-fold cross-validation.
    See: Nadeau & Bengio (2003), JMLR.

    Args:
        scores_a: Metric values per CV fold for model A.
        scores_b: Metric values per CV fold for model B.
        n_train: Training set size used per fold.
        n_test: Validation set size used per fold.

    Returns:
        (t_statistic, p_value)
    """
    diff = scores_a - scores_b
    n = len(diff)
    mean_diff = np.mean(diff)
    var_diff = np.var(diff, ddof=1)
    # correction factor for data overlap
    correction = (1.0 / n + n_test / n_train)
    var_corrected = correction * var_diff
    if var_corrected <= 0:
        return float("nan"), 1.0
    t_stat = mean_diff / np.sqrt(var_corrected)
    p_val = float(2.0 * stats.t.sf(np.abs(t_stat), df=n - 1))
    return float(t_stat), p_val


def wilcoxon_comparison(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """Wilcoxon signed-rank test for paired model comparison across CV folds.

    Args:
        scores_a: Metric values per CV fold for model A.
        scores_b: Metric values per CV fold for model B.

    Returns:
        (statistic, p_value)
    """
    try:
        stat, p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return float("nan"), 1.0
