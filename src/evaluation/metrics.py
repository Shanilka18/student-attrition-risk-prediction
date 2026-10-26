"""Metric computation suite for ERI model zoo.

Computes full metric suites for Track C (classification) and Track R (regression),
with 95% bootstrap confidence intervals on the validation set.

No metric is ever invented — every number comes from real predictions.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error) — 15-bin
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error computed with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() / n * abs(avg_conf - avg_acc)
    return float(ece)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (point_estimate, lower_ci, upper_ci) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = metric_fn(y_true[idx], y_score[idx])
            bootstrap_scores.append(score)
        except Exception:
            pass
    if not bootstrap_scores:
        point = metric_fn(y_true, y_score)
        return float(point), float("nan"), float("nan")

    arr = np.array(bootstrap_scores)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.nanpercentile(arr, 100 * alpha))
    upper = float(np.nanpercentile(arr, 100 * (1 - alpha)))
    point = float(metric_fn(y_true, y_score))
    return point, lower, upper


# ---------------------------------------------------------------------------
# Classification metrics (Track C)
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full classification metric suite.

    Parameters
    ----------
    y_true : 1-D array of {0, 1}
    y_pred : 1-D array of predicted labels
    y_prob : 1-D array of positive-class probabilities
    split : str — 'train', 'val', or 'test'
    n_bootstrap : int — number of bootstrap samples (for CI on val)
    seed : int

    Returns
    -------
    dict with all metric keys; CI keys only populated when split == 'val'.
    """
    n = len(y_true)
    pos_count = int(np.sum(y_true == 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))
        kappa = float(cohen_kappa_score(y_true, y_pred))
        logloss = float(log_loss(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))
        ece = compute_ece(y_true, y_prob)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
        specificity = float(tn / max(tn + fp, 1))

    metrics: Dict[str, Any] = {
        "split": split,
        "samples": n,
        "positive_count": pos_count,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "log_loss": logloss,
        "brier_score": brier,
        "ece": ece,
    }

    # Bootstrap CIs — only on val split (computationally heavy)
    if split == "val" and n >= 50:
        for metric_name, fn_ci in [
            ("roc_auc", lambda yt, ys: roc_auc_score(yt, ys)),
            ("pr_auc", lambda yt, ys: average_precision_score(yt, ys)),
            ("brier_score", lambda yt, ys: brier_score_loss(yt, ys)),
            ("balanced_accuracy", lambda yt, ys: balanced_accuracy_score(
                yt, (ys >= 0.5).astype(int))),
            ("f1", lambda yt, ys: f1_score(yt, (ys >= 0.5).astype(int), zero_division=0)),
        ]:
            try:
                pt, lo, hi = bootstrap_ci(y_true, y_prob, fn_ci,
                                          n_bootstrap=n_bootstrap, seed=seed)
                metrics[f"{metric_name}_ci_lo"] = lo
                metrics[f"{metric_name}_ci_hi"] = hi
            except Exception:
                metrics[f"{metric_name}_ci_lo"] = float("nan")
                metrics[f"{metric_name}_ci_hi"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Regression metrics (Track R)
# ---------------------------------------------------------------------------

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str,
    n_features: int = 1,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full regression metric suite.

    Parameters
    ----------
    y_true : 1-D float array
    y_pred : 1-D float array
    split : str
    n_features : int — used for adjusted R²
    n_bootstrap : int
    seed : int
    """
    n = len(y_true)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        r2 = float(r2_score(y_true, y_pred))
        adj_r2 = 1.0 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        ev = float(explained_variance_score(y_true, y_pred))
        spear = float(spearmanr(y_true, y_pred).statistic)

        # MAPE / SMAPE — handle zero targets
        nonzero = y_true != 0
        if nonzero.sum() > 0:
            mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])))
        else:
            mape = float("nan")
        smape = float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))

    metrics: Dict[str, Any] = {
        "split": split,
        "samples": n,
        "r2": r2,
        "adj_r2": adj_r2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "explained_variance": ev,
        "spearman_rho": spear,
    }

    if split == "val" and n >= 50:
        for metric_name, fn_ci in [
            ("r2", lambda yt, yp: r2_score(yt, yp)),
            ("mae", lambda yt, yp: mean_absolute_error(yt, yp)),
            ("rmse", lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp)))),
        ]:
            try:
                _, lo, hi = bootstrap_ci(y_true, y_pred, fn_ci,
                                         n_bootstrap=n_bootstrap, seed=seed)
                metrics[f"{metric_name}_ci_lo"] = lo
                metrics[f"{metric_name}_ci_hi"] = hi
            except Exception:
                metrics[f"{metric_name}_ci_lo"] = float("nan")
                metrics[f"{metric_name}_ci_hi"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Curve data (for frontend rendering)
# ---------------------------------------------------------------------------

def compute_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, List]:
    """Return ROC curve as dict of lists for JSON serialization."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}


def compute_pr_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, List]:
    """Return Precision-Recall curve as dict of lists."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    return {"precision": prec.tolist(), "recall": rec.tolist(),
            "thresholds": thresholds.tolist()}


def compute_calibration_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, List]:
    """Return calibration (reliability diagram) curve data."""
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins,
                                            strategy="uniform")
    return {"fraction_of_positives": frac_pos.tolist(), "mean_predicted": mean_pred.tolist()}
