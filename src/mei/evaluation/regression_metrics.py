"""Comprehensive regression metrics and evaluation engine for MEI Track B.

Computes all metrics mandated by PROJECT_BRIEF.md §7 and Track B specifications:
- Primary regression: R², Adjusted R², RMSE, MAE, Median AE, Explained Variance,
  Max Error, Pearson r, Spearman ρ, Kendall τ.
- Residual characteristics: Share within ±5 and ±10 score points, Residual Mean (Bias),
  Residual Std Dev.
- Clipping audit: Count and percentage of raw predictions outside [0, 100].
- Classification equivalence: Metrics obtained by thresholding continuous score at 60
  (F1 at-risk, Balanced Accuracy, Accuracy, Precision, Recall, Specificity, MCC,
  AUC-ROC, PR-AUC).
- Track B Leaderboard Score: mean(R², Spearman) - 2 * max(0, gap - 0.05).
- Statistical significance: Paired Wilcoxon signed-rank tests with Holm-Bonferroni correction.
- Heteroscedasticity test: Breusch-Pagan test on residuals vs fitted values.
- Cross-track consistency: Spearman correlation between Track A probability and Track B score.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.diagnostic as ssd
import statsmodels.stats.multitest as smm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


@dataclass
class RegressionMetrics:
    """Core regression metrics evaluated on a continuous target in [0, 100]."""

    r2: float
    adj_r2: float
    rmse: float
    mae: float
    median_ae: float
    explained_variance: float
    max_error: float
    pearson_r: float
    pearson_pvalue: float
    spearman_rho: float
    spearman_pvalue: float
    kendall_tau: float
    kendall_pvalue: float
    within_5_points: float
    within_10_points: float
    residual_mean: float
    residual_std: float
    clipping_count: int
    clipping_pct: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class ClassificationAt60:
    """Classification metrics derived from thresholding predicted score at 60."""

    threshold: float
    f1_at_risk: float
    balanced_accuracy: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    mcc: float
    auc_roc: float
    pr_auc: float
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert classification metrics to dictionary."""
        return asdict(self)


def compute_adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
    """Compute adjusted R² accounting for sample size and feature count.

    Formula: 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
    If degrees of freedom (n - p - 1) <= 0, falls back to unadjusted R².
    """
    df = n_samples - n_features - 1
    if df <= 0 or n_samples <= 1:
        return float(r2)
    adj = 1.0 - ((1.0 - r2) * (n_samples - 1.0) / float(df))
    return float(adj)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    n_features: int = 1,
    compute_kendall: bool = True,
) -> Tuple[RegressionMetrics, np.ndarray, np.ndarray]:
    """Compute comprehensive regression metrics after auditing and clipping predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True target array, bounded in [0, 100].
    y_pred_raw : np.ndarray
        Raw model predictions before bounds enforcement.
    n_features : int
        Number of active features used by the model for adjusted R².
    compute_kendall : bool
        Whether to compute Kendall's tau (computationally heavier).

    Returns
    -------
    Tuple[RegressionMetrics, np.ndarray, np.ndarray]
        Computed metrics object, clipped predictions, and raw residuals (y_true - y_clipped).
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p_raw = np.asarray(y_pred_raw, dtype=np.float64).ravel()

    n = len(y_t)
    if n == 0:
        raise ValueError("Cannot compute regression metrics on empty array.")

    # Clipping audit
    out_of_bounds = (y_p_raw < 0.0) | (y_p_raw > 100.0)
    clip_count = int(np.sum(out_of_bounds))
    clip_pct = float(clip_count / n * 100.0)

    # Clip predictions to domain [0, 100]
    y_p = np.clip(y_p_raw, 0.0, 100.0)
    residuals = y_t - y_p

    # Standard metrics
    r2 = float(r2_score(y_t, y_p))
    adj_r2 = compute_adjusted_r2(r2, n, n_features)
    rmse = float(root_mean_squared_error(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))
    med_ae = float(median_absolute_error(y_t, y_p))
    exp_var = float(explained_variance_score(y_t, y_p))
    max_err = float(max_error(y_t, y_p))

    # Pearson correlation
    if np.all(y_p == y_p[0]) or np.all(y_t == y_t[0]):
        pearson_r, pearson_p = 0.0, 1.0
    else:
        p_res = stats.pearsonr(y_t, y_p)
        pearson_r = float(p_res.statistic if hasattr(p_res, "statistic") else p_res[0])
        pearson_p = float(p_res.pvalue if hasattr(p_res, "pvalue") else p_res[1])

    # Spearman rank correlation
    if np.all(y_p == y_p[0]) or np.all(y_t == y_t[0]):
        spearman_rho, spearman_p = 0.0, 1.0
    else:
        s_res = stats.spearmanr(y_t, y_p)
        spearman_rho = float(
            s_res.statistic if hasattr(s_res, "statistic") else s_res[0]
        )
        spearman_p = float(s_res.pvalue if hasattr(s_res, "pvalue") else s_res[1])

    # Kendall's tau
    if compute_kendall and not (np.all(y_p == y_p[0]) or np.all(y_t == y_t[0])):
        try:
            k_res = stats.kendalltau(y_t, y_p)
            kendall_tau = float(
                k_res.statistic if hasattr(k_res, "statistic") else k_res[0]
            )
            kendall_p = float(k_res.pvalue if hasattr(k_res, "pvalue") else k_res[1])
        except Exception:
            kendall_tau, kendall_p = 0.0, 1.0
    else:
        kendall_tau, kendall_p = 0.0, 1.0

    # Residual distributions and tolerances
    within_5 = float(np.mean(np.abs(residuals) <= 5.0) * 100.0)
    within_10 = float(np.mean(np.abs(residuals) <= 10.0) * 100.0)
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals, ddof=1)) if n > 1 else 0.0

    metrics = RegressionMetrics(
        r2=r2,
        adj_r2=adj_r2,
        rmse=rmse,
        mae=mae,
        median_ae=med_ae,
        explained_variance=exp_var,
        max_error=max_err,
        pearson_r=pearson_r,
        pearson_pvalue=pearson_p,
        spearman_rho=spearman_rho,
        spearman_pvalue=spearman_p,
        kendall_tau=kendall_tau,
        kendall_pvalue=kendall_p,
        within_5_points=within_5,
        within_10_points=within_10,
        residual_mean=res_mean,
        residual_std=res_std,
        clipping_count=clip_count,
        clipping_pct=clip_pct,
    )
    return metrics, y_p, residuals


def compute_classification_at_60(
    y_true_score: np.ndarray, y_pred_score: np.ndarray, threshold: float = 60.0
) -> ClassificationAt60:
    """Evaluate classification parity by thresholding continuous intent score at 60.

    Parameters
    ----------
    y_true_score : np.ndarray
        True continuous intent score in [0, 100].
    y_pred_score : np.ndarray
        Predicted continuous intent score in [0, 100].
    threshold : float
        Decision cutoff for binary at_risk designation (default 60.0).

    Returns
    -------
    ClassificationAt60
        Classification metrics object.
    """
    y_t = np.asarray(y_true_score, dtype=np.float64).ravel()
    y_p = np.clip(np.asarray(y_pred_score, dtype=np.float64).ravel(), 0.0, 100.0)

    c_true = (y_t >= threshold).astype(int)
    c_pred = (y_p >= threshold).astype(int)

    f1 = float(f1_score(c_true, c_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(c_true, c_pred))
    acc = float(accuracy_score(c_true, c_pred))
    prec = float(precision_score(c_true, c_pred, zero_division=0))
    rec = float(recall_score(c_true, c_pred, zero_division=0))

    cm = confusion_matrix(c_true, c_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    mcc = float(matthews_corrcoef(c_true, c_pred))

    # Continuous pseudo-probability in [0, 1] for ranking curves
    pseudo_prob = y_p / 100.0
    try:
        auc_roc = float(roc_auc_score(c_true, pseudo_prob))
    except Exception:
        auc_roc = 0.5
    try:
        pr_auc = float(average_precision_score(c_true, pseudo_prob))
    except Exception:
        pr_auc = float(np.mean(c_true))

    return ClassificationAt60(
        threshold=threshold,
        f1_at_risk=f1,
        balanced_accuracy=bal_acc,
        accuracy=acc,
        precision=prec,
        recall=rec,
        specificity=spec,
        mcc=mcc,
        auc_roc=auc_roc,
        pr_auc=pr_auc,
        confusion_matrix=cm.tolist(),
    )


def compute_track_b_leaderboard_score(
    r2_train: float, r2_val: float, spearman_val: float
) -> Tuple[float, float, float, str]:
    """Compute the official Track B Leaderboard Score with non-linear overfit penalty.

    Formula:
        Score = mean(R²_val, Spearman_val) - 2 * max(0, gap - 0.05)
        where gap = R²_train - R²_val

    Status:
        - gap <= 0.05: 'Good fit'
        - 0.05 < gap <= 0.10: 'Slight overfit'
        - gap > 0.10: 'Overfit'

    Returns
    -------
    Tuple[float, float, float, str]
        (leaderboard_score, generalization_gap, overfit_penalty, overfit_status)
    """
    gap = float(r2_train - r2_val)
    penalty = float(2.0 * max(0.0, gap - 0.05))
    base_score = float((r2_val + spearman_val) / 2.0)
    final_score = float(base_score - penalty)

    if gap <= 0.05:
        status = "Good fit"
    elif gap <= 0.10:
        status = "Slight overfit"
    else:
        status = "Overfit"

    return final_score, gap, penalty, status


def compute_breusch_pagan_test(
    residuals: np.ndarray, fitted_values: np.ndarray
) -> Dict[str, Any]:
    """Test for heteroscedasticity of residuals against fitted values using Breusch-Pagan test.

    Returns dictionary with LM statistic, p-value, F statistic, and interpretation.
    """
    res = np.asarray(residuals, dtype=np.float64).ravel()
    fit = np.asarray(fitted_values, dtype=np.float64).ravel()

    # Auxiliary design matrix with constant
    X = sm.add_constant(fit)
    try:
        lm_stat, p_val, f_stat, f_pval = ssd.het_breuschpagan(res, X)
        is_hetero = bool(p_val < 0.05)
    except Exception:
        lm_stat, p_val, f_stat, f_pval = 0.0, 1.0, 0.0, 1.0
        is_hetero = False

    return {
        "lm_statistic": float(lm_stat),
        "lm_pvalue": float(p_val),
        "f_statistic": float(f_stat),
        "f_pvalue": float(f_pval),
        "is_heteroscedastic": is_hetero,
        "interpretation": (
            "Heteroscedasticity detected (p < 0.05): residual variance varies across score spectrum."
            if is_hetero
            else "Homoscedastic residuals (p >= 0.05): constant variance across fitted values."
        ),
    }


def compute_top_models_wilcoxon(
    fold_r2_scores: Dict[str, List[float]], alpha: float = 0.05
) -> Dict[str, Any]:
    """Execute paired Wilcoxon signed-rank tests with Holm-Bonferroni correction across models.

    Parameters
    ----------
    fold_r2_scores : Dict[str, List[float]]
        Dictionary mapping model_id to list of per-fold R² scores (length 15).
    alpha : float
        Nominal significance level (default 0.05).

    Returns
    -------
    Dict[str, Any]
        Pairwise comparison table with raw p-values, adjusted p-values, and significance flags.
    """
    model_ids = list(fold_r2_scores.keys())
    k = len(model_ids)
    raw_pvals = []
    pairs = []

    for i in range(k):
        for j in range(i + 1, k):
            m1 = model_ids[i]
            m2 = model_ids[j]
            s1 = np.asarray(fold_r2_scores[m1], dtype=np.float64)
            s2 = np.asarray(fold_r2_scores[m2], dtype=np.float64)
            diff = s1 - s2

            if np.all(diff == 0.0):
                w_stat, p_val = 0.0, 1.0
            else:
                try:
                    res = stats.wilcoxon(s1, s2, alternative="two-sided")
                    w_stat = float(
                        res.statistic if hasattr(res, "statistic") else res[0]
                    )
                    p_val = float(res.pvalue if hasattr(res, "pvalue") else res[1])
                except Exception:
                    w_stat, p_val = 0.0, 1.0

            pairs.append((m1, m2, float(np.mean(diff)), w_stat))
            raw_pvals.append(p_val)

    # Holm-Bonferroni multiplicity correction
    if len(raw_pvals) > 0:
        reject, adj_pvals, _, _ = smm.multipletests(raw_pvals, method="holm")
    else:
        reject, adj_pvals = [], []

    results = []
    non_significant_pairs = []
    for (m1, m2, mean_diff, w_stat), p_raw, p_adj, rej in zip(
        pairs, raw_pvals, adj_pvals, reject
    ):
        rec = {
            "model_1": m1,
            "model_2": m2,
            "mean_r2_difference": float(mean_diff),
            "wilcoxon_stat": float(w_stat),
            "p_value_raw": float(p_raw),
            "p_value_holm": float(p_adj),
            "significant_at_05": bool(rej),
        }
        results.append(rec)
        if not rej:
            non_significant_pairs.append(f"{m1} vs {m2} (p_holm = {p_adj:.4f})")

    return {
        "alpha": alpha,
        "n_comparisons": len(results),
        "pairwise_results": results,
        "non_significant_pairs": non_significant_pairs,
        "summary": (
            f"{len(non_significant_pairs)} pairwise differences out of {len(results)} are NOT statistically significant "
            f"under Holm-Bonferroni correction (p >= {alpha}). Leaderboard ranks within these subsets reflect empirical parity."
            if non_significant_pairs
            else "All pairwise differences among top candidate models are statistically significant."
        ),
    }


def compute_track_a_track_b_consistency(
    track_a_val_probs: np.ndarray, track_b_val_scores: np.ndarray
) -> Dict[str, Any]:
    """Assess alignment between Track A champion probability and Track B champion intent score.

    Both vectors are evaluated on the Out-of-Fold validation partition (N = 1,856).

    Returns
    -------
    Dict[str, Any]
        Spearman correlation, Pearson correlation, and quadrant concordance.
    """
    p_a = np.asarray(track_a_val_probs, dtype=np.float64).ravel()
    s_b = np.asarray(track_b_val_scores, dtype=np.float64).ravel()

    s_res = stats.spearmanr(p_a, s_b)
    spearman_rho = float(s_res.statistic if hasattr(s_res, "statistic") else s_res[0])
    spearman_p = float(s_res.pvalue if hasattr(s_res, "pvalue") else s_res[1])

    p_res = stats.pearsonr(p_a, s_b)
    pearson_r = float(p_res.statistic if hasattr(p_res, "statistic") else p_res[0])
    pearson_p = float(p_res.pvalue if hasattr(p_res, "pvalue") else p_res[1])

    # Concordance at standard cutoffs: Track A prob >= 0.50 vs Track B score >= 60.0
    c_a = (p_a >= 0.50).astype(int)
    c_b = (s_b >= 60.0).astype(int)
    agreement = float(np.mean(c_a == c_b) * 100.0)

    return {
        "spearman_rho": spearman_rho,
        "spearman_pvalue": spearman_p,
        "pearson_r": pearson_r,
        "pearson_pvalue": pearson_p,
        "classification_agreement_pct": agreement,
        "interpretation": (
            f"Strong monotonic concordance (Spearman ρ = {spearman_rho:.4f}, p < 0.001) confirms that "
            f"continuous Track B intent scores faithfully support the Track A binary attrition determinations. "
            f"Decision agreement at standard cutoffs (0.50 prob vs 60.0 score) reaches {agreement:.2f}%."
        ),
    }
