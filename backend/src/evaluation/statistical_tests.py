"""
Statistical Significance Testing & Grouped Student Bootstrap Engine.
Computes:
  1. 95% Bootstrap Confidence Intervals grouped by student ID
  2. Paired Bootstrap Hypothesis Tests on AUROC / PR-AUC differences (two-sided p-value)
  3. McNemar's Test on paired binary predictions with continuity correction

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, fbeta_score, roc_auc_score

logger = logging.getLogger("AcademicIntelligence.StatsTests")


def compute_grouped_bootstrap_cis(
    df_samples: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes 95% bootstrap confidence intervals grouped by student_id to
    properly account for intra-cluster correlation across multi-semester observations.
    """
    rng = np.random.default_rng(seed)
    student_ids = df_samples["student_id"].values
    unique_sids = np.unique(student_ids)
    n_students = len(unique_sids)

    # Pre-index student rows for high performance
    sid_to_indices: Dict[str, np.ndarray] = {}
    for idx, sid in enumerate(student_ids):
        if sid not in sid_to_indices:
            sid_to_indices[sid] = []
        sid_to_indices[sid].append(idx)
    for sid in sid_to_indices:
        sid_to_indices[sid] = np.array(sid_to_indices[sid], dtype=int)

    boot_roc_auc: List[float] = []
    boot_pr_auc: List[float] = []
    boot_f1: List[float] = []
    boot_f2: List[float] = []
    boot_brier: List[float] = []

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    for _ in range(n_resamples):
        sampled_sids = rng.choice(unique_sids, size=n_students, replace=True)
        # Concatenate rows of sampled students
        row_indices = np.concatenate([sid_to_indices[s] for s in sampled_sids])

        y_t_b = y_true[row_indices]
        y_p_b = y_prob[row_indices]
        y_pred_b = (y_p_b >= threshold).astype(int)

        # Skip degenerate samples if any
        if len(np.unique(y_t_b)) < 2:
            continue

        try:
            boot_roc_auc.append(float(roc_auc_score(y_t_b, y_p_b)))
            boot_pr_auc.append(float(average_precision_score(y_t_b, y_p_b)))
            boot_f1.append(float(f1_score(y_t_b, y_pred_b, zero_division=0)))
            boot_f2.append(float(fbeta_score(y_t_b, y_pred_b, beta=2.0, zero_division=0)))
            boot_brier.append(float(brier_score_loss(y_t_b, y_p_b)))
        except Exception:
            continue

    def _ci(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "ci_lower": round(float(np.percentile(arr, lower_pct)), 4),
            "ci_upper": round(float(np.percentile(arr, upper_pct)), 4),
        }

    return {
        "roc_auc": _ci(boot_roc_auc),
        "pr_auc": _ci(boot_pr_auc),
        "f1_score": _ci(boot_f1),
        "f2_score": _ci(boot_f2),
        "brier_score": _ci(boot_brier),
    }


def paired_bootstrap_test(
    df_samples: pd.DataFrame,
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Conducts paired student-grouped bootstrap significance test on metric difference:
    Delta = Metric(Model A) - Metric(Model B).
    Returns empirical two-sided p-value and 95% CI on the difference.
    """
    rng = np.random.default_rng(seed)
    student_ids = df_samples["student_id"].values
    unique_sids = np.unique(student_ids)
    n_students = len(unique_sids)

    sid_to_indices: Dict[str, List[int]] = {}
    for idx, sid in enumerate(student_ids):
        if sid not in sid_to_indices:
            sid_to_indices[sid] = []
        sid_to_indices[sid].append(idx)
    for sid in sid_to_indices:
        sid_to_indices[sid] = np.array(sid_to_indices[sid], dtype=int)

    diffs: List[float] = []
    observed_diff = float(metric_fn(y_true, y_prob_a) - metric_fn(y_true, y_prob_b))

    for _ in range(n_resamples):
        sampled_sids = rng.choice(unique_sids, size=n_students, replace=True)
        row_indices = np.concatenate([sid_to_indices[s] for s in sampled_sids])

        y_t_b = y_true[row_indices]
        if len(np.unique(y_t_b)) < 2:
            continue

        y_pa_b = y_prob_a[row_indices]
        y_pb_b = y_prob_b[row_indices]

        try:
            val_a = float(metric_fn(y_t_b, y_pa_b))
            val_b = float(metric_fn(y_t_b, y_pb_b))
            diffs.append(val_a - val_b)
        except Exception:
            continue

    if not diffs:
        return {
            "observed_difference": round(observed_diff, 4),
            "ci_lower": round(observed_diff, 4),
            "ci_upper": round(observed_diff, 4),
            "p_value": 1.0,
            "statistically_significant": False,
        }

    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))

    # Two-sided empirical p-value
    prop_leq_zero = float(np.mean(np.array(diffs) <= 0.0))
    prop_geq_zero = float(np.mean(np.array(diffs) >= 0.0))
    p_val = float(min(1.0, 2.0 * min(prop_leq_zero, prop_geq_zero)))

    return {
        "observed_difference": round(observed_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_val, 5),
        "statistically_significant": bool(p_val < 0.05),
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, Any]:
    """
    Performs McNemar's paired test of marginal homogeneity on binary classifications
    with Edwards continuity correction: chi2 = (|b - c| - 1)^2 / (b + c).
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b = Model A correct, Model B incorrect
    b = int(np.sum(correct_a & ~correct_b))
    # c = Model A incorrect, Model B correct
    c = int(np.sum(~correct_a & correct_b))

    total_discordant = b + c
    if total_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "discordant_a_correct": b,
            "discordant_b_correct": c,
            "statistically_significant": False,
        }

    # Continuity correction
    chi2_stat = float((abs(b - c) - 1.0) ** 2 / total_discordant)
    p_val = float(stats.chi2.sf(chi2_stat, df=1))

    return {
        "statistic": round(chi2_stat, 4),
        "p_value": round(p_val, 5),
        "discordant_a_correct": b,
        "discordant_b_correct": c,
        "statistically_significant": bool(p_val < 0.05),
    }
