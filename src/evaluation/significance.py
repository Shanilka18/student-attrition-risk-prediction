"""Statistical significance tests between candidate models across CV folds.

Implements:
1. Nadeau & Bengio corrected resampled paired t-test for cross-validation
2. Wilcoxon signed-rank test
3. Parsimony / Simplicity decision rule comparing top candidate against Logistic Regression
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir

logger = setup_logger("significance")


def corrected_resampled_ttest(
    diffs: np.ndarray,
    n_train: int,
    n_test: int,
) -> Tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated/K-fold CV.

    Corrects for the violation of independence among CV error differences:
        variance_corrected = (1/J + n_test/n_train) * sample_variance
    """
    J = len(diffs)
    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1)) if J > 1 else 0.0

    correction = (1.0 / J) + (n_test / n_train)
    corrected_var = var_diff * correction
    se = float(np.sqrt(corrected_var))

    if se < 1e-12:
        return 0.0, 1.0

    t_stat = mean_diff / se
    df = J - 1
    # Two-tailed p-value
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))
    return t_stat, p_value


def compare_models_cv(
    scores_a: List[float],
    scores_b: List[float],
    n_train: int = 1428,
    n_test: int = 286,
) -> Dict[str, Any]:
    """Execute paired hypothesis tests between Model A and Model B."""
    arr_a = np.array(scores_a, dtype=float)
    arr_b = np.array(scores_b, dtype=float)
    diffs = arr_a - arr_b

    t_stat, p_val_ttest = corrected_resampled_ttest(diffs, n_train=n_train, n_test=n_test)

    # Non-parametric Wilcoxon test
    if np.all(diffs == 0):
        p_val_wilcoxon = 1.0
        w_stat = 0.0
    else:
        try:
            w_res = stats.wilcoxon(diffs, zero_method="wilcox")
            w_stat, p_val_wilcoxon = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, p_val_wilcoxon = 0.0, 1.0

    mean_diff = float(np.mean(diffs))
    significant = bool(p_val_ttest < 0.05)

    return {
        "mean_diff": round(mean_diff, 4),
        "corrected_t_stat": round(t_stat, 4),
        "corrected_p_value": round(p_val_ttest, 4),
        "wilcoxon_stat": round(w_stat, 4),
        "wilcoxon_p_value": round(p_val_wilcoxon, 4),
        "is_statistically_significant": significant,
    }


def evaluate_parsimony_rule(
    best_model_name: str,
    best_model_scores: List[float],
    lr_scores: List[float],
    parsimony_margin: float = 0.015,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Check if the top candidate is significantly better than Logistic Regression.

    If difference is not statistically significant (p >= alpha) OR the performance
    delta is within the practical parsimony margin (< 0.015), the selection rule
    recommends selecting Logistic Regression for interpretability and safety.
    """
    if best_model_name.startswith("logistic"):
        return {
            "selected_model": best_model_name,
            "rule_applied": "Winner is already a linear/interpretable model",
            "fallback_to_logistic": False,
            "p_value": 1.0,
            "mean_delta": 0.0,
        }

    comparison = compare_models_cv(best_model_scores, lr_scores)
    mean_delta = comparison["mean_diff"]
    p_val = comparison["corrected_p_value"]

    # Rule evaluation
    insignificant = (p_val >= alpha)
    below_margin = (mean_delta < parsimony_margin)

    if insignificant or below_margin:
        reason = (
            f"No significant statistical difference (p={p_val:.4f} >= {alpha}) or "
            f"delta ({mean_delta:.4f} < {parsimony_margin}) -> prefer simpler/more interpretable model"
        )
        logger.info(f"Parsimony rule triggered: {reason}")
        return {
            "selected_model": "logistic_l2",
            "rule_applied": reason,
            "fallback_to_logistic": True,
            "comparison": comparison,
        }
    else:
        reason = (
            f"Top model {best_model_name} significantly outperforms Logistic Regression "
            f"(delta={mean_delta:.4f}, p={p_val:.4f})"
        )
        logger.info(reason)
        return {
            "selected_model": best_model_name,
            "rule_applied": reason,
            "fallback_to_logistic": False,
            "comparison": comparison,
        }
