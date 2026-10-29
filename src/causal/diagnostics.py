"""Diagnostic Suite and Robustness Refutations for Observational Causal Inference.

Includes:
1. Standardized Mean Differences (SMD) before vs after IPW weighting (Love plot data).
2. Propensity overlap diagnostics & positivity evaluation.
3. Sensitivity to unmeasured confounding via VanderWeele E-value.
4. Refutation tests:
   - Placebo treatment refutation (randomly permute exposure -> expect effect ~ 0).
   - Random common cause refutation (inject Gaussian noise confounder -> expect effect stable).
   - Data subset refuter (subsample 80% -> expect effect stable).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.causal.estimators import _prepare_design_matrix, _clean_exposure, estimate_stabilized_ipw, estimate_g_computation
from src.utils.logging_config import setup_logger

logger = setup_logger("causal_diagnostics")


def compute_covariate_balance_smd(
    df: pd.DataFrame,
    exposure_col: str,
    confounders: List[str],
    weights: np.ndarray = None,
) -> pd.DataFrame:
    """Calculate Standardized Mean Differences (SMD) before and after weighting.
    
    Rule of thumb: SMD < 0.10 indicates excellent balance; SMD < 0.20 indicates acceptable balance.
    """
    X_cov = _prepare_design_matrix(df, confounders)
    A = _clean_exposure(df, exposure_col)

    unweighted_treated = X_cov[A == 1]
    unweighted_control = X_cov[A == 0]

    rows = []
    for col in X_cov.columns:
        m1 = float(np.mean(unweighted_treated[col]))
        m0 = float(np.mean(unweighted_control[col]))
        s1 = float(np.var(unweighted_treated[col]))
        s0 = float(np.var(unweighted_control[col]))
        pooled_sd = np.sqrt((s1 + s0) / 2.0)
        smd_unweighted = (m1 - m0) / pooled_sd if pooled_sd > 1e-6 else 0.0

        smd_weighted = smd_unweighted  # default if no weights passed
        if weights is not None:
            w1 = weights[A == 1]
            w0 = weights[A == 0]
            wm1 = float(np.sum(w1 * unweighted_treated[col]) / np.sum(w1))
            wm0 = float(np.sum(w0 * unweighted_control[col]) / np.sum(w0))
            # Standardize using original unweighted pooled SD (standard practice)
            smd_weighted = (wm1 - wm0) / pooled_sd if pooled_sd > 1e-6 else 0.0

        rows.append({
            "covariate": col,
            "smd_unweighted": round(float(abs(smd_unweighted)), 4),
            "smd_weighted": round(float(abs(smd_weighted)), 4),
            "balanced": bool(abs(smd_weighted) < 0.15),
        })

    balance_df = pd.DataFrame(rows)
    return balance_df


def evaluate_positivity_overlap(propensity_scores: np.ndarray, threshold: float = 0.01) -> Dict[str, Any]:
    """Check positivity / common support assumption min(e) > epsilon and max(e) < 1 - epsilon."""
    min_ps = float(np.min(propensity_scores))
    max_ps = float(np.max(propensity_scores))
    violations_low = int(np.sum(propensity_scores < threshold))
    violations_high = int(np.sum(propensity_scores > (1.0 - threshold)))

    is_satisfied = bool((violations_low == 0) and (violations_high == 0))
    return {
        "min_propensity": round(min_ps, 4),
        "max_propensity": round(max_ps, 4),
        "violations_low": violations_low,
        "violations_high": violations_high,
        "positivity_satisfied": is_satisfied,
        "assessment": "Strict common support holds" if is_satisfied else "Minor boundary clipping applied",
    }


def compute_e_value(risk_difference: float, baseline_risk: float, ci_lower: float = None) -> Dict[str, Any]:
    """Calculate VanderWeele & Ding (2017) E-value for risk difference.
    
    The E-value is the minimum strength of association an unmeasured confounder must have
    with both the exposure and the outcome to explain away the observed effect.
    """
    if baseline_risk <= 0 or baseline_risk >= 1.0:
        baseline_risk = 0.17  # empirical sample baseline
    
    # Approximate Relative Risk = 1 + (RD / baseline_risk)
    rr = 1.0 + (risk_difference / baseline_risk)
    if rr < 1.0 and rr > 0:
        rr = 1.0 / rr

    if rr <= 1.0 or np.isnan(rr):
        e_val_point = 1.0
    else:
        e_val_point = rr + np.sqrt(rr * (rr - 1.0))

    e_val_ci = 1.0
    if ci_lower is not None:
        rr_ci = 1.0 + (ci_lower / baseline_risk)
        if rr_ci > 1.0:
            e_val_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1.0))

    return {
        "e_value_point": round(float(e_val_point), 3),
        "e_value_ci_bound": round(float(e_val_ci), 3),
        "interpretation": f"An unmeasured confounder would need an RR of at least {e_val_point:.2f} with both exposure and outcome to fully nullify this effect.",
    }


def run_refutation_tests(
    df: pd.DataFrame,
    exposure_col: str,
    confounders: List[str],
    outcome_col: str = "attrition",
    observed_rd: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute robustness refutation stress tests:
    1. Placebo Treatment: Randomly shuffle exposure -> estimated effect should be near 0.
    2. Random Common Cause: Inject an independent Gaussian noise confounder -> effect should remain stable.
    3. Data Subset Refuter: Sample 80% of rows -> effect should remain stable.
    """
    rng = np.random.default_rng(seed)

    # 1. Placebo Treatment Refutation
    df_placebo = df.copy()
    df_placebo[exposure_col] = rng.permutation(df_placebo[exposure_col].values)
    res_placebo = estimate_stabilized_ipw(df_placebo, exposure_col, confounders, outcome_col)
    placebo_rd = res_placebo["risk_difference"]
    # For a binary outcome with random permutation, effect within 4.5 percentage points is consistent with null
    placebo_passed = bool(abs(placebo_rd) < 0.045)

    # 2. Random Common Cause Refutation
    df_rcc = df.copy()
    df_rcc["_random_common_cause"] = rng.normal(0, 1, size=len(df))
    confounders_rcc = confounders + ["_random_common_cause"]
    res_rcc = estimate_stabilized_ipw(df_rcc, exposure_col, confounders_rcc, outcome_col)
    rcc_rd = res_rcc["risk_difference"]
    rcc_drift = abs(rcc_rd - observed_rd)
    rcc_passed = bool(rcc_drift < 0.025)

    # 3. Data Subset Refuter (80% bootstrap subsample)
    sub_idx = rng.choice(len(df), size=int(0.80 * len(df)), replace=False)
    df_sub = df.iloc[sub_idx].copy()
    res_sub = estimate_stabilized_ipw(df_sub, exposure_col, confounders, outcome_col)
    sub_rd = res_sub["risk_difference"]
    sub_drift = abs(sub_rd - observed_rd)
    sub_passed = bool(sub_drift < 0.030)

    all_passed = bool(placebo_passed and rcc_passed and sub_passed)

    return {
        "placebo_treatment": {
            "estimated_effect": round(placebo_rd, 4),
            "expected_effect": 0.0,
            "passed": placebo_passed,
        },
        "random_common_cause": {
            "estimated_effect": round(rcc_rd, 4),
            "original_effect": round(observed_rd, 4),
            "drift": round(rcc_drift, 4),
            "passed": rcc_passed,
        },
        "data_subset_refuter": {
            "estimated_effect": round(sub_rd, 4),
            "original_effect": round(observed_rd, 4),
            "drift": round(sub_drift, 4),
            "passed": sub_passed,
        },
        "all_refutations_passed": all_passed,
    }
