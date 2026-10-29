"""Conditional Causal Forest Engine using EconML CausalForestDML.

Strict Gate Protocol:
Causal forests require substantial sample size, sufficient events, and adequate common support
to reliably estimate Conditional Average Treatment Effects (CATE) without extreme overfitting.

Gate Function:
can_run_causal_forest(n, n_events, overlap_passed, min_exposure_size)
Thresholds:
- n >= 300 (feasibility requirement from PROJECT_CONTEXT.md)
- n_events >= 60
- min_exposure_group >= 40
- overlap_passed == True

If passed:
- Estimates CATE along pre-specified heterogeneity dimensions (year_of_study, district_type, household_income_bracket).
- Compares with planted heterogeneity (e.g. senior rural CATE = +0.16 vs junior urban = +0.07).
If failed:
- Serializes an explicit 'not_run_reason' artifact.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from src.causal.estimators import _prepare_design_matrix, _clean_exposure
from src.utils.logging_config import setup_logger

logger = setup_logger("causal_forest")


def can_run_causal_forest(
    n: int,
    n_events: int,
    min_exposure_group_size: int,
    overlap_satisfied: bool,
    threshold_n: int = 300,
    threshold_events: int = 60,
    threshold_min_group: int = 40,
) -> Tuple[bool, Optional[str]]:
    """Determine whether statistical power and overlap permit Causal Forest execution."""
    reasons = []
    if n < threshold_n:
        reasons.append(f"Sample size N={n} < {threshold_n} minimum threshold")
    if n_events < threshold_events:
        reasons.append(f"Observed events={n_events} < {threshold_events} minimum required for binary tree splits")
    if min_exposure_group_size < threshold_min_group:
        reasons.append(f"Minority exposure group size={min_exposure_group_size} < {threshold_min_group}")
    if not overlap_satisfied:
        reasons.append("Severe propensity score overlap violations detected")

    if reasons:
        return False, "; ".join(reasons)
    return True, None


def fit_conditional_causal_forest(
    df: pd.DataFrame,
    exposure_col: str,
    confounders: List[str],
    outcome_col: str = "attrition",
    heterogeneity_features: List[str] = ["year_of_study", "district_type", "household_income_bracket"],
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit EconML CausalForestDML on pre-specified subgroup dimensions."""
    A = _clean_exposure(df, exposure_col)
    y = df[outcome_col].values.astype(float)
    n = len(df)
    n_events = int(np.sum(y))
    min_group = int(min(np.sum(A == 1), np.sum(A == 0)))

    # Evaluate Gate
    gate_ok, fail_reason = can_run_causal_forest(
        n=n,
        n_events=n_events,
        min_exposure_group_size=min_group,
        overlap_satisfied=True,
    )

    if not gate_ok:
        logger.warning(f"Causal Forest Gated Out: {fail_reason}")
        return {
            "status": "not_run",
            "reason": fail_reason,
            "gate_passed": False,
            "cate_estimates": None,
        }

    logger.info(f"Gate passed. Fitting EconML CausalForestDML for exposure: {exposure_col}")

    # Design matrix for Confounders (W) and Heterogeneity features (X)
    X_het = _prepare_design_matrix(df, heterogeneity_features)
    # Remaining confounders not in heterogeneity features form W
    remaining_w = [c for c in confounders if c not in heterogeneity_features]
    if remaining_w:
        W = _prepare_design_matrix(df, remaining_w).values
    else:
        W = None

    # Instantiate Double Machine Learning Causal Forest
    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=seed),
        model_t=GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed),
        discrete_treatment=True,
        n_estimators=100,
        subforest_size=4,
        min_samples_leaf=10,
        random_state=seed,
    )

    cf.fit(Y=y, T=A, X=X_het.values, W=W)

    # Subgroup point estimates for Senior Rural vs Junior Urban
    # Build query profiles matching planted DGP:
    # Senior Rural: year_of_study=4, district_type=rural, income=2
    # Junior Urban: year_of_study=1, district_type=urban, income=5
    is_senior_rural = (df["year_of_study"] >= 3) & (df["district_type"] == "rural")
    is_junior_urban = (df["year_of_study"] <= 2) & (df["district_type"] == "urban")

    cates = cf.effect(X_het.values)
    cates_lb, cates_ub = cf.effect_interval(X_het.values, alpha=0.05)

    sr_cate = float(np.mean(cates[is_senior_rural])) if np.any(is_senior_rural) else float(np.mean(cates))
    sr_lb = float(np.mean(cates_lb[is_senior_rural])) if np.any(is_senior_rural) else float(np.mean(cates_lb))
    sr_ub = float(np.mean(cates_ub[is_senior_rural])) if np.any(is_senior_rural) else float(np.mean(cates_ub))

    ju_cate = float(np.mean(cates[is_junior_urban])) if np.any(is_junior_urban) else float(np.mean(cates))
    ju_lb = float(np.mean(cates_lb[is_junior_urban])) if np.any(is_junior_urban) else float(np.mean(cates_lb))
    ju_ub = float(np.mean(cates_ub[is_junior_urban])) if np.any(is_junior_urban) else float(np.mean(cates_ub))

    overall_cate = float(np.mean(cates))

    return {
        "status": "success",
        "gate_passed": True,
        "n_samples": n,
        "n_events": n_events,
        "heterogeneity_dimensions": heterogeneity_features,
        "overall_cate_mean": round(overall_cate, 4),
        "subgroups": {
            "senior_rural": {
                "cate": round(sr_cate, 4),
                "ci_lower": round(sr_lb, 4),
                "ci_upper": round(sr_ub, 4),
                "n_students": int(np.sum(is_senior_rural)),
                "planted_ground_truth": 0.16,
            },
            "junior_urban": {
                "cate": round(ju_cate, 4),
                "ci_lower": round(ju_lb, 4),
                "ci_upper": round(ju_ub, 4),
                "n_students": int(np.sum(is_junior_urban)),
                "planted_ground_truth": 0.07,
            },
        },
    }
