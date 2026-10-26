"""Causal Effect Estimators for Adjusted Risk Difference.

Estimands:
Adjusted Risk Difference: Delta = E[Y(1) - Y(0)] on the binary attrition outcome.

Estimators:
1. Unadjusted Risk Difference: Naive baseline showing empirical observational bias.
2. Outcome Regression (G-Computation): Parametric logistic regression Q-model with 500-iteration bootstrap 95% CIs.
3. Propensity Score (sIPW): Stabilized Inverse Probability of Treatment Weighting with weight trimming.
4. Propensity Score Matching (Sensitivity): 1:1 nearest-neighbor matching without replacement within caliper.
5. Doubly Robust AIPW: Augmented Inverse Probability Weighting (consistent if either propensity or outcome model is correct).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

from src.utils.logging_config import setup_logger

logger = setup_logger("causal_estimators")


def _prepare_design_matrix(df: pd.DataFrame, confounders: List[str]) -> pd.DataFrame:
    """One-hot encode categorical confounders, impute missing indicators/values, and return numeric matrix."""
    X = df[confounders].copy()
    
    # For any numeric column with missing values (e.g. household_income_bracket MNAR non-response),
    # impute with median and append an explicit missingness indicator column
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            if X[col].isna().any():
                X[f"{col}_is_missing"] = X[col].isna().astype(float)
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
        else:
            # Categorical column with missing
            if X[col].isna().any():
                X[col] = X[col].fillna("Missing")

    cat_cols = [c for c in confounders if X[c].dtype == "object" or X[c].dtype.name == "category"]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X.astype(float)


def _clean_exposure(df: pd.DataFrame, exposure_col: str) -> np.ndarray:
    """Extract exposure array ensuring no NaNs exist (handling MNAR income definition if needed)."""
    A = df[exposure_col].copy()
    if A.isna().any():
        if exposure_col == "financial_hardship":
            med_inc = df["household_income_bracket"].median()
            imp_inc = df["household_income_bracket"].fillna(med_inc)
            A = ((imp_inc <= 2) & (df["financial_stress"] >= 4)).astype(float)
        else:
            A = A.fillna(0.0)
    return A.values.astype(float)


def estimate_unadjusted_rd(df: pd.DataFrame, exposure_col: str, outcome_col: str = "attrition") -> Dict[str, Any]:
    """Compute naive unadjusted risk difference: E[Y | A=1] - E[Y | A=0]."""
    A = _clean_exposure(df, exposure_col)
    y = df[outcome_col].values

    treated = y[A == 1]
    control = y[A == 0]

    p1 = float(np.mean(treated))
    p0 = float(np.mean(control))
    rd = p1 - p0

    # Large sample standard error for risk difference
    se = float(np.sqrt((p1 * (1 - p1) / len(treated)) + (p0 * (1 - p0) / len(control))))
    ci_lower = rd - 1.96 * se
    ci_upper = rd + 1.96 * se

    return {
        "estimator": "unadjusted_rd",
        "risk_difference": round(rd, 4),
        "std_error": round(se, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p1_treated": round(p1, 4),
        "p0_control": round(p0, 4),
        "n_treated": len(treated),
        "n_control": len(control),
    }


def estimate_g_computation(
    df: pd.DataFrame,
    exposure_col: str,
    confounders: List[str],
    outcome_col: str = "attrition",
    n_bootstraps: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Outcome Regression / G-Computation for Average Treatment Effect (Risk Difference).
    
    Fits Q(A, W) = E[Y | A, W] via logistic regression.
    Predicts counterfactuals Y_hat(1) and Y_hat(0) across the entire cohort.
    Bootstrap percentile 95% confidence intervals.
    """
    X_cov = _prepare_design_matrix(df, confounders)
    A = _clean_exposure(df, exposure_col)
    y = df[outcome_col].values

    def _fit_predict(X_sample: pd.DataFrame, A_sample: np.ndarray, y_sample: np.ndarray) -> float:
        X_design = X_sample.copy()
        X_design["_exposure"] = A_sample
        model = LogisticRegression(max_iter=1000, penalty="l2", C=1.0)
        model.fit(X_design, y_sample)

        X_1 = X_sample.copy()
        X_1["_exposure"] = 1
        p1 = model.predict_proba(X_1)[:, 1]

        X_0 = X_sample.copy()
        X_0["_exposure"] = 0
        p0 = model.predict_proba(X_0)[:, 1]

        return float(np.mean(p1) - np.mean(p0))

    # Point estimate
    point_rd = _fit_predict(X_cov, A, y)

    # Bootstrap for valid CI
    rng = np.random.default_rng(seed)
    n = len(df)
    boot_estimates = []
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        try:
            boot_rd = _fit_predict(X_cov.iloc[boot_idx], A[boot_idx], y[boot_idx])
            boot_estimates.append(boot_rd)
        except Exception:
            continue

    boot_arr = np.array(boot_estimates)
    se = float(np.std(boot_arr))
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))

    return {
        "estimator": "g_computation_logistic",
        "risk_difference": round(point_rd, 4),
        "std_error": round(se, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "bootstrap_replicates": len(boot_estimates),
    }


def estimate_stabilized_ipw(
    df: pd.DataFrame,
    exposure_col: str,
    confounders: List[str],
    outcome_col: str = "attrition",
    ps_model_type: str = "logistic",
    trim_percentile: float = 0.01,
) -> Dict[str, Any]:
    """Stabilized Inverse Probability Weighting (sIPW) with weight trimming."""
    X_cov = _prepare_design_matrix(df, confounders)
    A = _clean_exposure(df, exposure_col)
    y = df[outcome_col].values

    if ps_model_type == "gradient_boosting":
        ps_model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    else:
        ps_model = LogisticRegression(max_iter=1000, penalty="l2", C=1.0)

    ps_model.fit(X_cov, A)
    propensity = ps_model.predict_proba(X_cov)[:, 1]
    # Bound away from strict 0 and 1 for numerical stability
    propensity = np.clip(propensity, 1e-4, 1.0 - 1e-4)

    # Marginal exposure probability for stabilization
    p_marginal = float(np.mean(A))

    # Stabilized weights: w = A * P(A=1) / e(X) + (1-A) * P(A=0) / (1 - e(X))
    weights = np.where(A == 1, p_marginal / propensity, (1.0 - p_marginal) / (1.0 - propensity))

    # Truncate / trim extreme weights
    lower_bound = np.percentile(weights, trim_percentile * 100)
    upper_bound = np.percentile(weights, (1.0 - trim_percentile) * 100)
    trimmed_weights = np.clip(weights, lower_bound, upper_bound)

    # Weighted mean risk difference
    w1 = trimmed_weights * A
    w0 = trimmed_weights * (1 - A)
    weighted_p1 = float(np.sum(w1 * y) / np.sum(w1))
    weighted_p0 = float(np.sum(w0 * y) / np.sum(w0))
    ipw_rd = weighted_p1 - weighted_p0

    # Effective sample size (Kish)
    n_eff = float((np.sum(trimmed_weights) ** 2) / np.sum(trimmed_weights ** 2))

    # Robust standard error for weighted difference
    var1 = np.sum((w1 * (y - weighted_p1)) ** 2) / (np.sum(w1) ** 2)
    var0 = np.sum((w0 * (y - weighted_p0)) ** 2) / (np.sum(w0) ** 2)
    se = float(np.sqrt(var1 + var0))

    ci_lower = ipw_rd - 1.96 * se
    ci_upper = ipw_rd + 1.96 * se

    return {
        "estimator": f"stabilized_ipw_{ps_model_type}",
        "risk_difference": round(ipw_rd, 4),
        "std_error": round(se, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "effective_sample_size": round(n_eff, 1),
        "max_weight": round(float(np.max(trimmed_weights)), 3),
        "min_weight": round(float(np.min(trimmed_weights)), 3),
        "propensity_scores": propensity,
        "weights": trimmed_weights,
    }


def estimate_matching_sensitivity(
    df: pd.DataFrame,
    exposure_col: str,
    confounders: List[str],
    outcome_col: str = "attrition",
    caliper_std: float = 0.25,
) -> Dict[str, Any]:
    """1:1 Propensity Score Nearest-Neighbor Matching within caliper as sensitivity analysis."""
    X_cov = _prepare_design_matrix(df, confounders)
    A = _clean_exposure(df, exposure_col)
    y = df[outcome_col].values

    ps_model = LogisticRegression(max_iter=1000, penalty="l2", C=1.0)
    ps_model.fit(X_cov, A)
    propensity = ps_model.predict_proba(X_cov)[:, 1]

    treated_idx = np.where(A == 1)[0]
    control_idx = np.where(A == 0)[0]

    caliper = caliper_std * np.std(propensity)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(propensity[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(propensity[treated_idx].reshape(-1, 1))

    matched_treated = []
    matched_control = []
    for t_i, dist, c_nn_idx in zip(treated_idx, distances.flatten(), indices.flatten()):
        if dist <= caliper:
            matched_treated.append(t_i)
            matched_control.append(control_idx[c_nn_idx])

    if len(matched_treated) == 0:
        return {
            "estimator": "ps_matching_1to1",
            "risk_difference": np.nan,
            "std_error": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_matched_pairs": 0,
        }

    y_treated_m = y[matched_treated]
    y_control_m = y[matched_control]
    diff = y_treated_m - y_control_m
    att = float(np.mean(diff))
    se = float(np.std(diff) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0

    return {
        "estimator": "ps_matching_1to1",
        "risk_difference": round(att, 4),
        "std_error": round(se, 4),
        "ci_lower": round(att - 1.96 * se, 4),
        "ci_upper": round(att + 1.96 * se, 4),
        "n_matched_pairs": len(matched_treated),
        "unmatched_treated": len(treated_idx) - len(matched_treated),
    }


def estimate_doubly_robust_aipw(
    df: pd.DataFrame,
    exposure_col: str,
    confounders: List[str],
    outcome_col: str = "attrition",
) -> Dict[str, Any]:
    """Augmented Inverse Probability Weighting (AIPW) Doubly Robust Estimator.
    
    Combines propensity score e(X) and conditional outcome regressions mu1(X), mu0(X).
    Consistent if EITHER the propensity score OR the outcome regression is correctly specified.
    """
    X_cov = _prepare_design_matrix(df, confounders)
    A = _clean_exposure(df, exposure_col)
    y = df[outcome_col].values

    # 1. Propensity Score
    ps_model = LogisticRegression(max_iter=1000, penalty="l2", C=1.0)
    ps_model.fit(X_cov, A)
    propensity = np.clip(ps_model.predict_proba(X_cov)[:, 1], 0.01, 0.99)

    # 2. Outcome Regressions: mu1(X) and mu0(X)
    X_cov_treated = X_cov[A == 1]
    y_treated = y[A == 1]
    q1 = LogisticRegression(max_iter=1000, penalty="l2", C=1.0)
    q1.fit(X_cov_treated, y_treated)
    mu1 = q1.predict_proba(X_cov)[:, 1]

    X_cov_ctrl = X_cov[A == 0]
    y_ctrl = y[A == 0]
    q0 = LogisticRegression(max_iter=1000, penalty="l2", C=1.0)
    q0.fit(X_cov_ctrl, y_ctrl)
    mu0 = q0.predict_proba(X_cov)[:, 1]

    # AIPW pseudo-outcomes
    gamma1 = mu1 + (A * (y - mu1) / propensity)
    gamma0 = mu0 + ((1.0 - A) * (y - mu0) / (1.0 - propensity))

    aipw_rd = float(np.mean(gamma1) - np.mean(gamma0))
    # Influence curve variance
    influence = gamma1 - gamma0 - aipw_rd
    se = float(np.sqrt(np.mean(influence ** 2) / len(df)))

    ci_lower = aipw_rd - 1.96 * se
    ci_upper = aipw_rd + 1.96 * se

    return {
        "estimator": "doubly_robust_aipw",
        "risk_difference": round(aipw_rd, 4),
        "std_error": round(se, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }


def compare_all_estimators(
    df: pd.DataFrame,
    exposure_col: str,
    confounders: List[str],
    outcome_col: str = "attrition",
) -> List[Dict[str, Any]]:
    """Run all 5 estimators side by side to make confounding bias and adjustment convergence transparent."""
    logger.info(f"Comparing causal estimators for exposure '{exposure_col}'")
    results = [
        estimate_unadjusted_rd(df, exposure_col, outcome_col),
        estimate_g_computation(df, exposure_col, confounders, outcome_col),
        estimate_stabilized_ipw(df, exposure_col, confounders, outcome_col, ps_model_type="logistic"),
        estimate_stabilized_ipw(df, exposure_col, confounders, outcome_col, ps_model_type="gradient_boosting"),
        estimate_matching_sensitivity(df, exposure_col, confounders, outcome_col),
        estimate_doubly_robust_aipw(df, exposure_col, confounders, outcome_col),
    ]
    return results
