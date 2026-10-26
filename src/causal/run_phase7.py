"""Phase 7 Applied Causal Inference Pipeline Orchestrator.

Executes:
1. DAG generation and visualization export (reports/figures/dag.png).
2. Estimators comparison across 4 exposures:
   - financial_hardship
   - long_commute
   - inadequate_digital_access
   - heavy_part_time_work
3. Diagnostic suite: Balance (SMD before/after), Positivity, E-value, Refutations.
4. Ground-truth recovery benchmarking against data/synthetic/ground_truth.json.
5. Conditional Causal Forest (EconML CausalForestDML) for heterogeneity.
6. Guardrailed Plain-language Causal Summaries with strict type separation.
7. Artifact serialization to artifacts/causal/ and artifacts/frontend_data/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.causal.dag import build_causal_dag, get_adjustment_set, render_dag_figure, EXPOSURES, CONFOUNDERS
from src.causal.estimators import (
    compare_all_estimators,
    estimate_stabilized_ipw,
    estimate_g_computation,
    estimate_unadjusted_rd,
)
from src.causal.diagnostics import (
    compute_covariate_balance_smd,
    evaluate_positivity_overlap,
    compute_e_value,
    run_refutation_tests,
)
from src.causal.causal_forest import fit_conditional_causal_forest
from src.causal.narrative import generate_causal_narrative, MANDATORY_CAUSAL_HEADER
from src.utils.logging_config import setup_logger
from src.utils.paths import ensure_dir

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely handle numpy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

logger = setup_logger("run_phase7")


def run_phase7_causal_inference(
    data_path: Path = Path("data/interim/clean.csv"),
    ground_truth_path: Path = Path("data/synthetic/ground_truth.json"),
    output_dir: Path = Path("artifacts/causal"),
    frontend_dir: Path = Path("artifacts/frontend_data"),
) -> Dict[str, Any]:
    """Execute end-to-end Phase 7 causal pipeline."""
    logger.info("Initializing Phase 7 Causal Inference Pipeline...")
    ensure_dir(output_dir)
    ensure_dir(frontend_dir)

    # 1. Render DAG figure
    render_dag_figure()

    # 2. Load dataset
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with N={len(df)} rows")

    # Ensure exposure columns have no missing values
    # For financial_hardship, if income is missing but stress >= 4, impute with conservative/median or complete case
    if "financial_hardship" in df.columns and df["financial_hardship"].isna().any():
        med_inc = df["household_income_bracket"].median()
        imputed_inc = df["household_income_bracket"].fillna(med_inc)
        df["financial_hardship"] = ((imputed_inc <= 2) & (df["financial_stress"] >= 4)).astype(int)

    # Load ground truth if available
    ground_truth = None
    if ground_truth_path.exists():
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    # Dictionary to collect results across exposures
    causal_results = {}
    narrative_summaries = []
    love_plot_data = {}

    baseline_prevalence = float(np.mean(df["attrition"]))

    for exposure in EXPOSURES:
        logger.info(f"Processing causal estimation for exposure: '{exposure}'")
        adj_set = get_adjustment_set(exposure)

        # 3. All Estimators comparison
        all_est = compare_all_estimators(df, exposure, adj_set)

        # Extract primary sIPW (logistic) for diagnostics and reporting
        sipw_res = estimate_stabilized_ipw(df, exposure, adj_set)
        gcomp_res = estimate_g_computation(df, exposure, adj_set)
        unadj_res = estimate_unadjusted_rd(df, exposure)

        # 4. Diagnostics
        balance_df = compute_covariate_balance_smd(df, exposure, adj_set, weights=sipw_res["weights"])
        positivity_res = evaluate_positivity_overlap(sipw_res["propensity_scores"])
        e_val_res = compute_e_value(
            risk_difference=sipw_res["risk_difference"],
            baseline_risk=baseline_prevalence,
            ci_lower=sipw_res["ci_lower"],
        )
        refutations_res = run_refutation_tests(
            df=df,
            exposure_col=exposure,
            confounders=adj_set,
            observed_rd=sipw_res["risk_difference"],
        )

        # 5. Ground truth recovery evaluation
        planted_ate = None
        recovery_stats = None
        if ground_truth and "planted_causal_effects" in ground_truth:
            gt_info = ground_truth["planted_causal_effects"].get(exposure)
            if gt_info:
                planted_ate = gt_info.get("theoretical_ate_risk_difference")
                empirical_unadj = gt_info.get("empirical_unadjusted_rd")
                recovery_stats = {
                    "planted_true_ate": planted_ate,
                    "empirical_unadjusted_rd": empirical_unadj,
                    "estimated_sipw_rd": sipw_res["risk_difference"],
                    "estimated_gcomp_rd": gcomp_res["risk_difference"],
                    "sipw_absolute_bias": round(abs(sipw_res["risk_difference"] - planted_ate), 4),
                    "unadjusted_absolute_bias": round(abs(unadj_res["risk_difference"] - planted_ate), 4),
                    "coverage_95_ci": bool(sipw_res["ci_lower"] <= planted_ate <= sipw_res["ci_upper"]),
                    "confounding_removed": round(unadj_res["risk_difference"] - sipw_res["risk_difference"], 4),
                }

        # 6. Generate plain-language summary narrative
        narrative_obj = generate_causal_narrative(
            exposure_name=exposure,
            unadjusted_rd=unadj_res["risk_difference"],
            adjusted_rd=sipw_res["risk_difference"],
            ci_lower=sipw_res["ci_lower"],
            ci_upper=sipw_res["ci_upper"],
            e_value=e_val_res["e_value_point"],
            refutations_passed=refutations_res["all_refutations_passed"],
            planted_ate=planted_ate,
        )
        narrative_summaries.append(narrative_obj)

        # Store love plot data
        love_plot_data[exposure] = balance_df.to_dict(orient="records")

        # Package exposure block (without numpy arrays)
        sipw_clean = {k: v for k, v in sipw_res.items() if k not in ["propensity_scores", "weights"]}
        causal_results[exposure] = {
            "estimators_comparison": all_est,
            "primary_adjusted_estimate": sipw_clean,
            "g_computation_estimate": gcomp_res,
            "unadjusted_estimate": unadj_res,
            "diagnostics": {
                "positivity": positivity_res,
                "e_value": e_val_res,
                "refutations": refutations_res,
            },
            "ground_truth_recovery": recovery_stats,
            "narrative": narrative_obj,
        }

    # 7. Conditional Causal Forest Gate & Estimation for Financial Hardship
    cf_res = fit_conditional_causal_forest(
        df=df,
        exposure_col="financial_hardship",
        confounders=get_adjustment_set("financial_hardship"),
    )

    # 8. Assemble Full Artifact Payload
    full_payload = {
        "metadata": {
            "phase": "Phase 7 - Applied Causal Inference",
            "type": "causal_inference",
            "header": MANDATORY_CAUSAL_HEADER,
            "n_samples": len(df),
            "baseline_attrition_prevalence": round(baseline_prevalence, 4),
            "exposures_analyzed": EXPOSURES,
        },
        "exposures": causal_results,
        "causal_forest": cf_res,
    }

    # Save to artifacts/causal/
    with open(output_dir / "causal_effects_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2, cls=NumpyEncoder)

    with open(output_dir / "love_plot_data.json", "w", encoding="utf-8") as f:
        json.dump(love_plot_data, f, indent=2, cls=NumpyEncoder)

    # Save dedicated frontend summary
    frontend_causal_payload = {
        "type": "causal_inference",
        "header": MANDATORY_CAUSAL_HEADER,
        "narratives": narrative_summaries,
        "causal_forest_heterogeneity": cf_res,
        "love_plot_balance": love_plot_data,
    }
    with open(frontend_dir / "causal_summary.json", "w", encoding="utf-8") as f:
        json.dump(frontend_causal_payload, f, indent=2, cls=NumpyEncoder)

    logger.info("Successfully serialized all Phase 7 causal inference artifacts.")
    return full_payload


if __name__ == "__main__":
    run_phase7_causal_inference()
