"""Plain-Language Causal Summary Generator with Strict Guardrails.

Rules:
1. Every causal statement MUST be labelled:
   "Adjusted association under stated assumptions (observational data); not experimentally validated".
2. ZERO mixing with predictive SHAP numbers.
3. Quantifies effect on the Risk Difference percentage-point scale.
4. Highlights unadjusted vs adjusted gap to show confounding direction.
"""

from typing import Any, Dict, List

MANDATORY_CAUSAL_HEADER = "Adjusted association under stated assumptions (observational data); not experimentally validated"


def generate_causal_narrative(
    exposure_name: str,
    unadjusted_rd: float,
    adjusted_rd: float,
    ci_lower: float,
    ci_upper: float,
    e_value: float,
    refutations_passed: bool,
    planted_ate: float = None,
) -> Dict[str, Any]:
    """Generate a methodologically responsible, non-hallucinating plain-language causal narrative."""
    # Convert to percentage points
    unadj_pp = unadjusted_rd * 100.0
    adj_pp = adjusted_rd * 100.0
    ci_l_pp = ci_lower * 100.0
    ci_u_pp = ci_upper * 100.0

    readable_exposure = exposure_name.replace("_", " ").title()

    confounding_bias = unadj_pp - adj_pp
    if confounding_bias > 0.5:
        bias_desc = f"Naive comparison overstates risk by {confounding_bias:.1f} percentage points due to baseline socioeconomic confounding."
    elif confounding_bias < -0.5:
        bias_desc = f"Naive comparison masks risk by {abs(confounding_bias):.1f} percentage points due to negative confounding."
    else:
        bias_desc = "Confounder adjustment yields estimates aligned closely with raw observational differences."

    refutation_status = "Robust to placebo exposure and noise confounder stress-tests." if refutations_passed else "Sensitive to refutation perturbations; interpret with heightened caution."

    narrative = (
        f"Under baseline confounder adjustment, {readable_exposure} is associated with a "
        f"{adj_pp:+.1f} percentage-point difference in attrition risk (95% CI: [{ci_l_pp:+.1f}%, {ci_u_pp:+.1f}%]). "
        f"{bias_desc} "
        f"E-value sensitivity analysis indicates an unmeasured confounder would need an RR of {e_value:.2f} "
        f"with both exposure and attrition to nullify this estimate. {refutation_status}"
    )

    result = {
        "type": "causal_inference",
        "header": MANDATORY_CAUSAL_HEADER,
        "exposure": exposure_name,
        "readable_exposure": readable_exposure,
        "narrative": narrative,
        "metrics": {
            "unadjusted_rd_pct": round(unadj_pp, 2),
            "adjusted_rd_pct": round(adj_pp, 2),
            "ci_95_pct": [round(ci_l_pp, 2), round(ci_u_pp, 2)],
            "confounding_gap_pp": round(confounding_bias, 2),
            "e_value": e_value,
            "refutations_passed": refutations_passed,
        },
    }

    if planted_ate is not None:
        result["validation"] = {
            "planted_ground_truth_ate_pct": round(planted_ate * 100.0, 2),
            "absolute_error_pp": round(abs(adj_pp - planted_ate * 100.0), 2),
            "covered_in_95_ci": bool(ci_lower <= planted_ate <= ci_upper),
        }

    return result
