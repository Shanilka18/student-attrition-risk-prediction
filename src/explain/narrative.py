"""Deterministic, Template-Based Plain-Language Explanation Generator.

Produces actionable, transparent explanations for academic advisors, counselors,
and students without relying on stochastic LLM APIs. Adheres strictly to the
scientific governance rule: factors are described as 'contributing indicators,
not direct causes', and every summary appends the mandatory research disclaimer.
"""

from typing import Any, Dict, List, Optional
import numpy as np

DISCLAIMER_TEXT: str = (
    "RESEARCH NOTICE: The Psychosocial Risk Index (PRI) is an academic decision-support "
    "indicator developed for educational research at SLIIT Kandy. It is not a clinical "
    "psychological evaluation or psychiatric diagnosis."
)

# Canonical domain naming and descriptions
FACTOR_DISPLAY_NAMES: Dict[str, str] = {
    "stress": "academic and situational stress",
    "factor_stress": "academic and situational stress",
    "risk_stress": "academic and situational stress",
    "motivation": "intrinsic academic motivation",
    "factor_motivation": "intrinsic academic motivation",
    "risk_motivation": "academic motivation deficit",
    "belongingness": "sense of campus belongingness",
    "factor_belongingness": "sense of campus belongingness",
    "risk_belongingness": "campus belongingness deficit",
    "self_efficacy": "academic self-efficacy and belief in capability",
    "factor_self_efficacy": "academic self-efficacy and belief in capability",
    "risk_self_efficacy": "self-efficacy deficit",
    "peer_support": "peer social and academic support networks",
    "factor_peer_support": "peer social and academic support networks",
    "risk_peer_support": "peer support isolation",
}

# Plain language descriptors based on factor value (1.0 to 5.0 scale)
def _describe_factor_level(factor_name: str, score: float) -> str:
    """Format an informative level descriptor for a given factor score."""
    clean_name = factor_name.lower().replace("factor_", "").replace("risk_", "")
    if clean_name == "stress":
        if score >= 3.8:
            return "elevated levels of academic stress"
        elif score >= 2.8:
            return "moderate levels of situational stress"
        else:
            return "low reported stress levels"
    else:
        # Protective factors
        if score >= 3.8:
            return f"strong {FACTOR_DISPLAY_NAMES.get(clean_name, clean_name)}"
        elif score >= 2.8:
            return f"moderate {FACTOR_DISPLAY_NAMES.get(clean_name, clean_name)}"
        else:
            return f"low or diminished {FACTOR_DISPLAY_NAMES.get(clean_name, clean_name)}"


def generate_student_narrative(
    study_id: str,
    pri_score: float,
    risk_category: str,
    top_risk_increasing: List[Dict[str, Any]],
    top_risk_decreasing: List[Dict[str, Any]],
    factor_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, str]:
    """Generate deterministic, human-readable student explanation narratives.

    Args:
        study_id: Pseudonymized student identifier.
        pri_score: Continuous Psychosocial Risk Index (0 to 100).
        risk_category: 'Low', 'Moderate', or 'High'.
        top_risk_increasing: Sorted list of dicts with keys 'feature', 'shap_value', 'feature_value'.
        top_risk_decreasing: Sorted list of dicts with keys 'feature', 'shap_value', 'feature_value'.
        factor_scores: Optional dictionary of raw 1-5 factor scores.

    Returns:
        Dict containing headline, summary narrative, contributing factors breakdown,
        actionable recommendations, and the required ethical research disclaimer.
    """
    pri_round = round(float(pri_score), 1)
    cat_upper = str(risk_category).upper()

    # 1. Primary headline
    headline = f"Student {study_id}: Psychosocial Risk Index is {cat_upper} ({pri_round}/100)."

    # 2. Risk Increasing Sentence
    increasing_phrases = []
    for item in top_risk_increasing[:3]:
        feat = item["feature"].lower().replace("factor_", "").replace("risk_", "")
        val = item.get("feature_value")
        display = FACTOR_DISPLAY_NAMES.get(feat, feat)
        if val is not None and not np.isnan(val):
            desc = _describe_factor_level(feat, float(val))
            increasing_phrases.append(f"{desc} (score: {val:.1f})")
        else:
            increasing_phrases.append(display)

    if increasing_phrases:
        if len(increasing_phrases) == 1:
            inc_text = f"The primary contributing indicator elevating psychosocial risk is {increasing_phrases[0]}."
        elif len(increasing_phrases) == 2:
            inc_text = f"The primary contributing indicators elevating psychosocial risk are {increasing_phrases[0]} and {increasing_phrases[1]}."
        else:
            inc_text = (
                f"The primary contributing indicators elevating psychosocial risk are "
                f"{increasing_phrases[0]}, {increasing_phrases[1]}, and {increasing_phrases[2]}."
            )
    else:
        inc_text = "No prominent risk-elevating indicators were detected for this profile."

    # 3. Protective Buffering Sentence
    decreasing_phrases = []
    for item in top_risk_decreasing[:3]:
        feat = item["feature"].lower().replace("factor_", "").replace("risk_", "")
        val = item.get("feature_value")
        display = FACTOR_DISPLAY_NAMES.get(feat, feat)
        if val is not None and not np.isnan(val):
            desc = _describe_factor_level(feat, float(val))
            decreasing_phrases.append(f"{desc} (score: {val:.1f})")
        else:
            decreasing_phrases.append(display)

    if decreasing_phrases:
        if len(decreasing_phrases) == 1:
            dec_text = f"This profile is actively buffered by {decreasing_phrases[0]}."
        elif len(decreasing_phrases) == 2:
            dec_text = f"This profile is actively buffered by {decreasing_phrases[0]} and {decreasing_phrases[1]}."
        else:
            dec_text = (
                f"This profile is actively buffered by {decreasing_phrases[0]}, "
                f"{decreasing_phrases[1]}, and {decreasing_phrases[2]}."
            )
    else:
        dec_text = "Protective buffering indicators are currently limited."

    # 4. Actionable recommendations based on contributing indicators
    recommendations: List[str] = []
    risk_factor_names = [item["feature"].lower().replace("factor_", "").replace("risk_", "") for item in top_risk_increasing[:3]]

    if "stress" in risk_factor_names:
        recommendations.append("Offer time-management workshops, workload review, or academic counseling check-in.")
    if "belongingness" in risk_factor_names:
        recommendations.append("Facilitate connection with faculty mentorship circles, campus student clubs, or orientation peer groups.")
    if "self_efficacy" in risk_factor_names:
        recommendations.append("Provide targeted academic skill support, low-stakes practice quizzes, or study technique mentoring.")
    if "peer_support" in risk_factor_names:
        recommendations.append("Connect student with peer study cohorts or collaborative lab partner pairings.")
    if "motivation" in risk_factor_names:
        recommendations.append("Schedule an exploratory career alignment session with academic program coordinators.")

    if not recommendations:
        if cat_upper == "LOW":
            recommendations.append("Maintain routine academic advising checkpoints; student profile demonstrates positive self-regulation.")
        else:
            recommendations.append("Schedule general academic check-in to monitor course progression.")

    # 5. Synthesis
    full_narrative = (
        f"{headline} {inc_text} {dec_text} "
        f"Note that these psychosocial factors serve as contributing indicators to support proactive retention, "
        f"rather than deterministic causes."
    )

    return {
        "study_id": study_id,
        "pri_score": pri_round,
        "risk_category": cat_upper,
        "headline": headline,
        "risk_increasing_narrative": inc_text,
        "protective_narrative": dec_text,
        "full_narrative": full_narrative,
        "recommendations": recommendations,
        "disclaimer": DISCLAIMER_TEXT,
    }
