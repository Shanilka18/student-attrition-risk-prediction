"""Deterministic Plain-Language Narrative Generator for Student Explanations.

Converts local SHAP contributions into human-interpretable institutional narratives:
- Identifies Top 3 positive risk drivers and protective factors
- Uses strict template-based synthesis (zero LLM hallucinations)
- Enforces mandatory anti-causal guardrails in every single explanation
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

# Plain English descriptions for transformed feature names
FEATURE_DESCRIPTIONS = {
    "commute_time_min": "daily commute duration",
    "commute_cost_lkr_week": "weekly commute transit cost",
    "financial_stress": "self-reported financial stress level",
    "household_income_bracket": "household income bracket",
    "internet_reliability": "home internet connectivity reliability",
    "internet_affordability": "internet service affordability",
    "device_quality": "learning device condition and quality",
    "device_type": "type of primary computing device",
    "work_hours_per_week": "weekly part-time employment hours",
    "dependents_count": "family dependent responsibilities",
    "quiet_study_space": "access to quiet study environment",
    "scholarship_bursary": "bursary or scholarship funding status",
    "part_time_employment": "part-time employment status",
    "baseline_gpa_band": "prior cumulative academic performance band",
    "housing_type": "residential housing and boarding arrangement",
    "district_type": "home district geographic classification",
    "year_of_study": "undergraduate academic year",
    "faculty": "academic faculty affiliation",
}

# Strictly forbidden causal terms to ensure XAI integrity
FORBIDDEN_CAUSAL_WORDS = [
    "causes",
    "caused by",
    "causing",
    "leads to",
    "leading to",
    "led to",
    "results in",
    "resulting in",
    "produces",
    "triggering causation",
    "proves causation",
]

MANDATORY_DISCLAIMER = "This explains the model's score. It does not prove these factors caused the outcome."


@lru_cache(maxsize=512)
def clean_feature_label(feat_name: str) -> str:
    """Map internal transformed column to friendly natural language description."""
    # Look for matching base feature
    for base, desc in FEATURE_DESCRIPTIONS.items():
        if base in feat_name:
            if "_" in feat_name.replace(base, ""):
                # One-hot encoded category
                cat_val = feat_name.split(base)[-1].strip("_")
                return f"{desc} ({cat_val})"
            return desc
    return feat_name.replace("_", " ")


def generate_student_narrative(
    participant_id: str,
    eri_score: float,
    feature_names: list[str],
    shap_vector: np.ndarray,
    top_k: int = 3,
) -> dict[str, Any]:
    """Generate deterministic, template-based explanation narrative for an individual student.

    Converts raw log-odds or continuous SHAP contributions into relative point indicators.
    """
    assert len(feature_names) == len(
        shap_vector
    ), "Feature names and SHAP vector dimensions mismatch"

    # Separate positive (risk-increasing) and negative (risk-reducing / protective) contributions
    pos_indices = [i for i, v in enumerate(shap_vector) if v > 0]
    neg_indices = [i for i, v in enumerate(shap_vector) if v < 0]

    # Sort by absolute magnitude
    pos_sorted = sorted(pos_indices, key=lambda i: abs(shap_vector[i]), reverse=True)[
        :top_k
    ]
    neg_sorted = sorted(neg_indices, key=lambda i: abs(shap_vector[i]), reverse=True)[
        :top_k
    ]

    # Determine risk band
    if eri_score < 25.0:
        band_str = "Low"
    elif eri_score < 50.0:
        band_str = "Moderate"
    elif eri_score < 75.0:
        band_str = "High"
    else:
        band_str = "Critical"

    # Scale factor to project SHAP log-odds to intuitive ERI point contributions
    total_abs = np.sum(np.abs(shap_vector)) + 1e-8
    # Baseline expected anchor around 18 points (dataset prevalence)
    delta_from_base = eri_score - 18.0

    drivers = []
    for idx in pos_sorted:
        lbl = clean_feature_label(feature_names[idx])
        # Approximate relative point weight
        pts = max(
            1,
            round((abs(shap_vector[idx]) / total_abs) * max(abs(delta_from_base), 15)),
        )
        drivers.append(
            {
                "feature": feature_names[idx],
                "description": lbl,
                "points": pts,
                "shap_raw": float(shap_vector[idx]),
            }
        )

    protective = []
    for idx in neg_sorted:
        lbl = clean_feature_label(feature_names[idx])
        pts = max(
            1,
            round((abs(shap_vector[idx]) / total_abs) * max(abs(delta_from_base), 15)),
        )
        protective.append(
            {
                "feature": feature_names[idx],
                "description": lbl,
                "points": -pts,
                "shap_raw": float(shap_vector[idx]),
            }
        )

    # Assemble narrative string
    driver_strs = [f"{d['description']} (+{d['points']} pts)" for d in drivers]
    prot_strs = [f"{p['description']} ({p['points']} pts)" for p in protective]

    parts = [f"{band_str} environmental risk (EFI {round(eri_score)}/100)."]
    if driver_strs:
        parts.append(f"Main drivers: {', '.join(driver_strs)}.")
    if prot_strs:
        parts.append(f"Protective: {', '.join(prot_strs)}.")
    parts.append(MANDATORY_DISCLAIMER)

    narrative_text = " ".join(parts)

    # Runtime assertion against causal language
    lowered = narrative_text.lower()
    for forbidden in FORBIDDEN_CAUSAL_WORDS:
        assert (
            forbidden not in lowered
        ), f"Violation: Narrative contained forbidden causal term '{forbidden}'"

    assert (
        MANDATORY_DISCLAIMER in narrative_text
    ), "Violation: Narrative missing mandatory anti-causal disclaimer"

    return {
        "participant_id": participant_id,
        "eri_score": round(float(eri_score), 2),
        "risk_band": band_str,
        "narrative": narrative_text,
        "main_drivers": drivers,
        "protective_factors": protective,
        "disclaimer": MANDATORY_DISCLAIMER,
    }


def batch_generate_narratives(
    sample_ids: list[str],
    eri_scores: np.ndarray,
    feature_names: list[str],
    shap_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    """Generate structured narratives across all students in cohort."""
    narratives = []
    for i, pid in enumerate(sample_ids):
        score = eri_scores[i]
        sv = shap_matrix[i]
        narratives.append(generate_student_narrative(pid, score, feature_names, sv))
    return narratives
