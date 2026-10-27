"""
Deterministic Plain-Language Academic Risk Narrative Generator.

Generates institutionally actionable, clinician-grade explanation sentences:
  e.g. "High risk (78.4/100): GPA has fallen across 3 consecutive semester(s),
        latest course attendance dropped to 64.0%, and coursework submissions are delayed by 4.2 days."

Strictly rules-based templates grounded in empirical SHAP values and true feature states.
NEVER free-invented or hallucinated text.

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AcademicIntelligence.SentenceGenerator")


def _format_factor_clause(factor: Dict[str, Any], raw_features: Dict[str, float]) -> str:
    """
    Transforms a single top-ranking factor into an informative plain-language clause.
    """
    name = factor.get("name", "")
    direction = factor.get("direction", "RISK_INCREASING")
    val = float(raw_features.get(name, factor.get("feature_value", 0.0)))

    if name == "gpa_slope":
        if val < -0.05:
            return f"GPA is declining steadily by {abs(val):.2f} grade points per semester"
        elif val > 0.05:
            return f"GPA trajectory is stabilizing with an upward slope (+{val:.2f}/sem)"
        else:
            return f"GPA trajectory is flat ({val:+.2f}/sem)"

    elif name == "consecutive_declines":
        if val >= 1:
            return f"GPA has fallen across {int(val)} consecutive semester(s)"
        else:
            return "academic progress shows no prolonged consecutive decline"

    elif name == "last_gpa":
        if val < 2.0:
            return f"latest semester GPA ({val:.2f}) dropped below the academic probation threshold (2.00)"
        else:
            return f"latest semester GPA is solid at {val:.2f}"

    elif name == "last_attendance_pct":
        if val < 65.0:
            return f"latest course attendance dropped to an exam-ineligible {val:.1f}%"
        elif val < 80.0:
            return f"latest course attendance decreased to {val:.1f}%"
        else:
            return f"attendance is maintained at a healthy {val:.1f}%"

    elif name == "attendance_slope":
        if val < -3.0:
            return f"lecture attendance is declining by {abs(val):.1f}% per semester"
        elif val > 3.0:
            return f"lecture attendance is improving (+{val:.1f}%/sem)"
        else:
            return f"attendance attendance rate is relatively stable ({val:+.1f}%/sem)"

    elif name == "last_assignment_delay_days":
        if val >= 2.0:
            return f"recent coursework submissions are delayed by an average of {val:.1f} days"
        elif val > 0.0:
            return f"minor assignment submission delays detected ({val:.1f} days late)"
        else:
            return "all recent coursework submitted strictly on time (0.0 days delay)"

    elif name == "delay_trend":
        if val > 0.5:
            return f"assignment submission delays are worsening by +{val:.1f} days per semester"
        else:
            return "submission timeliness is consistent across semesters"

    elif name == "sems_below_gpa_2":
        if val >= 1:
            return f"student has accumulated {int(val)} semester(s) below GPA 2.0 on academic probation"
        else:
            return "student has zero probation semesters on record"

    elif name == "sems_below_att_65":
        if val >= 1:
            return f"student has fallen below the mandatory 65% attendance requirement in {int(val)} semester(s)"
        else:
            return "attendance has consistently satisfied minimum exam eligibility standards"

    elif name == "gpa_max_drop":
        if val >= 0.4:
            return f"experienced an abrupt single-semester GPA shock decline of {val:.2f} points"
        else:
            return "no severe historical single-semester GPA shocks observed"

    elif name == "last_exam_marks":
        if val < 45.0:
            return f"latest examination average dropped to a failing {val:.1f}/100"
        elif val < 60.0:
            return f"latest examination average is marginal at {val:.1f}/100"
        else:
            return f"latest examination average is satisfactory at {val:.1f}/100"

    elif name == "exam_mark_slope":
        if val < -3.0:
            return f"examination scores are trending downward by {abs(val):.1f} marks per semester"
        else:
            return f"examination performance is stable or improving ({val:+.1f} marks/sem)"

    elif name == "late_count_total":
        if val >= 3:
            return f"cumulative total of {int(val)} late assignments across semesters"
        else:
            return f"low cumulative late submissions ({int(val)} late tasks)"

    elif name == "cumulative_gpa":
        if val < 2.0:
            return f"cumulative GPA remains below graduation threshold at {val:.2f}"
        else:
            return f"cumulative GPA stands at {val:.2f}"

    # Generic fallback
    disp = factor.get("display_name", name.replace("_", " ").title())
    unit = factor.get("unit", "")
    sign = "elevating attrition risk" if direction == "RISK_INCREASING" else "supporting student retention"
    unit_str = f" {unit}" if unit else ""
    return f"{disp} is {val:.2f}{unit_str} ({sign})"


def generate_explanation_narrative(
    academic_risk_index: float,
    risk_band: str,
    top_factors: List[Dict[str, Any]],
    raw_features: Optional[Dict[str, float]] = None,
) -> str:
    """
    Constructs an institutional diagnostic sentence summarizing student attrition risk.
    """
    feats = raw_features or {}
    ari_str = f"{academic_risk_index:.1f}/100"

    if not top_factors:
        return f"{risk_band} risk ({ari_str}): Insufficient feature attributions available for student."

    # Process up to 3 factors
    clauses = []
    seen_clauses = set()
    for factor in top_factors[:3]:
        clause = _format_factor_clause(factor, feats)
        if clause not in seen_clauses:
            clauses.append(clause)
            seen_clauses.add(clause)

    if not clauses:
        return f"{risk_band} risk ({ari_str}): Academic indicators align with historical cohort baseline."

    if len(clauses) == 1:
        narrative = clauses[0]
    elif len(clauses) == 2:
        narrative = f"{clauses[0]}, and {clauses[1]}"
    else:
        narrative = f"{clauses[0]}, {clauses[1]}, and {clauses[2]}"

    # Capitalize appropriately
    return f"{risk_band} risk ({ari_str}): Key drivers indicate that {narrative}."
