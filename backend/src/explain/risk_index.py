"""
Academic Risk Index (ARI) & Clinical Risk Banding Engine.

Computes:
  ARI = Calibrated Probability x 100  (Range: [0.0, 100.0])
  Risk Bands:
    - Low:      [0.0,  5.0)   (Below operational threshold ~5.0; routine academic tracking)
    - Moderate: [5.0, 12.0)   (Between operational threshold and ~2x base rate; counseling alert)
    - High:     [12.0, 22.0)  (Between 2x and 3.5x base rate; formal intervention required)
    - Critical: [22.0, 100.0] (Top 5-7% extreme risk; emergency dean-level review)

All cut-points are empirically derived from validation data distributions and aligned
with the recall-constrained decision threshold (min_recall >= 0.80).

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("AcademicIntelligence.RiskIndex")

RiskBandName = Literal["Low", "Moderate", "High", "Critical"]


@dataclass(frozen=True)
class RiskBandDefinition:
    name: RiskBandName
    lower_bound: float
    upper_bound: float
    color_hex: str
    action_required: str
    clinical_priority: str
    description: str


# Empirically validated cut-points derived from validation set distributions:
# - Cut-point 1 (5.0): Recall-constrained threshold (Captures >= 80% of true attriters)
# - Cut-point 2 (12.0): 80th percentile of cohort (~2x empirical base rate 6.4%)
# - Cut-point 3 (22.0): 95th percentile of cohort (top 5% extreme attrition risk)
DEFAULT_RISK_BANDS: List[RiskBandDefinition] = [
    RiskBandDefinition(
        name="Low",
        lower_bound=0.0,
        upper_bound=5.0,
        color_hex="#2ca02c",  # Green
        action_required="Standard academic monitoring. No active intervention necessary.",
        clinical_priority="ROUTINE",
        description="Performance within expected retention boundaries. Unlikely to attrite within next 2 semesters.",
    ),
    RiskBandDefinition(
        name="Moderate",
        lower_bound=5.0,
        upper_bound=12.0,
        color_hex="#ff7f0e",  # Orange
        action_required="Peer tutoring invitation and course load advisory notification.",
        clinical_priority="MONITORING",
        description="Early trajectory decline detected. Exceeds operational threshold; warrants preventive counseling.",
    ),
    RiskBandDefinition(
        name="High",
        lower_bound=12.0,
        upper_bound=22.0,
        color_hex="#d62728",  # Red
        action_required="Mandatory academic advising session and financial/personal support check-in.",
        clinical_priority="URGENT",
        description="Severe multi-semester decline in marks and attendance. Immediate intervention needed.",
    ),
    RiskBandDefinition(
        name="Critical",
        lower_bound=22.0,
        upper_bound=100.0,
        color_hex="#7f0000",  # Dark Crimson
        action_required="Emergency faculty-level review, customized study plan, and student welfare outreach.",
        clinical_priority="EMERGENCY",
        description="Extreme attrition risk (top 5% of student body). Imminent withdrawal or academic dismissal.",
    ),
]


class AcademicRiskIndexEngine:
    """
    Transforms calibrated model probabilities into the standardized
    Academic Risk Index (0-100) and maps them to clinical risk tiers.
    """

    def __init__(self, bands: List[RiskBandDefinition] = None):
        self.bands = bands or DEFAULT_RISK_BANDS

    def compute_risk_index(self, calibrated_probs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Computes ARI = p_calibrated * 100, clipped to [0.0, 100.0].
        """
        arr = np.asarray(calibrated_probs, dtype=np.float64)
        ari = np.clip(arr * 100.0, 0.0, 100.0)
        if np.ndim(calibrated_probs) == 0:
            return round(float(ari), 2)
        return np.round(ari, 2)

    def assign_risk_band(self, risk_index: float) -> RiskBandDefinition:
        """
        Assigns a single ARI score to its matching risk band.
        """
        val = float(risk_index)
        for band in self.bands:
            # Handle upper boundary edge case for Critical band
            if band.name == "Critical":
                if val >= band.lower_bound:
                    return band
            elif band.lower_bound <= val < band.upper_bound:
                return band
        return self.bands[-1]  # Default to Critical if somehow > 100

    def batch_assign_bands(self, risk_indices: np.ndarray) -> List[RiskBandDefinition]:
        """
        Batch maps an array of ARI scores to their risk band definitions.
        """
        return [self.assign_risk_band(score) for score in risk_indices]

    def summarize_cohort(self, risk_indices: np.ndarray) -> Dict[str, Any]:
        """
        Computes cohort-level risk band distribution and summary statistics.
        """
        scores = np.asarray(risk_indices, dtype=np.float64)
        n = len(scores)
        counts = {b.name: 0 for b in self.bands}

        for score in scores:
            band = self.assign_risk_band(score)
            counts[band.name] += 1

        pcts = {name: round(count / max(n, 1) * 100.0, 2) for name, count in counts.items()}

        return {
            "n_students": n,
            "mean_ari": round(float(np.mean(scores)), 2),
            "median_ari": round(float(np.median(scores)), 2),
            "std_ari": round(float(np.std(scores)), 2),
            "p25_ari": round(float(np.percentile(scores, 25)), 2),
            "p75_ari": round(float(np.percentile(scores, 75)), 2),
            "p90_ari": round(float(np.percentile(scores, 90)), 2),
            "p95_ari": round(float(np.percentile(scores, 95)), 2),
            "band_counts": counts,
            "band_percentages": pcts,
            "cut_points": {
                "moderate_cutoff": self.bands[1].lower_bound,
                "high_cutoff": self.bands[2].lower_bound,
                "critical_cutoff": self.bands[3].lower_bound,
            },
        }
