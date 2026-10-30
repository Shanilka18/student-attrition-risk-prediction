"""
SHAP TreeExplainer Engine for Tabular Gradient Boosting & Tree Ensembles.

Computes exact tree-based Shapley values for:
  - Global feature importance (mean absolute SHAP)
  - Local sample-level attributions (direction, size, rank)
  - Top-k risk-increasing and risk-decreasing drivers

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger("AcademicIntelligence.ShapExplainer")

# Friendly labels and units for all 22 engineered features
FEATURE_METADATA: Dict[str, Dict[str, str]] = {
    "last_gpa": {"display_name": "Latest Semester GPA", "unit": "GPA (0.0-4.0)", "domain": "Academic"},
    "last_exam_marks": {"display_name": "Latest Exam Average", "unit": "Marks (0-100)", "domain": "Academic"},
    "last_attendance_pct": {"display_name": "Latest Attendance Rate", "unit": "%", "domain": "Engagement"},
    "last_assignment_delay_days": {"display_name": "Latest Submission Delay", "unit": "Days late", "domain": "Behavioral"},
    "last_late_assignments_count": {"display_name": "Latest Late Submissions", "unit": "Count", "domain": "Behavioral"},
    "gpa_slope": {"display_name": "GPA Trajectory Slope", "unit": "GPA/sem", "domain": "Trend"},
    "exam_mark_slope": {"display_name": "Exam Mark Trajectory", "unit": "Marks/sem", "domain": "Trend"},
    "attendance_slope": {"display_name": "Attendance Trajectory", "unit": "%/sem", "domain": "Trend"},
    "delay_trend": {"display_name": "Assignment Delay Trend", "unit": "Days/sem", "domain": "Trend"},
    "gpa_rolling_mean": {"display_name": "Recent GPA Rolling Mean", "unit": "GPA (0.0-4.0)", "domain": "Academic"},
    "gpa_volatility": {"display_name": "GPA Volatility (Std Dev)", "unit": "GPA", "domain": "Consistency"},
    "cumulative_gpa": {"display_name": "Cumulative Grade Point Average (cGPA)", "unit": "cGPA (0.0-4.0)", "domain": "Academic"},
    "exam_mark_mean": {"display_name": "Historical Exam Average", "unit": "Marks (0-100)", "domain": "Academic"},
    "attendance_mean": {"display_name": "Historical Attendance Mean", "unit": "%", "domain": "Engagement"},
    "attendance_variance": {"display_name": "Attendance Fluctuation (Variance)", "unit": "%^2", "domain": "Consistency"},
    "delay_mean": {"display_name": "Average Assignment Delay", "unit": "Days", "domain": "Behavioral"},
    "delay_max": {"display_name": "Maximum Assignment Delay", "unit": "Days", "domain": "Behavioral"},
    "late_count_total": {"display_name": "Cumulative Late Assignments", "unit": "Count", "domain": "Behavioral"},
    "consecutive_declines": {"display_name": "Consecutive Declining Semesters", "unit": "Semesters", "domain": "Trend"},
    "gpa_max_drop": {"display_name": "Single-Semester Worst GPA Drop", "unit": "GPA points", "domain": "Shock"},
    "sems_below_gpa_2": {"display_name": "Semesters Below GPA 2.0 (Probation)", "unit": "Semesters", "domain": "Probation"},
    "sems_below_att_65": {"display_name": "Semesters Below 65% Attendance", "unit": "Semesters", "domain": "Engagement"},
}


class TabularTreeSHAPExplainer:
    """
    SHAP TreeExplainer wrapper supporting CatBoost, XGBoost, LightGBM, and Random Forest.
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.raw_clf = getattr(model, "_clf", model)

        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.raw_clf)
        self.expected_value = float(np.ravel(self.explainer.expected_value)[-1])
        self._cached_shap_values: Optional[np.ndarray] = None
        self._cached_X: Optional[np.ndarray] = None

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Computes or returns cached SHAP values matrix of shape (n_samples, n_features).
        """
        X = np.asarray(X, dtype=np.float32)
        shap_vals = self.explainer.shap_values(X)

        # Some classifiers (e.g. multiclass / binary list) return list of 2 arrays
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]

        self._cached_shap_values = np.asarray(shap_vals, dtype=np.float32)
        self._cached_X = X
        return self._cached_shap_values

    def get_global_importance(self, X: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Computes mean absolute SHAP importance per feature:
          Importance_j = (1 / N) * sum_i |phi_{i, j}|
        """
        if X is not None or self._cached_shap_values is None:
            shap_vals = self.explain(X)
        else:
            shap_vals = self._cached_shap_values

        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        total_imp = float(np.sum(mean_abs_shap))
        relative_pct = (mean_abs_shap / max(total_imp, 1e-9)) * 100.0

        records = []
        for idx, col in enumerate(self.feature_names):
            meta = FEATURE_METADATA.get(col, {"display_name": col, "unit": "", "domain": "General"})
            records.append({
                "feature_name": col,
                "display_name": meta["display_name"],
                "domain": meta["domain"],
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 4),
                "relative_importance_pct": round(float(relative_pct[idx]), 2),
                "rank": 0,
            })

        df_imp = pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df_imp["rank"] = range(1, len(df_imp) + 1)
        return df_imp

    def get_local_explanation(
        self,
        sample_idx: int,
        X: Optional[np.ndarray] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Extracts structured local attribution for sample_idx:
          - Base value
          - Top-k risk-increasing drivers (positive SHAP)
          - Top-k risk-decreasing stabilizers (negative SHAP)
          - Overall top-k signed drivers
        """
        if X is not None or self._cached_shap_values is None:
            shap_vals = self.explain(X)
            features = X
        else:
            shap_vals = self._cached_shap_values
            features = self._cached_X

        row_shap = shap_vals[sample_idx]
        row_feat = features[sample_idx]

        drivers = []
        for idx, col in enumerate(self.feature_names):
            phi = float(row_shap[idx])
            val = float(row_feat[idx])
            meta = FEATURE_METADATA.get(col, {"display_name": col, "unit": "", "domain": "General"})
            direction = "RISK_INCREASING" if phi > 0 else "RISK_DECREASING"

            drivers.append({
                "name": col,
                "display_name": meta["display_name"],
                "direction": direction,
                "contribution": round(phi, 4),
                "abs_contribution": round(abs(phi), 4),
                "feature_value": round(val, 2),
                "unit": meta["unit"],
                "domain": meta["domain"],
            })

        # Sort by absolute impact
        drivers_sorted = sorted(drivers, key=lambda x: x["abs_contribution"], reverse=True)
        top_factors = drivers_sorted[:top_k]

        risk_increasing = [d for d in drivers_sorted if d["direction"] == "RISK_INCREASING"][:top_k]
        risk_decreasing = [d for d in drivers_sorted if d["direction"] == "RISK_DECREASING"][:top_k]

        return {
            "sample_index": sample_idx,
            "base_value": round(self.expected_value, 4),
            "sum_shap": round(float(np.sum(row_shap)), 4),
            "top_factors": top_factors,
            "risk_increasing_factors": risk_increasing,
            "risk_decreasing_factors": risk_decreasing,
            "all_factors": drivers,
        }
