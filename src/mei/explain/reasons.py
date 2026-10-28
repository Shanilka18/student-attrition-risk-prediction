"""Reason generation engine for the MEI explainability subsystem.

Translates mathematical SHAP feature attribution values into plain-English,
empathetic, and non-judgemental reasons for non-technical stakeholders (advisors, faculty, students).
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from mei.config import CONFIG_DIR


def load_reason_templates() -> Dict[str, Dict[str, str]]:
    """Load human-friendly reason templates from config/reason_templates.yaml."""
    tmpl_path = CONFIG_DIR / "reason_templates.yaml"
    if not tmpl_path.exists():
        raise FileNotFoundError(f"Reason templates file not found at: {tmpl_path}")
    with open(tmpl_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def convert_shap_to_mei_points(
    shap_values: np.ndarray,
    mei_score: Optional[float] = None,
    baseline_mei: Optional[float] = None,
) -> np.ndarray:
    """Convert SHAP attribution values into integer points on the MEI 0-100 scale.

    If the explainer operates in probability space, sum(shap_values) = p(x) - E[p].
    Scaling by 100 directly maps to MEI points. Any small rounding discrepancy
    between sum(points) and (mei_score - baseline_mei) is distributed to the largest
    magnitude contribution to guarantee exact additivity.
    """
    raw_points = shap_values * 100.0

    if mei_score is not None and baseline_mei is not None:
        target_delta = round(mei_score - baseline_mei)
        rounded_points = np.round(raw_points).astype(int)
        discrepancy = target_delta - int(np.sum(rounded_points))
        if discrepancy != 0 and len(rounded_points) > 0:
            # Add discrepancy to the largest absolute contributor
            max_idx = int(np.argmax(np.abs(raw_points)))
            rounded_points[max_idx] += discrepancy
        return rounded_points

    return np.round(raw_points).astype(int)


def generate_student_reasons(
    feature_values: Union[np.ndarray, List[float], Dict[str, float]],
    shap_values: Union[np.ndarray, List[float]],
    feature_names: List[str],
    top_n_risk: int = 3,
    top_n_protective: int = 2,
    mei_score: Optional[float] = None,
    baseline_mei: Optional[float] = None,
    templates: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate top risk-increasing and risk-decreasing reasons for a single student.

    Args:
        feature_values: 1D array or dict of raw feature values for the student.
        shap_values: 1D array of local SHAP feature attributions.
        feature_names: List of feature names corresponding to the indices.
        top_n_risk: Number of top risk-increasing drivers to extract (default 3).
        top_n_protective: Number of top protective / risk-reducing drivers (default 2).
        mei_score: Final calculated student MEI score (0-100).
        baseline_mei: Expected cohort baseline MEI score (0-100).
        templates: Optional pre-loaded templates dictionary.

    Returns:
        List of reason dicts, each with keys:
            - text: Plain English reason sentence.
            - direction: 'risk_increasing' | 'risk_decreasing'
            - points: Points contribution in MEI scale (+X or -X).
            - feature: Feature column name.
            - plain_name: Human-readable feature name.
            - value: Student's empirical feature value.
    """
    if templates is None:
        templates = load_reason_templates()

    shap_arr = np.asarray(shap_values, dtype=float)
    if isinstance(feature_values, dict):
        vals_arr = np.array([feature_values.get(f, 0.0) for f in feature_names])
    else:
        vals_arr = np.asarray(feature_values, dtype=float)

    # Convert SHAP values to integer MEI points
    points_arr = convert_shap_to_mei_points(
        shap_arr, mei_score=mei_score, baseline_mei=baseline_mei
    )

    reasons: List[Dict[str, Any]] = []

    # 1. Identify top risk-increasing features (shap > 0)
    pos_indices = np.where(shap_arr > 0)[0]
    if len(pos_indices) > 0:
        sorted_pos = pos_indices[np.argsort(-shap_arr[pos_indices])][:top_n_risk]
        for idx in sorted_pos:
            feat = feature_names[idx]
            pts = int(points_arr[idx])
            tmpl = templates.get(feat, {})
            text = tmpl.get(
                "risk_increasing",
                f"Metrics for {feat.replace('_', ' ')} indicate lower engagement",
            )
            plain_name = tmpl.get("plain_name", feat.replace("_", " ").title())
            reasons.append(
                {
                    "text": text,
                    "direction": "risk_increasing",
                    "points": max(1, pts) if pts > 0 else 1,
                    "feature": feat,
                    "plain_name": plain_name,
                    "value": round(float(vals_arr[idx]), 3),
                }
            )

    # 2. Identify top protective / risk-decreasing features (shap < 0)
    neg_indices = np.where(shap_arr < 0)[0]
    if len(neg_indices) > 0:
        sorted_neg = neg_indices[np.argsort(shap_arr[neg_indices])][:top_n_protective]
        for idx in sorted_neg:
            feat = feature_names[idx]
            pts = int(points_arr[idx])
            tmpl = templates.get(feat, {})
            text = tmpl.get(
                "risk_decreasing",
                f"Maintains positive consistency in {feat.replace('_', ' ')}",
            )
            plain_name = tmpl.get("plain_name", feat.replace("_", " ").title())
            reasons.append(
                {
                    "text": text,
                    "direction": "risk_decreasing",
                    "points": min(-1, pts) if pts < 0 else -1,
                    "feature": feat,
                    "plain_name": plain_name,
                    "value": round(float(vals_arr[idx]), 3),
                }
            )

    return reasons
