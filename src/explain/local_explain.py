"""Local Per-Student Attribution and Waterfall Export Module.

Decomposes individual student risk scores into local additive contributions:
- Isolates top 3 risk-increasing indicators and top 3 risk-decreasing protective factors.
- Generates structured waterfall payloads for UI visualization and API contracts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("explain.local_explain")


def explain_student_instance(
    student_idx: int,
    shap_values: np.ndarray,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: List[str],
    base_value: float,
    study_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract individual student attribution breakdown.

    Args:
        student_idx: Index of student in dataset (0 to N-1).
        shap_values: Array of SHAP values, shape (N, p).
        X: Feature matrix.
        feature_names: Column labels for features.
        base_value: Model base/expected value (phi_0).
        study_id: Pseudonymized study ID.

    Returns:
        Structured dictionary with waterfall decomposition, top risk drivers, and protective buffers.
    """
    row_shap = shap_values[student_idx]
    if isinstance(X, pd.DataFrame):
        row_feat = X.iloc[student_idx].values
    else:
        row_feat = X[student_idx]

    student_id = study_id or f"STUDENT_{student_idx:04d}"

    # Build feature breakdown list
    breakdown = []
    for f_name, f_val, s_val in zip(feature_names, row_feat, row_shap):
        breakdown.append({
            "feature": str(f_name),
            "feature_value": round(float(f_val), 3),
            "shap_value": round(float(s_val), 4),
            "effect_direction": "risk_increasing" if s_val > 0 else "risk_decreasing",
        })

    # Sort positive (risk-increasing) and negative (protective)
    risk_increasing = [item for item in breakdown if item["shap_value"] > 0]
    risk_increasing.sort(key=lambda x: x["shap_value"], reverse=True)

    risk_decreasing = [item for item in breakdown if item["shap_value"] < 0]
    risk_decreasing.sort(key=lambda x: x["shap_value"], reverse=False)  # Most negative first

    # Sum of SHAP values + base value
    total_shap = float(np.sum(row_shap))
    predicted_val = float(base_value + total_shap)

    return {
        "study_id": student_id,
        "student_idx": int(student_idx),
        "base_value": round(float(base_value), 4),
        "total_shap_sum": round(total_shap, 4),
        "predicted_value": round(predicted_val, 4),
        "feature_breakdown": breakdown,
        "top_risk_increasing": risk_increasing,
        "top_risk_decreasing": risk_decreasing,
    }


def export_student_waterfall_json(
    student_explanation: Dict[str, Any],
    output_path: Union[Path, str],
) -> Path:
    """Export local waterfall data as a JSON contract for UI and API consumption."""
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(student_explanation, f, indent=2)

    logger.info("Saved local waterfall JSON to %s", save_path)
    return save_path
