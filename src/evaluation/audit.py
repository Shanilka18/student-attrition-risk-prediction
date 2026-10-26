"""Subgroup fairness audit and demographic parity verification.

Audits model performance and calibration across sensitive subgroups:
- gender (Female / Male / Other) - AUDIT ONLY, never a model feature
- district_type (Urban / Semi-Urban / Rural)
- household_income_bracket (1-6)

Computes Equalized Odds, Demographic Parity difference, and calibration error across slices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.metrics import compute_ece
from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir

logger = setup_logger("audit")


def evaluate_slice(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    slice_name: str,
    group_val: Any,
    min_samples: int = 15,
) -> Dict[str, Any]:
    """Compute performance metrics for a specific subgroup slice."""
    n = len(y_true)
    n_pos = int(np.sum(y_true))
    prev = float(n_pos / n) if n > 0 else 0.0

    if n < min_samples or n_pos == 0 or n_pos == n:
        return {
            "group": str(group_val),
            "samples": n,
            "positives": n_pos,
            "prevalence": round(prev, 4),
            "note": "insufficient_positives_or_samples",
            "roc_auc": None,
            "pr_auc": None,
            "brier": round(float(np.mean((y_prob - y_true) ** 2)), 4) if n > 0 else None,
            "ece": round(float(compute_ece(y_true, y_prob)), 4) if n > 0 else None,
        }

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(np.mean((y_prob - y_true) ** 2))
    ece = float(compute_ece(y_true, y_prob))

    # Binary decision threshold at 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    tpr = float(np.sum((y_pred == 1) & (y_true == 1)) / max(n_pos, 1))
    fpr = float(np.sum((y_pred == 1) & (y_true == 0)) / max(n - n_pos, 1))
    selection_rate = float(np.mean(y_pred))

    return {
        "group": str(group_val),
        "samples": n,
        "positives": n_pos,
        "prevalence": round(prev, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "tpr_recall": round(tpr, 4),
        "fpr": round(fpr, 4),
        "selection_rate": round(selection_rate, 4),
    }


def run_subgroup_fairness_audit(
    df_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """Run full demographic fairness audit across gender, district_type, and income."""
    audit_results: Dict[str, Any] = {
        "model": model_name,
        "total_samples": len(df_raw),
        "subgroups": {},
        "disparity_summary": {},
    }

    subgroup_cols = ["gender", "district_type", "household_income_bracket"]

    for col in subgroup_cols:
        if col not in df_raw.columns:
            continue
        slice_reports = []
        for val in sorted(df_raw[col].dropna().unique()):
            idx = (df_raw[col] == val).values
            sub_y = y_true[idx]
            sub_prob = y_prob[idx]
            rep = evaluate_slice(sub_y, sub_prob, col, val)
            slice_reports.append(rep)

        audit_results["subgroups"][col] = slice_reports

        # Calculate demographic parity & equalized odds disparities where valid
        tprs = [r["tpr_recall"] for r in slice_reports if r.get("tpr_recall") is not None]
        fprs = [r["fpr"] for r in slice_reports if r.get("fpr") is not None]
        s_rates = [r["selection_rate"] for r in slice_reports if r.get("selection_rate") is not None]

        max_tpr_diff = round(float(max(tprs) - min(tprs)), 4) if len(tprs) > 1 else 0.0
        max_fpr_diff = round(float(max(fprs) - min(fprs)), 4) if len(fprs) > 1 else 0.0
        max_sr_diff = round(float(max(s_rates) - min(s_rates)), 4) if len(s_rates) > 1 else 0.0

        audit_results["disparity_summary"][col] = {
            "max_equalized_odds_tpr_gap": max_tpr_diff,
            "max_equalized_odds_fpr_gap": max_fpr_diff,
            "demographic_parity_gap": max_sr_diff,
        }

    return audit_results
