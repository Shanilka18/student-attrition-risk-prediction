"""Stage 9: Outcome item validation and target score derivation."""

from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

from mei.cleaning.models import RuleLog
from mei.io.schema import LabelsConfig, load_labels


def process_outcomes(
    df: pd.DataFrame,
    labels_cfg: Optional[LabelsConfig] = None,
    mode: str = "training",
    threshold: float = 60.0,
) -> Tuple[pd.DataFrame, Set[int], Optional[Dict[str, Any]], RuleLog]:
    """Process outcome items A1–A3, compute targets, and audit class balance.

    In training mode: rows with missing outcome items are dropped.
    In scoring mode: outcomes are optional; scorable rows are kept.

    Returns:
        (processed_df, missing_outcome_indices, class_balance, RuleLog)
    """
    labels = labels_cfg or load_labels()
    outcome_cols = list(labels.outcome_items.keys())

    # Check if all outcome columns exist
    present_outcomes = [c for c in outcome_cols if c in df.columns]

    if len(present_outcomes) < len(outcome_cols):
        if mode == "training":
            raise ValueError(
                f"Training mode requires all outcome items {outcome_cols}, "
                f"but only found: {present_outcomes}"
            )
        else:
            log = RuleLog(
                rule_id="R09_OUTCOME_PROCESSING",
                rule_name="Outcome Processing & Target Calculation",
                rows_affected=0,
                columns_affected=present_outcomes,
                action="skipped_unlabelled_scoring",
                details=(
                    "Unlabelled dataset in scoring mode; skipping target derivation."
                ),
            )
            return df, set(), None, log

    out_df = df.copy()

    # Calculate row-level mean of outcome items
    # Check for missing outcome items
    missing_outcome_mask = out_df[outcome_cols].isna().any(axis=1)
    missing_outcome_indices = set(out_df.index[missing_outcome_mask])

    # Compute continuous attrition intent score: ((mean(A1,A2,A3) - 1) / 4) * 100
    outcome_mean = out_df[outcome_cols].mean(axis=1)
    scores = ((outcome_mean - 1.0) / 4.0) * 100.0
    scores = scores.round(2)

    # Compute binary classification target: score >= threshold
    at_risk = (scores >= threshold).astype(float)
    # If outcome was missing, target is NaN
    at_risk[list(missing_outcome_indices)] = np.nan
    scores[list(missing_outcome_indices)] = np.nan

    out_df["attrition_intent_score"] = scores
    out_df["at_risk"] = at_risk

    rows_affected = len(missing_outcome_indices)

    if mode == "training":
        filtered_df = out_df.drop(index=list(missing_outcome_indices)).copy()
        # Convert at_risk to integer
        filtered_df["at_risk"] = filtered_df["at_risk"].astype(int)
    else:
        filtered_df = out_df.copy()

    # Compute class balance statistics on valid labelled rows
    valid_at_risk = filtered_df["at_risk"].dropna()
    class_balance = None

    if len(valid_at_risk) > 0:
        total = len(valid_at_risk)
        pos = int((valid_at_risk == 1).sum())
        neg = int((valid_at_risk == 0).sum())
        class_balance = {
            "total_labelled": total,
            "at_risk_count": pos,
            "at_risk_pct": round(pos / total * 100, 2),
            "not_at_risk_count": neg,
            "not_at_risk_pct": round(neg / total * 100, 2),
            "threshold": threshold,
        }

    log = RuleLog(
        rule_id="R09_OUTCOME_PROCESSING",
        rule_name="Outcome Processing & Target Derivation",
        rows_affected=rows_affected,
        columns_affected=outcome_cols + ["attrition_intent_score", "at_risk"],
        action="compute_targets_and_filter"
        if mode == "training"
        else "compute_targets_optional",
        details=(
            f"Derived attrition_intent_score and at_risk (threshold: {threshold}). "
            f"Filtered {rows_affected} incomplete outcome rows in training."
        ),
    )

    return filtered_df, missing_outcome_indices, class_balance, log
