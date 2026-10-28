"""Stages 3, 6, 7, and 8: Filters, duplicate handling, and missingness."""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from mei.cleaning.models import RuleLog


def filter_consent(
    df: pd.DataFrame,
    mode: str = "training",
    consent_col: str = "consent",
) -> Tuple[pd.DataFrame, Set[int], RuleLog]:
    """Stage 3a: Filter out submissions that did not grant informed consent.

    Returns:
        (filtered_df, non_consent_indices, RuleLog)
    """
    if consent_col not in df.columns:
        log = RuleLog(
            rule_id="R03A_CONSENT_FILTER",
            rule_name="Informed Consent Verification",
            rows_affected=0,
            columns_affected=[],
            action="skipped",
            details="Consent column not present; skipping consent filter.",
        )
        return df, set(), log

    valid_mask = pd.Series(True, index=df.index)
    for idx, val in df[consent_col].items():
        if pd.isna(val) or val is None:
            valid_mask.loc[idx] = False
            continue

        s = str(val).strip().lower()
        # Accept code 1, "yes", "i consent to participate", true
        is_yes = (
            s in ["1", "1.0", "yes", "true", "consent"]
            or "consent to participate" in s
            or "agree" in s
        )
        # Reject code 0, "no", "i do not consent", false
        is_no = (
            s in ["0", "0.0", "no", "false"] or "do not consent" in s or "disagree" in s
        )

        if is_no or not is_yes:
            valid_mask.loc[idx] = False

    non_consent_indices = set(df.index[~valid_mask])
    rows_affected = len(non_consent_indices)

    if mode == "training":
        filtered_df = df.loc[valid_mask].copy()
    else:
        # In scoring mode, rows are preserved but tracked
        filtered_df = df.copy()

    log = RuleLog(
        rule_id="R03A_CONSENT_FILTER",
        rule_name="Informed Consent Verification",
        rows_affected=rows_affected,
        columns_affected=[consent_col],
        action="drop_non_consent" if mode == "training" else "flag_non_consent",
        details=f"Identified {rows_affected} unconsented submissions.",
    )

    return filtered_df, non_consent_indices, log


def filter_attention_check(
    df: pd.DataFrame,
    mode: str = "training",
    enabled: bool = True,
    attn_col: str = "attention_check",
) -> Tuple[pd.DataFrame, Set[int], RuleLog]:
    """Stage 3b: Screen out inattentive responders based on the attention-check item."""
    if not enabled or attn_col not in df.columns:
        log = RuleLog(
            rule_id="R03B_ATTENTION_CHECK",
            rule_name="Attention Check Screening",
            rows_affected=0,
            columns_affected=[attn_col] if attn_col in df.columns else [],
            action="skipped",
            details="Attention check filter disabled or column missing.",
        )
        return df, set(), log

    passed_mask = pd.Series(False, index=df.index)
    for idx, val in df[attn_col].items():
        if pd.isna(val) or val is None:
            continue
        s = str(val).strip().lower()
        # Expected response: "Sometimes" or code 3
        if s in ["sometimes", "3", "3.0"]:
            passed_mask.loc[idx] = True

    failed_indices = set(df.index[~passed_mask])
    rows_affected = len(failed_indices)

    if mode == "training":
        filtered_df = df.loc[passed_mask].copy()
    else:
        filtered_df = df.copy()

    log = RuleLog(
        rule_id="R03B_ATTENTION_CHECK",
        rule_name="Attention Check Screening",
        rows_affected=rows_affected,
        columns_affected=[attn_col],
        action="drop_inattentive" if mode == "training" else "flag_inattentive",
        details=f"Identified {rows_affected} attention check failures.",
    )

    return filtered_df, failed_indices, log


def resolve_duplicates(
    df: pd.DataFrame,
    mode: str = "training",
    id_col: str = "student_id",
    ts_col: str = "timestamp",
) -> Tuple[pd.DataFrame, Set[int], RuleLog]:
    """Stage 6: Identify duplicates and retain the LAST valid submission per student."""
    if id_col not in df.columns:
        log = RuleLog(
            rule_id="R06_DUPLICATE_RESOLUTION",
            rule_name="Duplicate Submission Resolution",
            rows_affected=0,
            columns_affected=[],
            action="skipped",
            details="ID column missing; skipping duplicate resolution.",
        )
        return df, set(), log

    # Sort by timestamp ascending if available
    temp_df = df.copy()
    if ts_col in temp_df.columns:
        try:
            # Parse timestamp for sorting
            temp_df["_parsed_ts"] = pd.to_datetime(temp_df[ts_col], errors="coerce")
            temp_df = temp_df.sort_values(by="_parsed_ts", ascending=True).drop(
                columns=["_parsed_ts"]
            )
        except Exception:
            pass

    # Find duplicates keeping last
    duplicate_mask = temp_df.duplicated(subset=[id_col], keep="last")
    duplicate_indices = set(temp_df.index[duplicate_mask])
    rows_affected = len(duplicate_indices)

    if mode == "training":
        filtered_df = df.drop(index=list(duplicate_indices)).copy()
    else:
        filtered_df = df.copy()

    log = RuleLog(
        rule_id="R06_DUPLICATE_RESOLUTION",
        rule_name="Duplicate Submission Resolution",
        rows_affected=rows_affected,
        columns_affected=[id_col],
        action="keep_last_submission",
        details=(
            f"Found {rows_affected} duplicate submissions. "
            f"Retained the most recent submission per student ID."
        ),
    )

    return filtered_df, duplicate_indices, log


def detect_careless_responding(
    df: pd.DataFrame,
    mode: str = "training",
    drop: bool = True,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Set[int], RuleLog]:
    """Stage 7: Detect straight-liners (invariant responding across indicators).

    Criteria:
    1. Zero variance across all 28 indicator answers, OR
    2. Zero variance across >= 6 of the 7 indicators across all periods.
    """
    f_cols = feature_cols or [
        c
        for c in df.columns
        if any(
            ind in c
            for ind in [
                "resource_access",
                "study_time",
                "review_time",
                "quiz_practice",
                "assignment_engagement",
                "study_consistency",
                "participation",
            ]
        )
        and c.endswith(("_p1", "_p2", "_p3", "_p4"))
    ]

    if len(f_cols) < 7:
        log = RuleLog(
            rule_id="R07_CARELESS_RESPONDING",
            rule_name="Straight-Lining & Careless Responding",
            rows_affected=0,
            columns_affected=[],
            action="skipped",
            details="Insufficient feature columns for variance calculation.",
        )
        return df, set(), log

    min_required_items = min(14, len(f_cols))
    f_df = df[f_cols]
    valid_counts = f_df.notna().sum(axis=1)
    eligible_mask = valid_counts >= min_required_items

    # Rule 1: Zero overall variance across all answers
    overall_std = f_df.std(axis=1, ddof=0)
    rule1_mask = eligible_mask & (overall_std == 0.0)

    # Rule 2: Zero variance across >= 6 of the 7 indicators
    indicators = [
        "resource_access",
        "study_time",
        "review_time",
        "quiz_practice",
        "assignment_engagement",
        "study_consistency",
        "participation",
    ]
    zero_var_indicator_counts = np.zeros(len(df), dtype=int)
    for ind in indicators:
        ind_cols = [c for c in f_cols if ind in c]
        if ind_cols:
            ind_sub = f_df[ind_cols]
            ind_valid = ind_sub.notna().sum(axis=1) >= 3
            ind_zero_std = ind_sub.std(axis=1, ddof=0) == 0.0
            zero_var_indicator_counts += (ind_valid & ind_zero_std).astype(int)

    rule2_mask = eligible_mask & (zero_var_indicator_counts >= 6)
    straight_mask = rule1_mask | rule2_mask
    straight_liner_indices: Set[int] = set(df.index[straight_mask])

    rows_affected = len(straight_liner_indices)

    if mode == "training" and drop:
        filtered_df = df.drop(index=list(straight_liner_indices)).copy()
    else:
        filtered_df = df.copy()

    log = RuleLog(
        rule_id="R07_CARELESS_RESPONDING",
        rule_name="Straight-Lining & Careless Responding",
        rows_affected=rows_affected,
        columns_affected=f_cols,
        action="drop_straight_liners"
        if (mode == "training" and drop)
        else "flag_straight_liners",
        details=(
            f"Detected {rows_affected} straight-liners with zero response variance."
        ),
    )

    return filtered_df, straight_liner_indices, log


def apply_missingness_policy(
    df: pd.DataFrame,
    mode: str = "training",
    max_missing_pct: float = 0.25,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Set[int], RuleLog]:
    """Stage 8: Filter rows with excessive missingness and compute missingness summary.

    LEAKAGE GUARD (Global Rule G3):
    Missing values below the threshold are NOT imputed here. Imputation belongs
    inside cross-validation modeling pipelines.
    """
    f_cols = feature_cols or [
        c
        for c in df.columns
        if any(
            ind in c
            for ind in [
                "resource_access",
                "study_time",
                "review_time",
                "quiz_practice",
                "assignment_engagement",
                "study_consistency",
                "participation",
            ]
        )
        and c.endswith(("_p1", "_p2", "_p3", "_p4"))
    ]

    total_f_cols = len(f_cols)
    if total_f_cols == 0:
        summary: Dict[str, Any] = {"overall_missing_pct": 0.0}
        log = RuleLog(
            rule_id="R08_MISSINGNESS_POLICY",
            rule_name="Missingness Threshold Filter (>25%)",
            rows_affected=0,
            columns_affected=[],
            action="skipped",
            details="No feature columns found for missingness evaluation.",
        )
        return df, summary, set(), log

    row_missing_count = df[f_cols].isna().sum(axis=1)
    row_missing_pct = row_missing_count / total_f_cols

    excessive_missing_mask = row_missing_pct > max_missing_pct
    excessive_indices = set(df.index[excessive_missing_mask])
    rows_affected = len(excessive_indices)

    # Missingness summary
    col_missing = df[f_cols].isna().sum().to_dict()
    col_missing_pct = (df[f_cols].isna().mean() * 100).round(2).to_dict()

    period_missing: Dict[str, Dict[str, Any]] = {}
    for p in ["p1", "p2", "p3", "p4"]:
        p_cols = [c for c in f_cols if c.endswith(f"_{p}")]
        if p_cols:
            p_miss = int(df[p_cols].isna().sum().sum())
            p_total = len(df) * len(p_cols)
            period_missing[p.upper()] = {
                "missing_cells": p_miss,
                "total_cells": p_total,
                "missing_pct": round(p_miss / max(1, p_total) * 100, 2),
            }

    overall_missing_cells = int(df[f_cols].isna().sum().sum())
    overall_total_cells = len(df) * total_f_cols
    overall_missing_pct = round(
        overall_missing_cells / max(1, overall_total_cells) * 100, 2
    )

    summary = {
        "overall_missing_pct": overall_missing_pct,
        "overall_missing_cells": overall_missing_cells,
        "overall_total_cells": overall_total_cells,
        "column_missing_counts": col_missing,
        "column_missing_pct": col_missing_pct,
        "period_missing": period_missing,
        "rows_with_excessive_missing": rows_affected,
        "max_missing_allowed_pct": max_missing_pct * 100,
    }

    if mode == "training":
        filtered_df = df.drop(index=list(excessive_indices)).copy()
    else:
        filtered_df = df.copy()

    log = RuleLog(
        rule_id="R08_MISSINGNESS_POLICY",
        rule_name="Missingness Threshold Filter (>25%)",
        rows_affected=rows_affected,
        columns_affected=f_cols,
        action="drop_excessive_missing"
        if mode == "training"
        else "flag_excessive_missing",
        details=(
            f"Enforced max {max_missing_pct * 100:.0f}% missing feature threshold. "
            f"Identified {rows_affected} rows with excessive missingness. "
            f"Overall feature missingness: {overall_missing_pct:.2f}%."
        ),
    )

    return filtered_df, summary, excessive_indices, log
