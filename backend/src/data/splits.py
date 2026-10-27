"""
Leakage-Safe Grouped and Temporal Data Splitting.
Enforces non-negotiable rule #2: Grouped by student_id, stratified on attrition status.
Zero student overlap across train, validation, and test partitions.

Component: Academic Intelligence Framework (IT23164680)
"""

from typing import Dict, List, Set, Tuple, Any
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

logger = logging.getLogger("AcademicIntelligence.Splits")


def assert_zero_student_overlap(
    train_students: List[str],
    val_students: List[str],
    test_students: List[str]
) -> None:
    """
    Formally verifies that no student_id appears in more than one partition.
    Guarantees strict zero-leakage across train, val, and test splits.
    """
    s_train = set(train_students)
    s_val = set(val_students)
    s_test = set(test_students)

    tv_overlap = s_train.intersection(s_val)
    tt_overlap = s_train.intersection(s_test)
    vt_overlap = s_val.intersection(s_test)

    if tv_overlap:
        raise ValueError(f"LEAKAGE DETECTED: {len(tv_overlap)} students overlap between Train and Val!")
    if tt_overlap:
        raise ValueError(f"LEAKAGE DETECTED: {len(tt_overlap)} students overlap between Train and Test!")
    if vt_overlap:
        raise ValueError(f"LEAKAGE DETECTED: {len(vt_overlap)} students overlap between Val and Test!")

    logger.info(
        f"Split Leakage Verification PASSED: {len(s_train)} train, "
        f"{len(s_val)} val, {len(s_test)} test students. Exact zero overlap."
    )


def create_stratified_group_splits(
    df_outcomes: pd.DataFrame,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    n_cv_folds: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Partitions student cohort into 70/15/15 train/val/test splits,
    grouped by student_id and stratified on attrition outcome.
    Also creates a 5-fold grouped stratified CV on train+val.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    students_df = df_outcomes[["student_id", "attrited"]].drop_duplicates().reset_index(drop=True)

    sids = students_df["student_id"].values
    labels = students_df["attrited"].values

    # Step 1: 70% Train, 30% Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_sids, temp_sids, y_train, y_temp = train_test_split(
        sids, labels,
        test_size=temp_ratio,
        stratify=labels,
        random_state=seed
    )

    # Step 2: Split Temp evenly into Val (15%) and Test (15%)
    val_weight_in_temp = val_ratio / temp_ratio  # 0.50
    val_sids, test_sids, y_val, y_test = train_test_split(
        temp_sids, y_temp,
        test_size=(1.0 - val_weight_in_temp),
        stratify=y_temp,
        random_state=seed
    )

    train_list = sorted(train_sids.tolist())
    val_list = sorted(val_sids.tolist())
    test_list = sorted(test_sids.tolist())

    # Formally verify zero overlap
    assert_zero_student_overlap(train_list, val_list, test_list)

    # Step 3: Create 5-fold stratified CV folds on train+val students
    train_val_sids = np.array(train_list + val_list)
    # Get labels for train_val
    outcomes_map = dict(zip(students_df["student_id"], students_df["attrited"]))
    train_val_labels = np.array([outcomes_map[s] for s in train_val_sids])

    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
    cv_folds: List[Dict[str, List[str]]] = []
    for fold_idx, (trn_idx, vld_idx) in enumerate(skf.split(train_val_sids, train_val_labels)):
        cv_folds.append({
            "fold": fold_idx + 1,
            "train_students": sorted(train_val_sids[trn_idx].tolist()),
            "val_students": sorted(train_val_sids[vld_idx].tolist())
        })

    # Save split JSONs
    with open(output_dir / "train_students.json", "w", encoding="utf-8") as f:
        json.dump(train_list, f, indent=2)
    with open(output_dir / "val_students.json", "w", encoding="utf-8") as f:
        json.dump(val_list, f, indent=2)
    with open(output_dir / "test_students.json", "w", encoding="utf-8") as f:
        json.dump(test_list, f, indent=2)
    with open(output_dir / "cv_5folds.json", "w", encoding="utf-8") as f:
        json.dump(cv_folds, f, indent=2)

    stats = {
        "total_students": len(students_df),
        "train_students": len(train_list),
        "val_students": len(val_list),
        "test_students": len(test_list),
        "train_attrition_rate": float(np.mean([outcomes_map[s] for s in train_list])),
        "val_attrition_rate": float(np.mean([outcomes_map[s] for s in val_list])),
        "test_attrition_rate": float(np.mean([outcomes_map[s] for s in test_list])),
        "n_cv_folds": n_cv_folds
    }

    logger.info(
        f"Splits created: Train={stats['train_students']} ({stats['train_attrition_rate']:.1%}), "
        f"Val={stats['val_students']} ({stats['val_attrition_rate']:.1%}), "
        f"Test={stats['test_students']} ({stats['test_attrition_rate']:.1%})"
    )

    return stats


def create_temporal_split(
    df_outcomes: pd.DataFrame,
    output_dir: Path,
    train_years: Tuple[int, ...] = (2021, 2022, 2023),
    test_year: int = 2024
) -> Dict[str, Any]:
    """
    Creates temporal out-of-time evaluation split:
    Train on historical matriculation cohorts (e.g. 2021-2023), test on future cohort (2024).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = df_outcomes[df_outcomes["cohort_year"].isin(train_years)]
    test_df = df_outcomes[df_outcomes["cohort_year"] == test_year]

    train_sids = sorted(train_df["student_id"].unique().tolist())
    test_sids = sorted(test_df["student_id"].unique().tolist())

    # Verify zero overlap
    overlap = set(train_sids).intersection(set(test_sids))
    if overlap:
        raise ValueError(f"Temporal overlap detected: {len(overlap)} students appear in both historical and future cohorts!")

    with open(output_dir / "temporal_train_students.json", "w", encoding="utf-8") as f:
        json.dump(train_sids, f, indent=2)
    with open(output_dir / "temporal_test_students.json", "w", encoding="utf-8") as f:
        json.dump(test_sids, f, indent=2)

    stats = {
        "train_cohort_years": list(train_years),
        "test_cohort_year": test_year,
        "temporal_train_students": len(train_sids),
        "temporal_test_students": len(test_sids),
        "temporal_train_attrition_rate": float(train_df["attrited"].mean()),
        "temporal_test_attrition_rate": float(test_df["attrited"].mean())
    }

    logger.info(
        f"Temporal split created: Historical ({train_years})={stats['temporal_train_students']} students "
        f"({stats['temporal_train_attrition_rate']:.1%}), "
        f"Future ({test_year})={stats['temporal_test_students']} students "
        f"({stats['temporal_test_attrition_rate']:.1%})"
    )
    return stats
