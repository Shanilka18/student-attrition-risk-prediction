"""
Prediction Task Definition: Early Longitudinal Attrition Forecasting.
Constructs prefix observation windows (t in [2, 3, 4, 5, 6]) and labels whether
a student will attrite within the next 2 semesters (t+1 or t+2).

Component: Academic Intelligence Framework (IT23164680)
"""

from typing import List, Dict, Any, Tuple
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("AcademicIntelligence.Task")


def create_prefix_task_samples(
    df_long: pd.DataFrame,
    df_outcomes: pd.DataFrame,
    min_window: int = 2,
    max_window: int = 6
) -> pd.DataFrame:
    """
    Constructs prefix prediction samples for each student across available observation horizons.

    Input Window: Observations up to semester t (where min_window <= t <= max_window).
    Binary Target: Will the student attrite within the subsequent 2 semesters (s in [t+1, t+2])?
    Grouping: Retains student_id for strict grouped train/val/test splitting.
    """
    # Merge outcomes metadata to check student trajectory bounds
    outcomes_map = df_outcomes.set_index("student_id").to_dict("index")

    samples: List[Dict[str, Any]] = []

    # Group by student
    student_groups = df_long.groupby("student_id")

    for sid, group in student_groups:
        student_outcome = outcomes_map.get(sid)
        if not student_outcome:
            continue

        attrited = student_outcome["attrited"] == 1
        attrition_sem = student_outcome["attrition_semester"]
        last_sem = student_outcome["last_semester_observed"]

        available_sems = set(group["semester"].unique())

        for t in range(min_window, max_window + 1):
            # A student can only be evaluated at horizon t if they were still active up to semester t
            if t > last_sem or t not in available_sems:
                continue

            # Target definition: Attrition occurs at t+1 or t+2
            if attrited and attrition_sem is not None and (t + 1 <= attrition_sem <= t + 2):
                target = 1
            else:
                target = 0

            samples.append({
                "sample_id": f"{sid}_t{t}",
                "student_id": sid,
                "cohort_year": student_outcome.get("cohort_year", 2021),
                "window_t": t,
                "target_attrite_next_2_sems": target,
                "overall_attrited": 1 if attrited else 0,
                "attrition_semester": attrition_sem
            })

    df_samples = pd.DataFrame(samples)
    pos_rate = df_samples["target_attrite_next_2_sems"].mean() if len(df_samples) > 0 else 0.0
    logger.info(
        f"Generated {len(df_samples)} prefix samples across {df_samples['student_id'].nunique()} students. "
        f"Window range: [{min_window}, {max_window}]. Positive rate (next-2-sems): {pos_rate:.1%}"
    )

    return df_samples
