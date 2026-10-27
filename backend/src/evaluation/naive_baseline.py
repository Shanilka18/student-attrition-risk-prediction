"""
Naive Baseline Evaluation: Static GPA < 2.0 Threshold Rule.
Demonstrates empirical failure of static single-point threshold early-warning systems,
motivating sequence-aware and longitudinal trend modeling.

Component: Academic Intelligence Framework (IT23164680)
"""

from typing import Dict, Any
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger("AcademicIntelligence.NaiveBaseline")


def evaluate_naive_gpa_cutoff(
    df_long: pd.DataFrame,
    df_task_samples: pd.DataFrame,
    gpa_threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Evaluates the standard institutional rule: Flag student as at-risk if GPA at window t < threshold.
    Compares against binary target: Attrition within next 2 semesters.
    """
    # Merge current semester GPA at window t
    # For each sample (student_id, window_t), lookup GPA at semester == window_t
    merged = pd.merge(
        df_task_samples,
        df_long[["student_id", "semester", "gpa", "exam_marks", "attendance_pct"]],
        left_on=["student_id", "window_t"],
        right_on=["student_id", "semester"],
        how="inner"
    )

    # Impute missing GPA with median if any
    filled_gpa = merged["gpa"].fillna(merged["gpa"].median())

    # Naive decision rule: Flag as positive (at-risk) if current GPA < threshold
    y_pred = (filled_gpa < gpa_threshold).astype(int).values
    y_true = merged["target_attrite_next_2_sems"].values

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    results = {
        "rule": f"GPA < {gpa_threshold}",
        "total_evaluated_samples": len(merged),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "motivating_finding": (
            f"The static rule 'GPA < {gpa_threshold}' misses {fn} out of {tp + fn} "
            f"approaching dropouts (Recall: {rec:.1%}), proving that single-point threshold "
            f"monitoring fails to provide timely early warning."
        )
    }

    logger.info(f"Naive Baseline ({results['rule']}): Recall={rec:.1%}, Precision={prec:.1%}, F1={f1:.4f}")
    return results
