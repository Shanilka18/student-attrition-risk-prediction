"""
Dual-Paradigm Feature Builder: Sequence Tensors & Tabular Longitudinal Features.
Constructs aligned representations for both sequence-aware deep models (PyTorch)
and gradient boosting tree models (XGBoost/LightGBM), plus static snapshot comparator.

Component: Academic Intelligence Framework (IT23164680)
"""

from typing import Dict, List, Tuple, Any, Optional
import logging
import pandas as pd
import numpy as np
import torch
from backend.src.features.preprocessor import AcademicPreprocessor

logger = logging.getLogger("AcademicIntelligence.FeatureBuilder")

FEATURE_COLUMNS_FOR_SEQ = [
    "gpa",
    "exam_marks",
    "attendance_pct",
    "assignment_delay_days"
]


def _compute_ols_slope(y_vals: np.ndarray) -> float:
    """Computes simple linear slope beta_1 over integer time indices [1..k]."""
    k = len(y_vals)
    if k <= 1:
        return 0.0
    x_vals = np.arange(1, k + 1, dtype=float)
    x_mean = np.mean(x_vals)
    y_mean = np.mean(y_vals)
    numerator = np.sum((x_vals - x_mean) * (y_vals - y_mean))
    denominator = np.sum((x_vals - x_mean) ** 2)
    return float(numerator / denominator) if denominator > 1e-9 else 0.0


def _count_consecutive_declines(y_vals: np.ndarray) -> int:
    """Counts consecutive decreases in y leading directly up to the final element."""
    k = len(y_vals)
    if k <= 1:
        return 0
    count = 0
    for idx in range(k - 1, 0, -1):
        if y_vals[idx] < y_vals[idx - 1]:
            count += 1
        else:
            break
    return count


def build_tabular_engineered_features(
    df_clean_long: pd.DataFrame,
    df_task_samples: pd.DataFrame
) -> pd.DataFrame:
    """
    Constructs hand-engineered longitudinal trend features for Gradient Boosting.
    Strictly observes data up to horizon t for each sample (zero lookahead).
    """
    logger.info(f"Extracting engineered trend features for {len(df_task_samples)} prefix samples...")

    # Index clean records by (student_id, semester)
    records_by_student = {
        sid: group.sort_values("semester")
        for sid, group in df_clean_long.groupby("student_id")
    }

    tabular_rows: List[Dict[str, Any]] = []

    for _, row in df_task_samples.iterrows():
        sid = row["student_id"]
        t = int(row["window_t"])
        sample_id = row["sample_id"]
        target = int(row["target_attrite_next_2_sems"])

        s_records = records_by_student.get(sid)
        if s_records is None:
            continue

        # Strictly filter observations up to horizon t
        sub = s_records[s_records["semester"] <= t]
        if len(sub) == 0:
            continue

        gpa_series = sub["gpa"].values
        exam_series = sub["exam_marks"].values
        att_series = sub["attendance_pct"].values
        delay_series = sub["assignment_delay_days"].values
        late_series = sub["late_assignments_count"].values

        # 1. Last semester (window t) instantaneous values
        last_gpa = float(gpa_series[-1])
        last_exam = float(exam_series[-1])
        last_att = float(att_series[-1])
        last_delay = float(delay_series[-1])
        last_late = int(late_series[-1])

        # 2. Longitudinal Trend & Velocity Features
        gpa_slope = _compute_ols_slope(gpa_series)
        exam_slope = _compute_ols_slope(exam_series)
        att_slope = _compute_ols_slope(att_series)
        delay_trend = _compute_ols_slope(delay_series)

        # 3. Aggregations & Volatility
        gpa_mean = float(np.mean(gpa_series))
        gpa_volatility = float(np.std(gpa_series))
        exam_mean = float(np.mean(exam_series))
        att_mean = float(np.mean(att_series))
        att_variance = float(np.var(att_series))
        delay_mean = float(np.mean(delay_series))
        delay_max = float(np.max(delay_series))
        late_total = int(np.sum(late_series))

        # 4. Critical Event / Inflection Features
        # Consecutive GPA declines
        consec_declines = _count_consecutive_declines(gpa_series)

        # Max consecutive drop between any adjacent semesters
        adjacent_drops = [max(0.0, gpa_series[idx - 1] - gpa_series[idx]) for idx in range(1, len(gpa_series))]
        gpa_max_drop = float(max(adjacent_drops)) if adjacent_drops else 0.0

        # Semesters below warning thresholds
        sems_below_gpa_2 = int(np.sum(gpa_series < 2.0))
        sems_below_att_65 = int(np.sum(att_series < 65.0))

        tabular_rows.append({
            "sample_id": sample_id,
            "student_id": sid,
            "window_t": t,
            "target": target,
            # Instantaneous last-semester signals
            "last_gpa": round(last_gpa, 2),
            "last_exam_marks": round(last_exam, 1),
            "last_attendance_pct": round(last_att, 1),
            "last_assignment_delay_days": round(last_delay, 1),
            "last_late_assignments_count": last_late,
            # Longitudinal trend slopes (OLS velocity)
            "gpa_slope": round(gpa_slope, 4),
            "exam_mark_slope": round(exam_slope, 4),
            "attendance_slope": round(att_slope, 4),
            "delay_trend": round(delay_trend, 4),
            # Aggregations & Dispersion
            "gpa_rolling_mean": round(gpa_mean, 2),
            "gpa_volatility": round(gpa_volatility, 4),
            "cumulative_gpa": round(gpa_mean, 2),
            "exam_mark_mean": round(exam_mean, 1),
            "attendance_mean": round(att_mean, 1),
            "attendance_variance": round(att_variance, 2),
            "delay_mean": round(delay_mean, 1),
            "delay_max": round(delay_max, 1),
            "late_count_total": late_total,
            # Inflection & Academic Risk Counters
            "consecutive_declines": consec_declines,
            "gpa_max_drop": round(gpa_max_drop, 2),
            "sems_below_gpa_2": sems_below_gpa_2,
            "sems_below_att_65": sems_below_att_65
        })

    df_tab = pd.DataFrame(tabular_rows)
    logger.info(f"Engineered tabular matrix constructed: {df_tab.shape[0]} rows, {df_tab.shape[1]} columns")
    return df_tab


def build_static_snapshot_features(df_tabular: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the naive static baseline features (last GPA and last attendance rate only),
    representing the conventional point-in-time early warning mechanisms.
    """
    static_cols = ["sample_id", "student_id", "window_t", "target", "last_gpa", "last_attendance_pct"]
    return df_tabular[static_cols].copy()


def build_sequence_tensors(
    df_clean_long: pd.DataFrame,
    df_task_samples: pd.DataFrame,
    preprocessor: AcademicPreprocessor,
    max_window: int = 6
) -> Dict[str, Any]:
    """
    Constructs 3D PyTorch tensors and padding masks for recurrent models (LSTM / GRU).
    Shape:
        tensors: (N_samples, max_window, num_features=4)
        padding_mask: (N_samples, max_window) [True where real data exists, False where padded]
        seq_lengths: (N_samples,) [int length <= max_window]
        sample_ids: (N_samples,)
        student_ids: (N_samples,)
        targets: (N_samples,)
    """
    logger.info(f"Building sequence tensors for {len(df_task_samples)} prefix samples...")

    # Normalize clean features using preprocessor
    df_norm = df_clean_long.copy()
    for col in FEATURE_COLUMNS_FOR_SEQ:
        df_norm[col] = preprocessor.normalize_feature(df_norm[col], col)

    records_by_student = {
        sid: group.sort_values("semester")
        for sid, group in df_norm.groupby("student_id")
    }

    n_samples = len(df_task_samples)
    n_features = len(FEATURE_COLUMNS_FOR_SEQ)

    # Initialize padded tensors with zeros
    tensor_data = np.zeros((n_samples, max_window, n_features), dtype=np.float32)
    padding_mask = np.zeros((n_samples, max_window), dtype=bool)
    seq_lengths = np.zeros((n_samples,), dtype=np.int64)
    targets = np.zeros((n_samples,), dtype=np.int64)
    sample_ids: List[str] = []
    student_ids: List[str] = []

    for idx, (_, row) in enumerate(df_task_samples.iterrows()):
        sid = row["student_id"]
        t = int(row["window_t"])
        sample_id = row["sample_id"]
        target = int(row["target_attrite_next_2_sems"])

        sample_ids.append(sample_id)
        student_ids.append(sid)
        targets[idx] = target

        s_records = records_by_student.get(sid)
        if s_records is None:
            continue

        sub = s_records[s_records["semester"] <= t]
        actual_len = min(len(sub), max_window)
        seq_lengths[idx] = actual_len

        if actual_len > 0:
            feat_vals = sub[FEATURE_COLUMNS_FOR_SEQ].values[:actual_len]
            tensor_data[idx, :actual_len, :] = feat_vals
            padding_mask[idx, :actual_len] = True

    payload = {
        "tensors": torch.tensor(tensor_data, dtype=torch.float32),
        "padding_mask": torch.tensor(padding_mask, dtype=torch.bool),
        "seq_lengths": torch.tensor(seq_lengths, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "sample_ids": sample_ids,
        "student_ids": student_ids,
        "feature_names": FEATURE_COLUMNS_FOR_SEQ
    }

    logger.info(
        f"Sequence tensors constructed: shape={payload['tensors'].shape}, "
        f"padding_mask={payload['padding_mask'].shape}, targets={payload['targets'].shape}"
    )
    return payload
