"""
Dataset and Payload Validator for Academic Intelligence Framework.
Validates uploaded tabular files (CSV/Excel) against data/schema/schema.json.
Supports auto-detection and conversion of wide format to canonical long format.

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


REQUIRED_LONG_COLUMNS = [
    "student_id",
    "semester",
    "gpa",
    "exam_marks",
    "attendance_pct",
    "assignment_delay_days",
    "late_assignments_count",
]

NUMERIC_RANGES = {
    "gpa": (0.0, 4.0),
    "exam_marks": (0.0, 100.0),
    "attendance_pct": (0.0, 100.0),
    "assignment_delay_days": (0.0, 45.0),
    "late_assignments_count": (0, 15),
}

STUDENT_ID_REGEX = re.compile(r"^[A-Za-z0-9_-]+$")

# Synonyms mapping to canonical long columns for flexible ingestion
COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "student_id": ["student_id", "studentid", "participant_id", "participantid", "id", "student"],
    "semester": ["semester", "sem", "term", "academic_semester"],
    "gpa": ["gpa", "sgpa", "semester_gpa", "term_gpa"],
    "exam_marks": ["exam_marks", "exam_average", "exam_avg", "exam_score", "exam_marks_avg", "exam_mark", "exams"],
    "attendance_pct": ["attendance_pct", "attendance_percentage", "attendance_rate", "attendance", "attendance_percent"],
    "assignment_delay_days": [
        "assignment_delay_days", "assignment_submission_delay_avg_days", "delay_days", "assignment_delay",
        "avg_assignment_delay", "assignment_submission_delay_days", "delay_mean"
    ],
    "late_assignments_count": [
        "late_assignments_count", "late_assignment_count", "late_count", "late_assignments", "late_submissions_count"
    ],
}

# Synonyms for outcome & cohort metadata columns in retraining datasets
OUTCOME_SYNONYMS: Dict[str, List[str]] = {
    "attrited": ["attrited", "attrition", "dropout", "dropped_out", "withdrawn", "label", "target", "is_attrited"],
    "attrition_semester": ["attrition_semester", "drop_semester", "dropout_semester", "attrited_semester"],
    "cohort_year": ["cohort_year", "cohort", "year_of_entry", "admission_year", "intake_year"],
}


def normalize_column_name(col: str) -> str:
    """Normalize raw column names to canonical schema keys using synonym matching."""
    cleaned = re.sub(r"[^a-z0-9]", "_", str(col).strip().lower()).strip("_")
    for canonical, syns in COLUMN_SYNONYMS.items():
        if cleaned == canonical or cleaned in syns:
            return canonical
    for canonical, syns in OUTCOME_SYNONYMS.items():
        if cleaned == canonical or cleaned in syns:
            return canonical
    return cleaned


# Patterns for detecting wide format columns:
# e.g., gpa_sem1, gpa_1, sem1_gpa, gpa1
WIDE_PATTERNS = [
    re.compile(r"^(gpa|exam_marks|attendance_pct|assignment_delay_days|late_assignments_count)[_]?(?:sem)?([1-8])$", re.IGNORECASE),
    re.compile(r"^(?:sem)?([1-8])[_]?(gpa|exam_marks|attendance_pct|assignment_delay_days|late_assignments_count)$", re.IGNORECASE),
]



def is_wide_format(df: pd.DataFrame) -> bool:
    """Detects whether the uploaded dataframe represents a wide-format academic table."""
    cols = [str(c).strip() for c in df.columns]
    if "semester" in [c.lower() for c in cols]:
        return False
    
    wide_col_count = 0
    for col in cols:
        for pattern in WIDE_PATTERNS:
            if pattern.match(col):
                wide_col_count += 1
                break
    return wide_col_count >= 2


def convert_wide_to_long(df_wide: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Converts wide-format student observations into canonical long format.
    Returns (df_long, conversion_notes).
    """
    notes: List[str] = []
    
    # Identify student id column
    id_col = None
    for c in df_wide.columns:
        if str(c).strip().lower() in ["student_id", "participant_id", "id", "studentid"]:
            id_col = c
            break
            
    if id_col is None:
        raise ValueError("Wide-format table must include a 'student_id' column.")

    # Parse columns into (metric, semester)
    col_mapping: Dict[int, Dict[str, str]] = {}
    for col in df_wide.columns:
        if col == id_col:
            continue
        col_clean = str(col).strip()
        matched = False
        for pattern in WIDE_PATTERNS:
            m = pattern.match(col_clean)
            if m:
                g1, g2 = m.group(1), m.group(2)
                if g1.isdigit():
                    sem = int(g1)
                    metric = g2.lower()
                else:
                    sem = int(g2)
                    metric = g1.lower()
                
                if sem not in col_mapping:
                    col_mapping[sem] = {}
                col_mapping[sem][metric] = col
                matched = True
                break

    if not col_mapping:
        raise ValueError("Could not identify any valid wide-format semester columns (e.g. gpa_sem1, attendance_pct_sem1).")

    # Identify any student-level metadata columns (e.g. outcomes/cohort)
    extra_meta_cols: Dict[str, str] = {}
    for c in df_wide.columns:
        norm = normalize_column_name(c)
        if norm in OUTCOME_SYNONYMS and c != id_col:
            extra_meta_cols[norm] = c

    rows: List[Dict[str, Any]] = []
    for _, row in df_wide.iterrows():
        sid = str(row[id_col]).strip() if pd.notna(row[id_col]) else None
        if not sid:
            continue
            
        for sem in sorted(col_mapping.keys()):
            metrics_dict = col_mapping[sem]
            # Check if student has any data for this semester
            has_data = any(pd.notna(row.get(orig_col)) for orig_col in metrics_dict.values())
            if not has_data:
                continue
                
            long_row: Dict[str, Any] = {
                "student_id": sid,
                "semester": sem,
                "gpa": row.get(metrics_dict.get("gpa")),
                "exam_marks": row.get(metrics_dict.get("exam_marks")),
                "attendance_pct": row.get(metrics_dict.get("attendance_pct")),
                "assignment_delay_days": row.get(metrics_dict.get("assignment_delay_days")),
                "late_assignments_count": row.get(metrics_dict.get("late_assignments_count")),
            }
            for norm_key, orig_col in extra_meta_cols.items():
                long_row[norm_key] = row.get(orig_col)
            rows.append(long_row)

    df_long = pd.DataFrame(rows)
    notes.append(f"Successfully converted wide format table with {len(df_wide)} students into {len(df_long)} longitudinal semester rows.")
    return df_long, notes


def validate_academic_dataframe(
    df: pd.DataFrame
) -> Tuple[bool, List[Dict[str, Any]], Optional[pd.DataFrame], List[str]]:
    """
    Validates an academic dataframe against /data/schema/schema.json requirements.
    
    Returns:
      (is_valid, error_list, cleaned_dataframe, warning_list)
      
    Where error_list items contain:
      {"row": int, "column": str, "error_type": str, "message": str}
    """
    errors: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if df.empty:
        errors.append({
            "row": None,
            "column": None,
            "error_type": "EMPTY_FILE",
            "message": "The uploaded file contains no data rows."
        })
        return False, errors, None, warnings

    # Check format: Wide vs Long
    is_wide = is_wide_format(df)
    if is_wide:
        try:
            df, conv_notes = convert_wide_to_long(df)
            warnings.extend(conv_notes)
        except Exception as e:
            errors.append({
                "row": None,
                "column": None,
                "error_type": "WIDE_FORMAT_CONVERSION_ERROR",
                "message": f"Failed to parse wide-format table: {str(e)}"
            })
            return False, errors, None, warnings

    # Normalize column names (strip whitespace, lowercase, and resolve synonyms)
    col_map = {c: normalize_column_name(c) for c in df.columns}
    df = df.rename(columns=col_map)

    # Check required columns
    missing_cols = [c for c in REQUIRED_LONG_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append({
            "row": None,
            "column": ", ".join(missing_cols),
            "error_type": "MISSING_COLUMNS",
            "message": (
                f"Missing required columns: {missing_cols}. "
                f"Expected long format columns: {REQUIRED_LONG_COLUMNS}."
            )
        })
        return False, errors, None, warnings

    cleaned_rows: List[Dict[str, Any]] = []
    seen_student_semesters = set()
    student_semester_counts: Dict[str, List[int]] = {}

    for idx, row in df.iterrows():
        # Display 1-indexed row number matching Excel / CSV lines (header is row 1, first data row is 2)
        row_num = int(idx) + 2

        # 1. student_id validation
        raw_sid = row["student_id"]
        if pd.isna(raw_sid) or str(raw_sid).strip() == "":
            errors.append({
                "row": row_num,
                "column": "student_id",
                "error_type": "NULL_STUDENT_ID",
                "message": f"Row {row_num}: 'student_id' cannot be empty or null."
            })
            continue

        sid = str(raw_sid).strip()
        if not STUDENT_ID_REGEX.match(sid):
            errors.append({
                "row": row_num,
                "column": "student_id",
                "error_type": "INVALID_STUDENT_ID_FORMAT",
                "message": f"Row {row_num}: 'student_id' ('{sid}') contains invalid characters. Must be alphanumeric, hyphen, or underscore."
            })
            continue

        # 2. semester validation
        raw_sem = row["semester"]
        if pd.isna(raw_sem):
            errors.append({
                "row": row_num,
                "column": "semester",
                "error_type": "NULL_SEMESTER",
                "message": f"Row {row_num}: 'semester' cannot be empty or null."
            })
            continue

        try:
            sem_val = float(raw_sem)
            if not sem_val.is_integer() or not (1 <= int(sem_val) <= 8):
                errors.append({
                    "row": row_num,
                    "column": "semester",
                    "error_type": "INVALID_SEMESTER_RANGE",
                    "message": f"Row {row_num}: 'semester' must be an integer between 1 and 8 (received {raw_sem})."
                })
                continue
            semester = int(sem_val)
        except (ValueError, TypeError):
            errors.append({
                "row": row_num,
                "column": "semester",
                "error_type": "NON_NUMERIC_SEMESTER",
                "message": f"Row {row_num}: 'semester' must be a numeric integer between 1 and 8 (received '{raw_sem}')."
            })
            continue

        # 3. Duplicate check for (student_id, semester)
        key = (sid, semester)
        if key in seen_student_semesters:
            errors.append({
                "row": row_num,
                "column": "semester",
                "error_type": "DUPLICATE_OBSERVATION",
                "message": f"Row {row_num}: Duplicate semester {semester} observation for student '{sid}'."
            })
            continue
        seen_student_semesters.add(key)

        if sid not in student_semester_counts:
            student_semester_counts[sid] = []
        student_semester_counts[sid].append(semester)

        # 4. Numeric metric range checks
        valid_row = True
        cleaned_metrics: Dict[str, Any] = {"student_id": sid, "semester": semester}

        for col, (low, high) in NUMERIC_RANGES.items():
            val = row[col]
            if pd.isna(val):
                cleaned_metrics[col] = np.nan
                continue

            try:
                num_val = float(val)
                if np.isnan(num_val):
                    cleaned_metrics[col] = np.nan
                elif not (low <= num_val <= high):
                    errors.append({
                        "row": row_num,
                        "column": col,
                        "error_type": "OUT_OF_BOUNDS",
                        "message": f"Row {row_num}: '{col}' value {num_val} is outside valid range [{low}, {high}]."
                    })
                    valid_row = False
                else:
                    if col == "late_assignments_count":
                        cleaned_metrics[col] = int(num_val)
                    else:
                        cleaned_metrics[col] = num_val
            except (ValueError, TypeError):
                errors.append({
                    "row": row_num,
                    "column": col,
                    "error_type": "TYPE_ERROR",
                    "message": f"Row {row_num}: '{col}' must be a numeric value (received '{val}')."
                })
                valid_row = False

        if valid_row:
            cleaned_rows.append(cleaned_metrics)

    # 5. Check student history warnings (minimum semesters rule)
    for sid, sems in student_semester_counts.items():
        if len(sems) == 1:
            warnings.append(
                f"Student '{sid}' has only 1 semester of observations (Semester {sems[0]}). "
                "Minimum Semesters Rule: Trajectory slopes and longitudinal velocity features default to 0.0 "
                "until at least 2 consecutive semesters are recorded."
            )
        sorted_sems = sorted(sems)
        expected_range = list(range(sorted_sems[0], sorted_sems[-1] + 1))
        if sorted_sems != expected_range:
            missing_sems = [s for s in expected_range if s not in sorted_sems]
            warnings.append(
                f"Student '{sid}' has gap(s) in semester sequence. Missing semesters: {missing_sems}. "
                "Temporal forward-fill imputation will be applied across gaps."
            )

    if errors:
        return False, errors, None, warnings

    cleaned_df = pd.DataFrame(cleaned_rows).sort_values(["student_id", "semester"]).reset_index(drop=True)
    return True, [], cleaned_df, warnings
