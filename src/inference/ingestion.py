"""Inference Ingestion Engine with Schema Alignment and Strict PII Stripping.

Handles:
- CSV, XLSX, XLS file ingestion with automatic header row detection.
- Case- and space-insensitive column mapping with synonyms.
- Strict PII stripping (names, NIC, phone numbers, emails, student IDs).
- Automated pseudonymous ID generation when participant_id is absent.
- Schema alignment report detailing matched, missing, ignored, and coerced columns.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("inference_ingestion")

# Forbidden direct identifiers that must be dropped immediately with a security warning
PII_COLUMNS = [
    "name", "full_name", "first_name", "last_name", "student_name",
    "nic", "national_id", "id_number", "passport",
    "phone", "mobile", "contact_no", "telephone",
    "email", "email_address", "student_email",
    "student_number", "student_id", "reg_no", "registration_no", "index_no", "admission_no",
]

# Canonical feature names and their common synonyms for survey ingestion
COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "participant_id": ["participant_id", "participant", "id", "student_token", "hash_id", "record_id"],
    "household_income_bracket": [
        "household_income_bracket", "income_bracket", "monthly_income", "household_income",
        "family_income", "income", "monthly_household_income", "income_level",
    ],
    "financial_stress": ["financial_stress", "financial_pressure", "financial_difficulty", "economic_stress"],
    "scholarship_bursary": [
        "scholarship_bursary", "scholarship", "bursary", "mahapola", "grant", "financial_aid",
    ],
    "dependents_count": ["dependents_count", "dependents", "family_dependents", "num_dependents", "children_dependents"],
    "housing_type": ["housing_type", "living_situation", "accommodation", "residence_type", "housing"],
    "commute_time_min": [
        "commute_time_min", "commute_time", "travel_time", "commute_minutes", "travel_time_min",
        "commute_duration", "daily_commute",
    ],
    "commute_cost_lkr_week": [
        "commute_cost_lkr_week", "commute_cost", "weekly_commute_cost", "travel_cost", "transit_fare",
    ],
    "internet_reliability": ["internet_reliability", "internet_quality", "connection_stability", "wifi_reliability"],
    "internet_affordability": ["internet_affordability", "data_cost_affordability", "internet_expense"],
    "device_type": ["device_type", "primary_device", "device_access", "computer_type"],
    "device_quality": ["device_quality", "laptop_condition", "device_rating"],
    "quiet_study_space": ["quiet_study_space", "study_space", "quiet_room", "study_environment"],
    "part_time_employment": ["part_time_employment", "employed", "has_job", "working", "job_status"],
    "work_hours_per_week": ["work_hours_per_week", "work_hours", "job_hours", "hours_worked"],
    "faculty": ["faculty", "department", "academic_faculty", "school"],
    "year_of_study": ["year_of_study", "academic_year", "study_year", "year"],
    "district_type": ["district_type", "home_location", "residence_zone", "origin_district", "locality"],
    "baseline_gpa_band": ["baseline_gpa_band", "gpa_band", "baseline_gpa", "prior_gpa", "gpa_range"],
    "baseline_engagement_score": ["baseline_engagement_score", "engagement_score", "prior_engagement", "lms_engagement"],
    "gender": ["gender", "sex"],
    "attrition": ["attrition", "dropout", "withdrawn", "status_attrition"],
}


def _normalize_name(name: str) -> str:
    """Normalize string: lowercase, remove special characters, replace whitespace with underscore."""
    clean = re.sub(r"[^\w\s]", "", str(name).strip().lower())
    return re.sub(r"\s+", "_", clean)


def detect_and_strip_pii(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Scan and purge any direct personal identifiers. Never stores or logs PII values."""
    dropped_pii = []
    df_clean = df.copy()

    for col in list(df_clean.columns):
        norm = _normalize_name(col)
        for pii in PII_COLUMNS:
            if norm == pii or norm.startswith(pii + "_") or norm.endswith("_" + pii):
                df_clean.drop(columns=[col], inplace=True)
                dropped_pii.append(col)
                break

    if dropped_pii:
        logger.warning(
            f"SECURITY GUARDRAIL TRIGGERED: Direct personal identifier columns detected and stripped: {dropped_pii}. "
            "Under no circumstances will PII be persisted or forwarded to model inference."
        )

    return df_clean, dropped_pii


def load_raw_file(file_path: Path | str) -> pd.DataFrame:
    """Ingest CSV, XLSX, or XLS files, auto-detecting the header row."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found at: {path}")

    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            # Check first 5 rows for header row candidates
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(5)]

            header_idx = 0
            max_matches = 0
            all_synonyms = {syn for syns in COLUMN_SYNONYMS.values() for syn in syns}

            for i, line in enumerate(lines):
                tokens = [_normalize_name(t) for t in line.split(",")]
                matches = sum(1 for t in tokens if t in all_synonyms)
                if matches > max_matches:
                    max_matches = matches
                    header_idx = i

            df = pd.read_csv(path, skiprows=header_idx)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Must be .csv, .xlsx, or .xls.")

        if df.empty:
            raise ValueError("Provided dataset is empty (0 rows).")

        return df
    except (pd.errors.EmptyDataError, ValueError) as e:
        raise ValueError(f"Provided dataset is empty: {e}")


def align_dataframe_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Align raw columns against schema via synonym matching and produce schema report."""
    # 1. Purge PII
    df_safe, dropped_pii = detect_and_strip_pii(df)

    # 2. Map columns via synonym lookup
    mapping: Dict[str, str] = {}
    matched_canonical: Set[str] = set()
    ignored_columns: List[str] = []

    for col in df_safe.columns:
        norm = _normalize_name(col)
        matched = False
        for canonical, syns in COLUMN_SYNONYMS.items():
            if norm in syns:
                mapping[col] = canonical
                matched_canonical.add(canonical)
                matched = True
                break
        if not matched:
            ignored_columns.append(col)

    df_aligned = df_safe.rename(columns=mapping)

    # 3. Handle Participant ID
    if "participant_id" not in df_aligned.columns:
        logger.info("No participant_id column found. Generating pseudonymous IDs (P-00001, P-00002, ...)")
        df_aligned["participant_id"] = [f"P-{i+1:05d}" for i in range(len(df_aligned))]
    else:
        # Standardize formatting
        df_aligned["participant_id"] = df_aligned["participant_id"].astype(str)

    # 4. Identify missing mandatory questionnaire features
    core_features = [
        "household_income_bracket", "financial_stress", "scholarship_bursary", "dependents_count",
        "housing_type", "commute_time_min", "commute_cost_lkr_week", "internet_reliability",
        "internet_affordability", "device_type", "device_quality", "quiet_study_space",
        "part_time_employment", "work_hours_per_week", "faculty", "year_of_study",
        "district_type", "baseline_gpa_band", "baseline_engagement_score",
    ]
    missing_features = [f for f in core_features if f not in df_aligned.columns]

    # Categorical and string columns must have object/str dtype for sklearn encoders
    string_categorical_cols = {
        "housing_type", "device_type", "faculty", "district_type", "baseline_gpa_band", "gender",
    }

    # For missing features, initialize with appropriate NaN / None so preprocessor imputers handle them cleanly
    coerced_columns = []
    for f in missing_features:
        if f in string_categorical_cols:
            df_aligned[f] = pd.Series([None] * len(df_aligned), dtype="object")
        else:
            df_aligned[f] = pd.Series([np.nan] * len(df_aligned), dtype="float64")
        coerced_columns.append(f)

    # Ensure existing string categorical columns have object dtype even if missing
    for c in string_categorical_cols:
        if c in df_aligned.columns and df_aligned[c].isna().all():
            df_aligned[c] = df_aligned[c].astype("object")

    # 5. Build schema report
    report = {
        "total_rows_ingested": len(df_aligned),
        "columns_matched_count": len(mapping),
        "matched_features": list(mapping.values()),
        "missing_features_imputed": missing_features,
        "ignored_unknown_columns": ignored_columns,
        "dropped_pii_columns": dropped_pii,
        "has_outcome_column": bool("attrition" in df_aligned.columns),
    }

    return df_aligned, report
