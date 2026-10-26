"""Data contract, schema definitions, validation engine, and column mapper.

Defines:
- Required and optional schema specifications for the 15-item Psychosocial Questionnaire.
- File reader supporting .csv, .xlsx, and .xls formats.
- Structured validation report inspecting missing/extra columns, boundary violations,
  non-integers, duplicate IDs, missingness, and straight-lining (unengaged) respondents.
- Intelligent fuzzy column mapping helper with confidence scoring.
"""

from dataclasses import dataclass, field
import difflib
import io
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger("data.schema")

# Canonical column groupings
REQUIRED_ID_COL = "study_id"
REQUIRED_SURVEY_ITEMS = [
    "S1", "S2", "S3",
    "M1", "M2", "M3",
    "B1", "B2", "B3",
    "SE1", "SE2", "SE3",
    "PS1", "PS2", "PS3",
]
OPTIONAL_METADATA_COLS = ["age_group", "gender", "year_of_study", "programme"]
OPTIONAL_OUTCOME_COLS = ["attrition_label", "dropout_intention"]

ALL_CANONICAL_COLS: Set[str] = {
    REQUIRED_ID_COL,
    *REQUIRED_SURVEY_ITEMS,
    *OPTIONAL_METADATA_COLS,
    *OPTIONAL_OUTCOME_COLS,
}

REQUIRED_COLUMNS = [REQUIRED_ID_COL] + REQUIRED_SURVEY_ITEMS

# Exported symbols for external modules
__all__ = [
    "REQUIRED_ID_COL",
    "REQUIRED_SURVEY_ITEMS",
    "OPTIONAL_METADATA_COLS",
    "OPTIONAL_OUTCOME_COLS",
    "ALL_CANONICAL_COLS",
    "REQUIRED_COLUMNS",
    "read_survey_file",
    "suggest_column_mappings",
    "apply_column_mappings",
    "validate_dataframe",
]

@dataclass
class ColumnMappingSuggestion:
    """Suggestion record mapping an uploaded raw column to a target schema column."""

    raw_column: str
    suggested_target: str
    confidence: float
    reason: str


@dataclass
class ValidationReport:
    """Structured report returned by validate_dataframe."""

    passed: bool
    summary: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_required_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    out_of_range_items: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    non_integer_items: Dict[str, List[Any]] = field(default_factory=dict)
    duplicate_study_ids: List[str] = field(default_factory=list)
    missing_value_rates: Dict[str, float] = field(default_factory=dict)
    straight_lining_respondents: List[str] = field(default_factory=list)
    total_rows: int = 0
    valid_rows: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format for API serialization or logging."""
        return {
            "passed": self.passed,
            "summary": self.summary,
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_required_columns": self.missing_required_columns,
            "extra_columns": self.extra_columns,
            "out_of_range_items": self.out_of_range_items,
            "non_integer_items": self.non_integer_items,
            "duplicate_study_ids": self.duplicate_study_ids,
            "missing_value_rates": self.missing_value_rates,
            "straight_lining_respondents": self.straight_lining_respondents,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
        }


def load_schema_config(schema_path: str = "config/schema.yaml") -> Dict[str, Any]:
    """Load schema rules and aliases from YAML configuration.

    Args:
        schema_path (str): Relative or absolute path to schema.yaml.

    Returns:
        Dict[str, Any]: Parsed schema definition.
    """
    path = Path(schema_path)
    if not path.is_file():
        logger.warning("Schema config not found at %s. Using default internal contract.", schema_path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_survey_file(
    file_path_or_buffer: Union[str, Path, BinaryIO, io.BytesIO],
    file_type: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read survey data from .csv, .xlsx, or .xls file or in-memory buffer.

    Args:
        file_path_or_buffer: File system path or file-like buffer.
        file_type: Optional explicit file type override ('csv', 'xlsx', 'xls').
        **kwargs: Additional parameters passed to pandas reader.

    Returns:
        pd.DataFrame: Loaded survey DataFrame.

    Raises:
        ValueError: If file format is unsupported or cannot be parsed.
        FileNotFoundError: If the specified file path does not exist.
    """
    if isinstance(file_path_or_buffer, (str, Path)):
        path = Path(file_path_or_buffer)
        if not path.exists():
            raise FileNotFoundError(f"Survey file does not exist: {path}")
        ext = (file_type or path.suffix).lower().lstrip(".")
        target = path
    else:
        if not file_type:
            raise ValueError("file_type ('csv', 'xlsx', 'xls') must be provided when reading from a stream buffer.")
        ext = file_type.lower().lstrip(".")
        target = file_path_or_buffer

    if ext == "csv":
        try:
            return pd.read_csv(target, **kwargs)
        except UnicodeDecodeError:
            return pd.read_csv(target, encoding="latin1", **kwargs)
    elif ext in ("xlsx", "xls"):
        engine = "openpyxl" if ext == "xlsx" else "xlrd"
        try:
            return pd.read_excel(target, engine=engine, **kwargs)
        except Exception:
            # Fallback engine attempt
            return pd.read_excel(target, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension '.{ext}'. Supported formats are: .csv, .xlsx, .xls")


def suggest_column_mappings(
    columns: List[str],
    schema_path: str = "config/schema.yaml",
    threshold: float = 0.55,
) -> Dict[str, ColumnMappingSuggestion]:
    """Suggest canonical column mappings using aliases, exact matching, and fuzzy matching.

    Args:
        columns: List of raw column names from uploaded data.
        schema_path: Path to schema.yaml for alias lookups.
        threshold: Minimum similarity threshold for fuzzy suggestions.

    Returns:
        Dict[str, ColumnMappingSuggestion]: Mapping keyed by raw column name.
    """
    schema_cfg = load_schema_config(schema_path)
    alias_map: Dict[str, str] = {}

    # Extract aliases from schema.yaml
    items_cfg = schema_cfg.get("columns", {}).get("required_survey_items", [])
    for item in items_cfg:
        canonical = item["name"]
        alias_map[canonical.lower()] = canonical
        for a in item.get("aliases", []):
            alias_map[a.lower()] = canonical

    # Add id aliases
    for a in ["study_id", "studyid", "id", "student_id", "participant_id", "subject_id"]:
        alias_map[a.lower()] = REQUIRED_ID_COL

    # Add optional outcomes
    for outcome in schema_cfg.get("columns", {}).get("optional_outcomes", []):
        c_name = outcome["name"]
        alias_map[c_name.lower()] = c_name
        for a in outcome.get("aliases", []):
            alias_map[a.lower()] = c_name

    # Add optional metadata
    for meta in schema_cfg.get("columns", {}).get("optional_metadata", []):
        m_name = meta["name"]
        alias_map[m_name.lower()] = m_name
        for a in meta.get("aliases", []):
            alias_map[a.lower()] = m_name

    suggestions: Dict[str, ColumnMappingSuggestion] = {}

    for raw_col in columns:
        cleaned = raw_col.strip()
        norm = cleaned.lower()

        # 1. Exact canonical or alias match
        if norm in alias_map:
            target = alias_map[norm]
            confidence = 1.0 if cleaned == target else 0.95
            suggestions[raw_col] = ColumnMappingSuggestion(
                raw_column=raw_col,
                suggested_target=target,
                confidence=confidence,
                reason="Exact canonical or registered alias match",
            )
            continue

        # 2. Fuzzy match against aliases and canonical names
        best_target: Optional[str] = None
        best_score = 0.0
        best_matched_alias = ""

        for candidate_alias, canonical in alias_map.items():
            ratio = difflib.SequenceMatcher(None, norm, candidate_alias).ratio()
            # Bonus if candidate is a substring
            if candidate_alias in norm or norm in candidate_alias:
                ratio = max(ratio, 0.75)

            if ratio > best_score:
                best_score = ratio
                best_target = canonical
                best_matched_alias = candidate_alias

        if best_target and best_score >= threshold:
            suggestions[raw_col] = ColumnMappingSuggestion(
                raw_column=raw_col,
                suggested_target=best_target,
                confidence=round(best_score, 3),
                reason=f"Fuzzy similarity ({best_score:.2f}) to alias '{best_matched_alias}'",
            )

    return suggestions


def apply_column_mappings(df: pd.DataFrame, mappings: Dict[str, str]) -> pd.DataFrame:
    """Rename columns of a DataFrame according to an approved mapping dictionary.

    Args:
        df: Input DataFrame.
        mappings: Mapping dictionary {raw_column_name: target_column_name}.

    Returns:
        pd.DataFrame: DataFrame copy with renamed columns.
    """
    return df.rename(columns=mappings)


def validate_dataframe(
    df: pd.DataFrame,
    allow_extra_columns: bool = True,
    flag_straight_lining: bool = True,
) -> ValidationReport:
    """Execute thorough validation against the Psychosocial Framework Data Contract.

    Checks:
    - Missing required columns (`study_id` and all 15 Likert items).
    - Unrecognized / extra columns.
    - Out-of-range Likert responses (outside [1, 5]).
    - Non-integer or float responses for Likert items.
    - Outcome column constraints (`attrition_label` in {0, 1}, `dropout_intention` in [1, 5]).
    - Duplicate `study_id` records.
    - Missing-value (NaN/null) rates per column.
    - Straight-lining respondents (zero variance across all 15 survey items).

    Args:
        df: DataFrame to validate.
        allow_extra_columns: If False, unrecognized columns trigger validation errors.
        flag_straight_lining: If True, detects respondents with identical answers across all 15 items.

    Returns:
        ValidationReport: Complete diagnostic report.
    """
    errors: List[str] = []
    warnings: List[str] = []
    total_rows = len(df)

    # 1. Column presence checks
    present_cols = set(df.columns)
    missing_required = [c for c in [REQUIRED_ID_COL] + REQUIRED_SURVEY_ITEMS if c not in present_cols]
    if missing_required:
        errors.append(f"Missing required columns ({len(missing_required)}): {', '.join(missing_required)}")

    extra_cols = [c for c in df.columns if c not in ALL_CANONICAL_COLS]
    if extra_cols:
        msg = f"Extra unmapped columns detected ({len(extra_cols)}): {', '.join(extra_cols)}"
        if allow_extra_columns:
            warnings.append(msg)
        else:
            errors.append(msg)

    # 2. Missing-value rates per column
    # np.mean on the boolean array avoids TypeError on object-dtype columns
    missing_rates: Dict[str, float] = {}
    for col in df.columns:
        try:
            na_mask = df[col].isna()
            rate = float(np.mean(na_mask.values.astype(float)))
        except Exception:
            rate = 0.0
        if rate > 0.0:
            missing_rates[col] = round(rate, 4)
            if col in [REQUIRED_ID_COL] + REQUIRED_SURVEY_ITEMS:
                n_missing = int(na_mask.sum())
                errors.append(f"Required column '{col}' contains {n_missing} missing values ({rate:.1%}).")

    # 3. Duplicate study_ids
    duplicate_ids: List[str] = []
    if REQUIRED_ID_COL in df.columns:
        dups = df[df[REQUIRED_ID_COL].duplicated(keep=False)][REQUIRED_ID_COL].dropna().unique().tolist()
        duplicate_ids = [str(d) for d in dups]
        if duplicate_ids:
            errors.append(f"Duplicate study IDs found ({len(duplicate_ids)} duplicates): {duplicate_ids[:5]}...")

    # 4. Item boundary and integer checks
    out_of_range: Dict[str, List[Dict[str, Any]]] = {}
    non_integers: Dict[str, List[Any]] = {}

    for item in REQUIRED_SURVEY_ITEMS:
        if item not in df.columns:
            continue
        series = df[item].dropna()

        # Check integer type (or integer-valued floats like 3.0)
        invalid_types: List[Any] = []
        for idx, val in series.items():
            if isinstance(val, (int, np.integer)):
                continue
            elif isinstance(val, float) and val.is_integer():
                continue
            else:
                invalid_types.append({"index": idx, "value": str(val)})

        if invalid_types:
            non_integers[item] = invalid_types[:5]
            errors.append(f"Column '{item}' contains non-integer values (e.g. {invalid_types[0]['value']}).")

        # Range verification [1, 5]
        numeric_series = pd.to_numeric(series, errors="coerce")
        bad_range = numeric_series[(numeric_series < 1) | (numeric_series > 5)]
        if not bad_range.empty:
            out_of_range[item] = [
                {"index": int(i), "value": float(v)} for i, v in bad_range.head(5).items()
            ]
            errors.append(
                f"Column '{item}' has {len(bad_range)} values outside valid Likert range [1, 5] (e.g. {bad_range.iloc[0]})."
            )

    # 5. Outcome column validations
    if "attrition_label" in df.columns:
        col_data = df["attrition_label"]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        att = pd.to_numeric(col_data.dropna(), errors="coerce")
        bad_att = att[~att.isin([0, 1])]
        if not bad_att.empty:
            errors.append(f"attrition_label contains invalid values: must be strictly 0 or 1. Found: {bad_att.unique().tolist()}")

    if "dropout_intention" in df.columns:
        col_data = df["dropout_intention"]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        d_int = pd.to_numeric(col_data.dropna(), errors="coerce")
        bad_d_int = d_int[(d_int < 1) | (d_int > 5)]
        if not bad_d_int.empty:
            errors.append(f"dropout_intention contains values outside 1-5 scale: {bad_d_int.unique().tolist()}")

    # 6. Straight-lining respondent check (unengaged responses)
    straight_lining_ids: List[str] = []
    available_items = [c for c in REQUIRED_SURVEY_ITEMS if c in df.columns]
    if flag_straight_lining and len(available_items) == len(REQUIRED_SURVEY_ITEMS) and total_rows > 0:
        item_sub = df[available_items].apply(pd.to_numeric, errors="coerce")
        std_per_row = item_sub.std(axis=1)
        straight_mask = (std_per_row == 0.0) & (item_sub.notna().all(axis=1))
        straight_rows = df[straight_mask]

        if not straight_rows.empty:
            if REQUIRED_ID_COL in df.columns:
                straight_lining_ids = [str(x) for x in straight_rows[REQUIRED_ID_COL].tolist()]
            else:
                straight_lining_ids = [f"Row_{i}" for i in straight_rows.index.tolist()]
            warnings.append(
                f"Detected {len(straight_lining_ids)} straight-lining respondents with identical answers across all 15 items: {straight_lining_ids[:5]}."
            )

    # Pass/fail verdict
    passed = len(errors) == 0
    valid_rows = total_rows - len(straight_lining_ids) if passed else 0

    if passed:
        summary = f"PASSED: Survey DataFrame meets all data contract rules ({total_rows} rows evaluated)."
    else:
        summary = f"FAILED: Data contract violations detected ({len(errors)} error classes found)."

    logger.info("Validation completed. Status: %s | Errors: %d | Warnings: %d", "PASSED" if passed else "FAILED", len(errors), len(warnings))

    return ValidationReport(
        passed=passed,
        summary=summary,
        errors=errors,
        warnings=warnings,
        missing_required_columns=missing_required,
        extra_columns=extra_cols,
        out_of_range_items=out_of_range,
        non_integer_items=non_integers,
        duplicate_study_ids=duplicate_ids,
        missing_value_rates=missing_rates,
        straight_lining_respondents=straight_lining_ids,
        total_rows=total_rows,
        valid_rows=valid_rows,
    )
