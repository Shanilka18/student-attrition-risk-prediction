"""Unit tests for data contract, schema validation, fuzzy mapping, and readers."""

import io
import pytest
import pandas as pd
import numpy as np

from src.data.schema import (
    REQUIRED_SURVEY_ITEMS,
    REQUIRED_ID_COL,
    validate_dataframe,
    suggest_column_mappings,
    apply_column_mappings,
    read_survey_file,
    ValidationReport,
)


@pytest.fixture
def valid_survey_df() -> pd.DataFrame:
    """Fixture providing a strictly valid 5-student survey DataFrame."""
    data = {
        "study_id": [f"STD_{i:03d}" for i in range(1, 6)],
        "S1": [3, 4, 2, 5, 1],
        "S2": [2, 3, 2, 4, 2],
        "S3": [4, 4, 3, 5, 2],
        "M1": [4, 5, 3, 2, 5],
        "M2": [5, 4, 4, 1, 4],
        "M3": [4, 5, 4, 2, 5],
        "B1": [3, 4, 3, 2, 4],
        "B2": [4, 4, 4, 1, 5],
        "B3": [3, 3, 4, 2, 4],
        "SE1": [4, 5, 3, 2, 4],
        "SE2": [3, 4, 4, 1, 5],
        "SE3": [4, 4, 3, 2, 4],
        "PS1": [5, 4, 4, 2, 5],
        "PS2": [4, 5, 3, 1, 4],
        "PS3": [4, 4, 4, 2, 5],
        # Optional metadata
        "gender": ["male", "female", "female", "male", "non-binary"],
        "year_of_study": ["1", "2", "3", "4", "1"],
        # Optional outcomes
        "attrition_label": [0, 0, 0, 1, 0],
        "dropout_intention": [2, 1, 2, 5, 1],
    }
    return pd.DataFrame(data)


def test_validate_valid_dataframe(valid_survey_df):
    """Ensure a conforming DataFrame passes validation with 0 errors."""
    report: ValidationReport = validate_dataframe(valid_survey_df)
    assert report.passed is True
    assert len(report.errors) == 0
    assert report.total_rows == 5
    assert "PASSED" in report.summary


def test_missing_required_columns(valid_survey_df):
    """Ensure missing survey columns or ID triggers validation failure."""
    bad_df = valid_survey_df.drop(columns=["S1", "study_id"])
    report = validate_dataframe(bad_df)
    assert report.passed is False
    assert "S1" in report.missing_required_columns
    assert "study_id" in report.missing_required_columns
    assert any("Missing required columns" in e for e in report.errors)


def test_out_of_range_likert_values(valid_survey_df):
    """Ensure values outside [1, 5] fail validation."""
    bad_df = valid_survey_df.copy()
    bad_df.loc[0, "S2"] = 6
    bad_df.loc[1, "M1"] = 0
    bad_df.loc[2, "PS3"] = -2
    report = validate_dataframe(bad_df)
    assert report.passed is False
    assert "S2" in report.out_of_range_items
    assert "M1" in report.out_of_range_items
    assert "PS3" in report.out_of_range_items


def test_non_integer_values(valid_survey_df):
    """Ensure floats or strings in Likert columns fail validation."""
    bad_df = valid_survey_df.copy()
    bad_df["S3"] = bad_df["S3"].astype(object)
    bad_df.loc[0, "S3"] = 3.5
    bad_df.loc[1, "S3"] = "four"
    report = validate_dataframe(bad_df)
    assert report.passed is False
    assert "S3" in report.non_integer_items


def test_duplicate_study_ids(valid_survey_df):
    """Ensure duplicate study_ids are flagged as errors."""
    bad_df = valid_survey_df.copy()
    bad_df.loc[1, "study_id"] = bad_df.loc[0, "study_id"]  # Duplicate ID
    report = validate_dataframe(bad_df)
    assert report.passed is False
    assert len(report.duplicate_study_ids) > 0
    assert any("Duplicate study IDs found" in e for e in report.errors)


def test_missing_values_rates(valid_survey_df):
    """Ensure missing-value rate is recorded accurately and fails required items."""
    bad_df = valid_survey_df.copy()
    bad_df.loc[0, "B1"] = np.nan
    bad_df.loc[1, "B1"] = np.nan
    report = validate_dataframe(bad_df)
    assert report.passed is False
    assert "B1" in report.missing_value_rates
    assert report.missing_value_rates["B1"] == 0.40  # 2 out of 5


def test_straight_lining_respondents(valid_survey_df):
    """Ensure respondents with identical answers across all 15 items are flagged as unengaged."""
    bad_df = valid_survey_df.copy()
    # Student 0 answers '3' to everything (straight-lining)
    for col in REQUIRED_SURVEY_ITEMS:
        bad_df.loc[0, col] = 3

    report = validate_dataframe(bad_df, flag_straight_lining=True)
    assert "STD_001" in report.straight_lining_respondents
    assert any("straight-lining" in w for w in report.warnings)


def test_invalid_outcome_columns(valid_survey_df):
    """Ensure invalid attrition labels (>1) or dropout intention (>5) are rejected."""
    bad_df = valid_survey_df.copy()
    bad_df.loc[0, "attrition_label"] = 2
    bad_df.loc[0, "dropout_intention"] = 7
    report = validate_dataframe(bad_df)
    assert report.passed is False
    assert any("attrition_label contains invalid values" in e for e in report.errors)
    assert any("dropout_intention contains values outside" in e for e in report.errors)


def test_fuzzy_column_mapping_suggestions():
    """Ensure fuzzy column mapper correctly suggests canonical columns with high confidence."""
    raw_columns = [
        "Q1_stress",
        "Exam_Anxiety",
        "degree_interest",
        "peer_collaboration",
        "Participant_ID",
        "academic_resilience",
        "attrition",
    ]
    suggestions = suggest_column_mappings(raw_columns)
    assert "Q1_stress" in suggestions
    assert suggestions["Q1_stress"].suggested_target == "S1"
    assert suggestions["Q1_stress"].confidence >= 0.70

    assert suggestions["Exam_Anxiety"].suggested_target == "S2"
    assert suggestions["degree_interest"].suggested_target == "M1"
    assert suggestions["peer_collaboration"].suggested_target == "PS3"
    assert suggestions["Participant_ID"].suggested_target == "study_id"
    assert suggestions["academic_resilience"].suggested_target == "SE2"
    assert suggestions["attrition"].suggested_target == "attrition_label"


def test_apply_column_mappings():
    """Verify DataFrame columns are renamed accurately using mappings dictionary."""
    df = pd.DataFrame({
        "Q1_stress": [3, 4],
        "Participant_ID": ["ID1", "ID2"],
    })
    mapping = {"Q1_stress": "S1", "Participant_ID": "study_id"}
    renamed = apply_column_mappings(df, mapping)
    assert "S1" in renamed.columns
    assert "study_id" in renamed.columns
    assert "Q1_stress" not in renamed.columns


def test_read_survey_file_csv_and_excel(tmp_path, valid_survey_df):
    """Test reading from CSV and Excel (.xlsx) files."""
    csv_file = tmp_path / "survey.csv"
    xlsx_file = tmp_path / "survey.xlsx"

    valid_survey_df.to_csv(csv_file, index=False)
    valid_survey_df.to_excel(xlsx_file, index=False, engine="openpyxl")

    loaded_csv = read_survey_file(csv_file)
    assert len(loaded_csv) == len(valid_survey_df)
    assert "study_id" in loaded_csv.columns

    loaded_xlsx = read_survey_file(xlsx_file)
    assert len(loaded_xlsx) == len(valid_survey_df)
    assert "study_id" in loaded_xlsx.columns


def test_read_survey_file_buffer(valid_survey_df):
    """Test reading from in-memory BytesIO buffer."""
    buf = io.BytesIO()
    valid_survey_df.to_csv(buf, index=False)
    buf.seek(0)

    loaded = read_survey_file(buf, file_type="csv")
    assert len(loaded) == len(valid_survey_df)
    assert "study_id" in loaded.columns
