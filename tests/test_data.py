"""Unit tests for data generation and integrity verification."""

import pytest
import pandas as pd
from src.data.make_dataset import generate_synthetic_survey, validate_survey_data


def test_synthetic_data_generation_shape():
    """Verify generated synthetic dataset dimensions and required columns."""
    df = generate_synthetic_survey(n_samples=50, seed=123)
    assert len(df) == 50
    assert "study_id" in df.columns
    assert "is_synthetic" in df.columns
    assert df["is_synthetic"].all() == True

    # Check that all 15 questionnaire columns exist
    expected_items = [
        "S1", "S2", "S3",
        "M1", "M2", "M3",
        "B1", "B2", "B3",
        "SE1", "SE2", "SE3",
        "PS1", "PS2", "PS3",
    ]
    for item in expected_items:
        assert item in df.columns
        assert df[item].min() >= 1
        assert df[item].max() <= 5


def test_validation_passes_valid_data():
    """Ensure validation passes for valid synthetic survey data."""
    df = generate_synthetic_survey(n_samples=20, seed=42)
    assert validate_survey_data(df) is True


def test_validation_fails_out_of_bounds():
    """Ensure validation fails if any Likert response is outside [1, 5]."""
    df = generate_synthetic_survey(n_samples=20, seed=42)
    df.loc[0, "S1"] = 6  # Invalid Likert value
    with pytest.raises(ValueError, match="outside Likert range"):
        validate_survey_data(df)
