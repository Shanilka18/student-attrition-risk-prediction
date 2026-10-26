"""Unit tests for psychometric feature computation and PRI scaling."""

import pytest
import pandas as pd
import numpy as np

from src.features.build_features import (
    reverse_protective_items,
    compute_factor_scores,
    compute_pri_score,
    categorize_risk,
)


@pytest.fixture
def sample_survey_record():
    """Create a minimal survey sample DataFrame."""
    return pd.DataFrame([
        {
            "study_id": "TEST_001",
            "S1": 5, "S2": 5, "S3": 5,  # Max stress = 5.0
            "M1": 1, "M2": 1, "M3": 1,  # Min motivation = 1.0 (inverted = 5.0)
            "B1": 1, "B2": 1, "B3": 1,  # Min belongingness = 1.0 (inverted = 5.0)
            "SE1": 1, "SE2": 1, "SE3": 1, # Min self-efficacy = 1.0 (inverted = 5.0)
            "PS1": 1, "PS2": 1, "PS3": 1, # Min peer support = 1.0 (inverted = 5.0)
        },
        {
            "study_id": "TEST_002",
            "S1": 1, "S2": 1, "S3": 1,  # Min stress = 1.0
            "M1": 5, "M2": 5, "M3": 5,  # Max motivation = 5.0 (inverted = 1.0)
            "B1": 5, "B2": 5, "B3": 5,  # Max belongingness = 5.0 (inverted = 1.0)
            "SE1": 5, "SE2": 5, "SE3": 5, # Max self-efficacy = 5.0 (inverted = 1.0)
            "PS1": 5, "PS2": 5, "PS3": 5, # Max peer support = 5.0 (inverted = 1.0)
        },
    ])


def test_reverse_protective_items(sample_survey_record):
    """Test that protective items are reversed correctly on 1-5 scale."""
    rev_df = reverse_protective_items(sample_survey_record)
    # Row 0: M1 was 1 -> reversed should be 5
    assert rev_df["M1_rev"].iloc[0] == 5
    # Row 1: M1 was 5 -> reversed should be 1
    assert rev_df["M1_rev"].iloc[1] == 1


def test_compute_factor_scores(sample_survey_record):
    """Test that factor scores are computed as arithmetic mean of 3 items."""
    factors_df = compute_factor_scores(sample_survey_record)
    assert factors_df["factor_stress"].iloc[0] == 5.0
    assert factors_df["factor_motivation"].iloc[0] == 1.0
    assert factors_df["factor_stress"].iloc[1] == 1.0
    assert factors_df["factor_motivation"].iloc[1] == 5.0


def test_compute_pri_score_boundaries(sample_survey_record):
    """Test that PRI correctly scales between 0.0 and 100.0 on extreme cases."""
    factors_df = compute_factor_scores(sample_survey_record)
    pri = compute_pri_score(factors_df)
    # Highest possible risk scenario (max stress, min protective) -> PRI == 100
    assert np.isclose(pri.iloc[0], 100.0)
    # Lowest possible risk scenario (min stress, max protective) -> PRI == 0
    assert np.isclose(pri.iloc[1], 0.0)


def test_categorize_risk():
    """Test that risk categorization bins correctly into Low, Moderate, High using cutoffs (34.0, 66.0)."""
    test_scores = pd.Series([10.0, 33.9, 34.1, 50.0, 65.9, 66.1, 95.0])
    cats = categorize_risk(test_scores, low_cut=34.0, mod_cut=66.0)
    assert cats.iloc[0] == "Low"
    assert cats.iloc[1] == "Low"
    assert cats.iloc[2] == "Moderate"
    assert cats.iloc[3] == "Moderate"
    assert cats.iloc[4] == "Moderate"
    assert cats.iloc[5] == "High"
    assert cats.iloc[6] == "High"
