"""Tests for matching generated-data profile recovery to hidden truth."""

from __future__ import annotations

import pandas as pd

from src.profiles.workflow import _hidden_type_recovery


def test_hidden_type_recovery_joins_generated_ids_and_is_labeled_sample_only(
    tmp_path,
):
    sample_path = tmp_path / "survey_sample.csv"
    sample_path.touch()
    truth_path = tmp_path / "hidden_truth.csv"
    pd.DataFrame(
        {
            "type": ["stable", "strained", "stable", "strained"],
            "F": [0.1, 0.9, 0.2, 0.8],
        }
    ).to_csv(truth_path, index=False)
    profiles = pd.DataFrame(
        {
            "student_id": ["P-00001", "P-00002", "P-00003", "P-00004"],
            "profile_id": [0, 1, 0, 1],
        }
    )
    config = {
        "sample_data": {
            "outputs": {"hidden_truth": str(truth_path)},
            "identifiers": {"prefix": "P-", "digits": 5},
        }
    }

    result = _hidden_type_recovery(
        profiles,
        sample_path,
        sample_path,
        config,
        tmp_path,
    )

    assert result["status"] == "available"
    assert result["validity"] == "valid only for generated data"
    assert result["students_matched"] == 4
    assert result["ari"] == 1.0
    assert result["nmi"] == 1.0
