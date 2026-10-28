"""Comprehensive unit and end-to-end tests for the MEI cleaning pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd

from mei.cleaning.encoder import encode_and_validate_indicators
from mei.cleaning.filters import (
    apply_missingness_policy,
    detect_careless_responding,
    filter_attention_check,
    filter_consent,
    resolve_duplicates,
)
from mei.cleaning.mapper import map_headers
from mei.cleaning.models import TrafficLightStatus
from mei.cleaning.outcomes import process_outcomes
from mei.cleaning.pipeline import CleaningPipeline
from mei.cleaning.reader import read_survey_file
from mei.config import DATA_DIR


def test_rule_r01_reader_and_normalization(tmp_path: Path) -> None:
    """Stage 1: Ingestion must normalize quotes, dashes, and drop empty rows."""
    csv_file = tmp_path / "test_input.csv"
    csv_content = (
        "student_id,resource_access_p1\n"
        'STU_001,"  1–2 days  "\n'
        'STU_002,"Every day"\n'
        ",\n"  # empty row
        'STU_003,"“Every day”"\n'
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    df, log = read_survey_file(csv_file)
    assert len(df) == 3
    assert log.rule_id == "R01_INGESTION_CLEANUP"
    assert df.loc[0, "resource_access_p1"] == "1-2 days"
    assert df.loc[2, "resource_access_p1"] == '"Every day"'


def test_rule_r02_mapper_exact_norm_fuzzy() -> None:
    """Stage 2: Header mapping must resolve exact, normalized, and fuzzy suggestions."""
    df = pd.DataFrame(
        {
            "Student ID": ["S1"],
            "resource_access_p1": [1],
            (
                "How many days per week did you open online learning materials? "
                "[Period 2: Weeks 5–8]"
            ): [2],
            "extra_unmapped_column": [99],
        }
    )
    mapped_df, res, log = map_headers(df, mode="scoring")
    assert "student_id" in mapped_df.columns
    assert "resource_access_p1" in mapped_df.columns
    assert "resource_access_p2" in mapped_df.columns
    assert "extra_unmapped_column" in res.unknown_extra
    assert log.rule_id == "R02_HEADER_MAPPING"


def test_rule_r03a_consent_filter() -> None:
    """Stage 3a: Filter out non-consent in training and flag in scoring."""
    df = pd.DataFrame(
        {
            "student_id": ["S1", "S2", "S3", "S4"],
            "consent": ["I consent to participate", "I do not consent", "1", "0"],
        }
    )
    # Training mode: drop non-consent
    train_df, non_consent_train, log_train = filter_consent(df, mode="training")
    assert len(train_df) == 2
    assert set(train_df["student_id"]) == {"S1", "S3"}
    assert len(non_consent_train) == 2

    # Scoring mode: preserve rows
    score_df, non_consent_score, log_score = filter_consent(df, mode="scoring")
    assert len(score_df) == 4
    assert len(non_consent_score) == 2


def test_rule_r03b_attention_check_filter() -> None:
    """Stage 3b: Filter out attention check failures."""
    df = pd.DataFrame(
        {
            "student_id": ["S1", "S2", "S3"],
            "attention_check": ["Sometimes", "Never", "Always"],
        }
    )
    clean_df, failed, log = filter_attention_check(df, mode="training")
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["student_id"] == "S1"
    assert len(failed) == 2


def test_rule_r04_codec_typo_encoding() -> None:
    """Stage 4: Codec must convert text and tolerate minor typographical errors."""
    df = pd.DataFrame(
        {
            "resource_access_p1": [
                "Every day",
                "1–2 days",
                "Everyday",
                "UnknownGibberish",
            ],
            "quiz_practice_p1": ["Sometimes", "Some times", "Alwys", None],
        }
    )
    encoded_df, log4, log5 = encode_and_validate_indicators(df, typo_threshold=85.0)

    # "Every day" -> 5.0
    assert encoded_df.loc[0, "resource_access_p1"] == 5.0
    # "1–2 days" -> 2.0
    assert encoded_df.loc[1, "resource_access_p1"] == 2.0
    # "Everyday" -> typo corrected to "Every day" -> 5.0
    assert encoded_df.loc[2, "resource_access_p1"] == 5.0
    # "UnknownGibberish" -> NaN
    assert pd.isna(encoded_df.loc[3, "resource_access_p1"])

    # "Sometimes" -> 3.0
    assert encoded_df.loc[0, "quiz_practice_p1"] == 3.0
    # "Some times" -> typo corrected to "Sometimes" -> 3.0
    assert encoded_df.loc[1, "quiz_practice_p1"] == 3.0
    # "Alwys" -> typo corrected to "Always" -> 5.0
    assert encoded_df.loc[2, "quiz_practice_p1"] == 5.0


def test_rule_r05_range_validation() -> None:
    """Stage 5: Ordinal values must strictly lie within [1, 5]; out of range -> NaN."""
    df = pd.DataFrame(
        {
            "study_time_p1": [1, 5, 99, -1, 0, 3],
        }
    )
    encoded_df, _, log5 = encode_and_validate_indicators(df)
    vals = encoded_df["study_time_p1"].tolist()
    assert vals[0] == 1.0
    assert vals[1] == 5.0
    assert pd.isna(vals[2])
    assert pd.isna(vals[3])
    assert pd.isna(vals[4])
    assert vals[5] == 3.0


def test_rule_r06_duplicate_resolution() -> None:
    """Stage 6: Keep the LAST submission per student ID."""
    df = pd.DataFrame(
        {
            "student_id": ["STU_001", "STU_002", "STU_001"],
            "timestamp": [
                "2026-09-01 10:00:00",
                "2026-09-01 11:00:00",
                "2026-09-02 12:00:00",
            ],
            "val": [1, 2, 3],
        }
    )
    dedup_df, dup_idx, log = resolve_duplicates(df, mode="training")
    assert len(dedup_df) == 2
    assert set(dedup_df["student_id"]) == {"STU_001", "STU_002"}
    # Verify that the kept STU_001 row is the later submission (val=3)
    stu1_row = dedup_df[dedup_df["student_id"] == "STU_001"].iloc[0]
    assert stu1_row["val"] == 3


def test_rule_r07_straight_lining_careless() -> None:
    """Stage 7: Identify straight-liners with zero response variance."""
    cols = [f"resource_access_p{t}" for t in range(1, 5)] + [
        f"study_time_p{t}" for t in range(1, 5)
    ]
    # Row 0: Straight-liner (all 3s)
    # Row 1: Normal varied responder
    df = pd.DataFrame(
        [
            [3.0] * 8,
            [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0],
        ],
        columns=cols,
    )
    filtered_df, careless_idx, log = detect_careless_responding(
        df, mode="training", feature_cols=cols
    )
    assert len(careless_idx) == 1
    assert 0 in careless_idx
    assert len(filtered_df) == 1
    assert filtered_df.index[0] == 1


def test_rule_r08_missingness_policy_no_imputation() -> None:
    """Stage 8: Drop rows with >25% missing; NEVER impute remaining NaNs (G3)."""
    cols = [f"feat_{i}" for i in range(20)]
    # Row 0: 2 missing out of 20 = 10% (kept)
    # Row 1: 8 missing out of 20 = 40% (dropped)
    row0 = [1.0] * 18 + [np.nan, np.nan]
    row1 = [1.0] * 12 + [np.nan] * 8

    df = pd.DataFrame([row0, row1], columns=cols)
    clean_df, summary, excess_idx, log = apply_missingness_policy(
        df, mode="training", max_missing_pct=0.25, feature_cols=cols
    )

    assert len(clean_df) == 1
    assert 1 in excess_idx
    # Assert remaining NaNs in row 0 are PRESERVED and NOT imputed
    assert clean_df["feat_18"].isna().iloc[0]
    assert clean_df["feat_19"].isna().iloc[0]


def test_rule_r09_outcome_handling() -> None:
    """Stage 9: Compute attrition_intent_score and at_risk label correctly."""
    df = pd.DataFrame(
        {
            "intent_leave": [1, 5, 3],
            "intent_defer": [1, 5, 3],
            "intent_not_continue": [1, 5, 3],
        }
    )
    # Row 0: mean=1 -> score = ((1-1)/4)*100 = 0.0 -> at_risk = 0
    # Row 1: mean=5 -> score = ((5-1)/4)*100 = 100.0 -> at_risk = 1
    # Row 2: mean=3 -> score = ((3-1)/4)*100 = 50.0 -> at_risk = 0
    out_df, missing_idx, balance, log = process_outcomes(df, mode="training")
    assert out_df.loc[0, "attrition_intent_score"] == 0.0
    assert out_df.loc[0, "at_risk"] == 0
    assert out_df.loc[1, "attrition_intent_score"] == 100.0
    assert out_df.loc[1, "at_risk"] == 1
    assert out_df.loc[2, "attrition_intent_score"] == 50.0
    assert out_df.loc[2, "at_risk"] == 0
    assert balance["at_risk_count"] == 1
    assert balance["not_at_risk_count"] == 2


def test_end_to_end_raw_sample_cleaning() -> None:
    """End-to-end test on raw generated sample file."""
    raw_path = DATA_DIR / "raw" / "survey_raw_sample.csv"
    assert raw_path.exists(), "Raw sample file missing. Run make-sample-data first."

    pipeline = CleaningPipeline()
    clean_df, report = pipeline.clean(
        raw_path,
        mode="training",
        run_id="test_run_e2e",
        save_artifacts=True,
    )

    # 1. Verification of output dimensions and retention
    assert report.rows_in == 2412
    assert report.rows_out >= 2100  # Expect ~90% retention
    assert report.status == TrafficLightStatus.GREEN

    # 2. Assert removed anomaly counts match injected rates
    rules_map = {r.rule_id: r for r in report.rules_applied}

    # Duplicates injected: 0.5% (~12 rows)
    assert 8 <= rules_map["R06_DUPLICATE_RESOLUTION"].rows_affected <= 15

    # Consent refusals: 1.0% (~24 rows)
    assert 15 <= rules_map["R03A_CONSENT_FILTER"].rows_affected <= 30

    # Attention check failures: 2.0% (~48 rows)
    assert 40 <= rules_map["R03B_ATTENTION_CHECK"].rows_affected <= 75

    # Straight-liners: 2.0% (~48 rows)
    assert 35 <= rules_map["R07_CARELESS_RESPONDING"].rows_affected <= 100

    # 3. Verify clean parquet file
    parquet_path = DATA_DIR / "interim" / "clean_test_run_e2e.parquet"
    assert parquet_path.exists()
    reloaded_df = pd.read_parquet(parquet_path)
    assert len(reloaded_df) == report.rows_out
    assert "attrition_intent_score" in reloaded_df.columns
    assert "at_risk" in reloaded_df.columns


def test_scoring_mode_messy_upload() -> None:
    """Scoring mode on messy upload must produce suggestions without crash."""
    messy_path = DATA_DIR / "samples" / "sample_upload_messy.csv"
    assert messy_path.exists()

    pipeline = CleaningPipeline()
    clean_df, report = pipeline.clean(
        messy_path,
        mode="scoring",
        run_id="test_scoring_messy",
        save_artifacts=False,
    )

    # Scoring mode preserves total rows
    assert report.rows_in == 200
    assert report.rows_out == 200
    assert report.status == TrafficLightStatus.AMBER

    # Check warnings for extra unrecognized column
    assert any("campus_branch" in w for w in report.warnings)

    # Check non-scorable row reasons
    assert len(report.rejected_rows) > 0
    for rej in report.rejected_rows:
        assert len(rej.rejection_reasons) > 0
