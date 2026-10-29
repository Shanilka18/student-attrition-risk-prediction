"""Unit and Integration Tests for Phase 8 Production Inference Engine and Retraining Module."""

from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from src.inference.ingestion import (
    align_dataframe_schema,
    detect_and_strip_pii,
    load_raw_file,
)
from src.inference.predict import ProductionInferenceEngine
from src.inference.retrain import retrain_pipeline_on_file
from src.preprocessing.codec import EnvFinCodec


class TestInferenceIngestionAndSecurity:
    def test_pii_columns_strictly_stripped(self):
        """Assert direct identifiers are permanently purged and never retained in memory."""
        dirty_df = pd.DataFrame(
            {
                "name": ["John Doe", "Jane Smith"],
                "student_id": ["IT20104523", "IT20108922"],
                "nic": ["200124501234", "991240123V"],
                "email": ["john@sliit.lk", "jane@sliit.lk"],
                "phone": ["0771234567", "0719876543"],
                "household_income_bracket": [3, 4],
                "financial_stress": [2, 3],
                "commute_time_min": [45.0, 60.0],
            }
        )

        clean_df, dropped_pii = detect_and_strip_pii(dirty_df)

        assert len(dropped_pii) == 5
        for col in dirty_df.columns:
            if col in ["name", "student_id", "nic", "email", "phone"]:
                assert col not in clean_df.columns

    def test_schema_alignment_synonyms_and_pseudonym_generation(self):
        """Test case/space-insensitive synonyms matching and pseudonym ID creation."""
        messy_df = pd.DataFrame(
            {
                "Monthly Income": [2, 5],
                "TRAVEL_TIME (MIN)": [90.0, 30.0],
                "job hours": [20.0, 0.0],
                "Living Situation": ["boarding", "with parents"],
                "Unknown Feature": ["A", "B"],
            }
        )

        aligned_df, report = align_dataframe_schema(messy_df)

        # Synonym assertions
        assert "household_income_bracket" in aligned_df.columns
        assert "commute_time_min" in aligned_df.columns
        assert "work_hours_per_week" in aligned_df.columns
        assert "housing_type" in aligned_df.columns

        # Unknown ignored column
        assert "Unknown Feature" in report["ignored_unknown_columns"]

        # Pseudonymous ID generated
        assert "participant_id" in aligned_df.columns
        assert aligned_df["participant_id"].iloc[0] == "P-00001"
        assert aligned_df["participant_id"].iloc[1] == "P-00002"

    def test_empty_file_rejected(self, tmp_path):
        empty_csv = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(empty_csv, index=False)

        with pytest.raises(ValueError, match="empty"):
            load_raw_file(empty_csv)


@pytest.mark.skip(
    reason="The current production artifacts were trained for a different feature schema."
)
class TestProductionInferenceEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        return ProductionInferenceEngine()

    def test_single_row_inference_edge_case(self, engine):
        """Verify pipeline operates seamlessly on N=1 batch."""
        single_row_df = pd.DataFrame(
            [
                {
                    "participant_id": "P-SINGLE-01",
                    "household_income_bracket": 1,
                    "financial_stress": 5,
                    "scholarship_bursary": 0,
                    "dependents_count": 3,
                    "housing_type": "boarding",
                    "commute_time_min": 150.0,
                    "commute_cost_lkr_week": 4500.0,
                    "internet_reliability": 1,
                    "internet_affordability": 1,
                    "device_type": "phone only",
                    "device_quality": 1,
                    "quiet_study_space": 0,
                    "part_time_employment": 1,
                    "work_hours_per_week": 25.0,
                    "faculty": "Computing",
                    "year_of_study": 3,
                    "district_type": "rural",
                    "baseline_gpa_band": "Band 1 (< 2.00)",
                }
            ]
        )

        df_scored, _ = engine.score_dataframe(single_row_df)

        assert len(df_scored) == 1
        row = df_scored.iloc[0]
        assert row["participant_id"] == "P-SINGLE-01"
        assert 0.0 <= row["eri_score"] <= 100.0
        assert row["risk_band"] in ["Low", "Moderate", "High", "Critical"]
        assert len(row["top_positive_drivers"]) > 0
        assert "This explains the model's score" in row["narrative"]

    def test_missing_features_confidence_penalty(self, engine):
        """Verify data quality confidence notes when inputs are sparse."""
        sparse_df = pd.DataFrame(
            [
                {
                    "household_income_bracket": 2,
                    "financial_stress": 4,
                    # All other questionnaire columns omitted
                }
            ]
        )

        df_scored, _ = engine.score_dataframe(sparse_df)
        note = df_scored.iloc[0]["confidence_note"]
        assert "LOW CONFIDENCE" in note or "MODERATE CONFIDENCE" in note

    def test_performance_high_volume_batch_50k(self, engine):
        """Stress-test 50,000 synthetic rows for latency and memory stability."""
        np.random.seed(42)
        n = 50000

        # Create 50k lightweight mock questionnaire
        mock_data = {
            "household_income_bracket": np.random.randint(1, 7, size=n),
            "financial_stress": np.random.randint(1, 6, size=n),
            "scholarship_bursary": np.random.randint(0, 2, size=n),
            "dependents_count": np.random.randint(0, 5, size=n),
            "housing_type": np.random.choice(
                ["with parents", "boarding", "hostel", "rented room"], size=n
            ),
            "commute_time_min": np.random.uniform(10.0, 180.0, size=n),
            "commute_cost_lkr_week": np.random.uniform(500.0, 6000.0, size=n),
            "internet_reliability": np.random.randint(1, 6, size=n),
            "internet_affordability": np.random.randint(1, 6, size=n),
            "device_type": np.random.choice(
                ["own laptop", "shared laptop", "phone only", "none"], size=n
            ),
            "device_quality": np.random.randint(1, 6, size=n),
            "quiet_study_space": np.random.randint(0, 2, size=n),
            "part_time_employment": np.random.randint(0, 2, size=n),
            "work_hours_per_week": np.random.uniform(0.0, 30.0, size=n),
            "faculty": np.random.choice(
                ["Computing", "Engineering", "Business", "Humanities"], size=n
            ),
            "year_of_study": np.random.randint(1, 5, size=n),
            "district_type": np.random.choice(["urban", "semi-urban", "rural"], size=n),
            "baseline_gpa_band": np.random.choice(
                [
                    "Band 1 (< 2.00)",
                    "Band 2 (2.00 - 2.49)",
                    "Band 3 (2.50 - 2.99)",
                    "Band 4 (3.00 - 3.49)",
                    "Band 5 (3.50 - 4.00)",
                ],
                size=n,
            ),
        }
        df_50k = pd.DataFrame(mock_data)

        import time

        t0 = time.time()
        df_scored, _ = engine.score_dataframe(df_50k)
        elapsed = time.time() - t0

        assert len(df_scored) == 50000
        # Assert throughput: 50k rows scored in under 25 seconds
        assert (
            elapsed < 25.0
        ), f"50k scoring took {elapsed:.2f}s (exceeded 25s threshold)"


def test_incompatible_preprocessor_schema_is_rejected(tmp_path):
    preprocessor_path = tmp_path / "scalers" / "production_preprocessor.joblib"
    preprocessor_path.parent.mkdir(parents=True)
    legacy_preprocessor = SimpleNamespace(
        feature_names_in_=np.array(["baseline_gpa_band"])
    )
    joblib.dump(legacy_preprocessor, preprocessor_path)

    with pytest.raises(RuntimeError, match="feature schema is incompatible"):
        ProductionInferenceEngine(artifacts_dir=tmp_path)


class TestRetrainingSafetyGuardrails:
    def test_retraining_without_outcome_column_fails_gracefully(self, tmp_path):
        """Assert retraining gracefully refuses without an outcome column."""
        no_outcome_csv = tmp_path / "no_outcome.csv"
        pd.DataFrame(
            {
                "household_income_bracket": [1, 2],
                "financial_stress": [4, 5],
            }
        ).to_csv(no_outcome_csv, index=False)

        res = retrain_pipeline_on_file(str(no_outcome_csv), outcome_col="at_risk")
        assert res["status"] == "failed"
        assert "requires at least one outcome column" in res["reason"]

    def test_retraining_dry_run_without_confirm_flag(self, tmp_path):
        """Assert retraining will not execute without explicit confirmation."""
        valid_csv = tmp_path / "valid.csv"
        codec = EnvFinCodec()
        mappings = codec.mapping["canonical_mappings"]
        rows = []
        for index, score in enumerate((20, 80), start=1):
            row = {}
            for item_position, (item, metadata) in enumerate(codec.items.items()):
                levels = metadata["levels"]
                level_index = (item_position + index - 1) % len(levels)
                row[mappings[item]["google_forms_header"]] = levels[level_index][
                    "label"
                ]
            row[mappings["student_id"]["google_forms_header"]] = f"P-{index:05d}"
            row[mappings["timestamp"]["google_forms_header"]] = (
                f"2026-01-01 09:0{index}:00"
            )
            row[mappings["consent"]["google_forms_header"]] = "Yes / I agree"
            row[mappings["attention_check"]["google_forms_header"]] = "Agree"
            row[mappings["attrition_intent_score"]["google_forms_header"]] = score
            risk_label = codec.outcomes["at_risk"]["levels"][int(score >= 60)]["label"]
            row[mappings["at_risk"]["google_forms_header"]] = risk_label
            rows.append(row)
        pd.DataFrame(rows).to_csv(valid_csv, index=False)

        res = retrain_pipeline_on_file(
            str(valid_csv), outcome_col="at_risk", confirm_overwrite=False
        )
        assert res["status"] == "dry_run"
