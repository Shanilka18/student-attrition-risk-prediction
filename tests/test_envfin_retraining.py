from pathlib import Path

import pandas as pd
import pytest

from src.cleaning.envfin import CleaningResult
from src.inference import retrain


@pytest.mark.parametrize(
    ("confirm_overwrite", "expected_status"),
    [(False, "dry_run"), (True, "failed")],
)
def test_retraining_cleans_first_and_blocks_legacy_feature_contract(
    monkeypatch, confirm_overwrite, expected_status
):
    cleaned = CleaningResult(
        data=pd.DataFrame({"at_risk": [0, 1], "income_band": [0, 1]}),
        report={"traffic_light": {"status": "green", "messages": []}},
        parquet_path=Path("clean.parquet"),
        report_path=Path("quality.json"),
    )

    class FakeCleaner:
        def clean(self, file_path, mode):
            assert file_path == "survey.csv"
            assert mode == "training"
            return cleaned

        def write_result(self, result):
            assert result is cleaned
            return result

    monkeypatch.setattr(retrain, "EnvironmentalFinancialCleaner", FakeCleaner)

    result = retrain.retrain_pipeline_on_file(
        "survey.csv", confirm_overwrite=confirm_overwrite
    )

    assert result["status"] == expected_status
    assert "feature contract" in result.get("reason", result.get("message", ""))
    assert result["clean_parquet"] == "clean.parquet"
    assert result["quality_report"] == "quality.json"
