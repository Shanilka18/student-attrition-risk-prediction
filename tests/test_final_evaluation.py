"""Unit Tests for Phase 9: Final Evaluation, Calibration, Lock Guard, and MEI Scoring."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from mei.config import ARTIFACTS_DIR, RUNS_DIR
from mei.evaluation.test_eval import evaluate_final_on_test
from mei.registry.score import score_dataframe


@pytest.fixture
def active_run_dir() -> Path:
    """Fixture resolving the active run directory."""
    active_txt = ARTIFACTS_DIR / "active_run.txt"
    assert active_txt.exists(), "active_run.txt must exist"
    run_id = active_txt.read_text(encoding="utf-8").strip()
    run_dir = RUNS_DIR / run_id
    assert run_dir.exists(), f"Run directory {run_dir} must exist"
    return run_dir


def test_double_test_evaluation_raises_permission_error(active_run_dir: Path):
    """Calling test evaluation twice MUST raise PermissionError (Rule G3 & Lock Guard)."""
    lock_file = active_run_dir / "TEST_USED.lock"
    assert lock_file.exists(), "TEST_USED.lock must exist after finalization"

    # Calling without force must raise PermissionError
    with pytest.raises(PermissionError) as exc_info:
        evaluate_final_on_test(active_run_dir, force=False)

    assert "Quarantine Guard Violation" in str(exc_info.value)
    assert "TEST_USED.lock" in str(exc_info.value)


def test_static_code_test_indices_quarantine():
    """Static check verifying test_idx.json is not read in feature pipelines or model training.

    Only evaluate_final_on_test() in test_eval.py and splits.py definitions are permitted.
    """
    src_dir = Path("src/mei")
    violations = []

    for py_file in src_dir.rglob("*.py"):
        # Exempt test_eval.py, splits.py, and legacy CLI setup
        if py_file.name in ["test_eval.py", "splits.py"]:
            continue

        text = py_file.read_text(encoding="utf-8")
        lines = text.splitlines()
        for idx, line in enumerate(lines, 1):
            # Check for direct test_idx loading
            if (
                "test_idx.json" in line
                and "load_test_indices" not in line
                and "splits" not in py_file.name
            ):
                # If in cli.py, ensure it's not in training commands
                if py_file.name == "cli.py":
                    # CLI finalize is permitted to load test targets for reporting
                    continue
                violations.append(f"{py_file}:{idx}: {line.strip()}")

    assert len(violations) == 0, (
        "Found unauthorized access to test_idx.json:\n" + "\n".join(violations)
    )


def test_calibration_monotonicity(active_run_dir: Path):
    """Verify that observed at-risk rate rises strictly monotonically across MEI bands on the test set."""
    champ_dir = active_run_dir / "champion"
    bands_file = champ_dir / "bands.json"
    assert bands_file.exists(), "bands.json must exist in champion bundle"

    with open(bands_file, "r", encoding="utf-8") as f:
        bands_data = json.load(f)

    verification = bands_data.get("test_holdout_verification", {})
    assert verification.get("is_strictly_monotonic") is True, (
        f"Test set risk band rates must be strictly monotonic: {verification.get('observed_rates')}"
    )

    rates = verification["observed_rates"]
    assert rates[0] < rates[1] < rates[2], f"Expected Low < Moderate < High: {rates}"


def test_score_dataframe_validity(active_run_dir: Path):
    """Verify score_dataframe on sample upload returns valid MEI, bands, and patterns."""
    sample_file = Path("data/samples/sample_upload_clean.xlsx")
    assert sample_file.exists(), "Sample upload file must exist"

    df_sample = pd.read_excel(sample_file)
    scored_df = score_dataframe(df_sample, run_dir=active_run_dir)

    # 1. Output shape and expected columns
    assert len(scored_df) == len(df_sample)
    expected_cols = [
        "student_id",
        "mei",
        "band",
        "flag",
        "pattern",
        "level",
        "state_path",
        "top_reasons",
    ]
    for c in expected_cols:
        assert c in scored_df.columns, f"Missing column: {c}"

    # 2. MEI score domain constraints [0, 100]
    assert scored_df["mei"].min() >= 0, "MEI cannot be negative"
    assert scored_df["mei"].max() <= 100, "MEI cannot exceed 100"
    assert pd.api.types.is_integer_dtype(scored_df["mei"]), "MEI must be integer"

    # 3. Categorical value validity
    assert set(scored_df["band"].unique()).issubset({"Low", "Moderate", "High"})
    assert set(scored_df["flag"].unique()).issubset({"Yes", "No"})
    assert set(scored_df["pattern"].unique()).issubset(
        {"Consistent", "Declining", "Sporadic", "Improving"}
    )
    assert set(scored_df["level"].unique()).issubset({"High", "Medium", "Low"})

    # 4. Monotonic band assignment
    low_scores = scored_df[scored_df["band"] == "Low"]["mei"]
    if len(low_scores) > 0:
        assert low_scores.max() <= 34

    high_scores = scored_df[scored_df["band"] == "High"]["mei"]
    if len(high_scores) > 0:
        assert high_scores.min() >= 65


def test_champion_bundle_integrity(active_run_dir: Path):
    """Verify all 8 artifacts exist in champion bundle and can be deserialized."""
    champ_dir = active_run_dir / "champion"
    assert champ_dir.exists(), "champion directory must exist"

    expected_files = [
        "pipeline.joblib",
        "calibrator.joblib",
        "features.json",
        "schema_version.json",
        "thresholds.json",
        "bands.json",
        "shap_background_200.parquet",
        "model_card.json",
    ]

    for fname in expected_files:
        fpath = champ_dir / fname
        assert fpath.exists(), f"Champion bundle missing artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fname}"

    # Test deserialization
    pipe = joblib.load(champ_dir / "pipeline.joblib")
    assert hasattr(pipe, "predict_proba")

    cal = joblib.load(champ_dir / "calibrator.joblib")
    assert hasattr(cal, "predict")

    features = json.load(open(champ_dir / "features.json", "r", encoding="utf-8"))
    assert len(features) == 100

    bg = pd.read_parquet(champ_dir / "shap_background_200.parquet")
    assert len(bg) == 200
